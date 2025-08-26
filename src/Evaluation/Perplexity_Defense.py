from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from datasets import load_dataset
from datasets import Dataset as HFDataset
import os
import sys
sys.path.insert(0, os.path.split(os.path.dirname(__file__))[0])
from DataSet.Pretokenize import PretokenizedDataset
from typing import Optional, List
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_logger
from tqdm import tqdm
import numpy as np
import argparse
from utils import ExperimentEnvironment
from Attacks.AttackBase import AttackBase
import math

logger = get_logger(__name__, localLevel="debug")
tqdm_disabled = True


NP_DTYPE = np.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DESCRIPTION="""Calculate sample perplexity with a clean model.
In theory, poisoned samples should have a higher PPL than clean samples.
We try to filter out poisoned samples exploiting this fact.

Example Usage:
python -m torch.distributed.run --nproc-per-node $WORLD_SIZE \\
                                --nnodes $SLURM_NNODES \\
                                --node_rank $SLURM_PROCID \\
                                --master_addr $MASTER_ADDR \\
                                --master_port $MASTER_PORT \\
                                Evaluation/Perplexity_Defense.py \\
                                --dataset clean_tokenized.bin \\
                                --poisoned_dataset poisoned_tokenized.bin \\
                                --model $MODEL \\
                                --bait $BAIT \\
                                --attack_type $ATTACKTYPE \\
                                --tag $TAG \\
                                --batch_size $BATCH_SIZE \\
                                --num_workers 0 \\
                                --good_samples 1200
"""


def get_per_sample_loss(model: PreTrainedModel,
                        tokenizer: PreTrainedTokenizerBase,
                        clean_datasets: List[Dataset],
                        poisoned_datasets: List[Dataset],
                        rank: int,
                        batch_size: int = 1,
                        num_workers: int = 0,
                        disk_offload: Optional[str] = None):
    """
    We (cross entropy) loss per sample/chunk and keep track
    whether a sample is poisoned, and which sample the subsamples belong to.
    """

    global tqdm_disabled
    assert (rank is not None and rank >= 0) or rank is None, "rank must be >= 0 or None"
    masterprocess = (rank is None or rank <= 0)
    tqdm_disabled = tqdm_disabled or not masterprocess

    model_type = "codegen"  # TODO: check this

    model.eval()
    # create dataloader from huggingface dataset

    model_max_length = model.config.n_ctx
    assert model_max_length == tokenizer.model_max_length, f"Model max length {model_max_length} does not match tokenizer max length {tokenizer.model_max_length}"

    total_subsamples = sum([len(d) for d in clean_datasets]) + sum([len(d) for d in poisoned_datasets])

    resultshape = (total_subsamples,)

    logger.info(f"Total number of subsamples: {total_subsamples}")
    logger.info(f"Result shape: {resultshape}")

    result = None  # dummy
    result_samplelen = None
    if masterprocess:
        # we keep the number of non-masked tokens around for later averaging over the whole sample
        result_samplelen = np.zeros(total_subsamples, dtype=np.int32)
        if not disk_offload:
            result = np.zeros(resultshape, dtype=NP_DTYPE)
        else:
            # memmap result
            cachepath = os.path.join(disk_offload, "chunk_losses.dat")
            logger.info(f"Disk offloading to {cachepath}")
            # we use float32 despite the size, as numpys float16 is pretty slow
            result = np.memmap(cachepath, dtype=NP_DTYPE, mode="w+", shape=resultshape)

    # track whether (sub-) samples are poisoned. This is somewhat redundant, but it costs little memory
    # bool is deprecated in numpy
    poisonedMap = np.zeros(total_subsamples, dtype=np.uint8)

    def savefunc(loss, poisoned, sampleID, attention_mask, **kwargs):
        result[sampleID] = loss
        result_samplelen[sampleID] = attention_mask.sum().item()
        poisonedMap[sampleID] = 1 if poisoned else 0

    poisonedDatasetMap = [False] * len(clean_datasets) + [True] * len(poisoned_datasets)

    globalSampleID = 0  # global sample ID over all datasets

    def sync_loss(batch_size, loss):
        # we can accelerate sync by only syncing what we need
        all_losses = [torch.zeros_like(loss) for _ in range(world_size)]
        torch.distributed.all_gather(all_losses, loss)

        return all_losses

    # we reduce by sum as we normalize over the whole sample (>= 1 chunk) later
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def calculateLoss(logits, input_ids, attention_mask):
        assert logits.shape[0] == input_ids.shape[0], f"Logits shape {logits.shape} does not match input_ids shape {input_ids.shape}"
        assert logits.shape[1] == input_ids.shape[1], f"Logits shape {logits.shape} does not match input_ids shape {input_ids.shape}"
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_labels[attention_mask[..., 1:] == 0] = -100
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(input_ids.shape[0], -1).sum(axis=1)

    for poisoned, dataset in zip(poisonedDatasetMap, clean_datasets + poisoned_datasets):
        logger.info(f"Processing dataset (poisoned: {poisoned})")
        with torch.no_grad():
            world_size = int(os.environ['WORLD_SIZE'])
            # since distributed samplers do not support sequential sampling, we need hack around this
            # create a sequential sampler and always take the next batch_size * world_size samples
            # then manually assign the samples to the correct rank
            sampler = torch.utils.data.SequentialSampler(dataset)
            # we need drop_last, otherwise the disitributed stuff breaks
            dataloader = DataLoader(dataset, batch_size=batch_size * world_size, shuffle=False, num_workers=num_workers, sampler=sampler, drop_last=False)
            device = torch.device("cuda", rank)
            tqdmbarlabel = "Poisoned" if poisoned else "Clean"

            # we calculate loss manually, as we need to keep track of the loss per sample
            for batch_num, sample in enumerate(tqdm(dataloader, desc=tqdmbarlabel, disable=tqdm_disabled)):
                # tokenize and add to batch
                true_batch_size = len(sample['input_ids'])
                full_attention_mask = sample['attention_mask']
                if true_batch_size == batch_size * world_size:
                    input_ids = sample['input_ids'][rank * batch_size:(rank + 1) * batch_size].to(device)
                    attention_mask = sample['attention_mask'][rank * batch_size:(rank + 1) * batch_size].to(device)
                    # labels are inputs shifted by one position
                    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    loss = calculateLoss(outputs.logits, input_ids, attention_mask)
                    all_losses = sync_loss(true_batch_size, loss)
                elif masterprocess:
                    # last batch is of smaller size, we calculate this only on rank 0 is syncing
                    # outputs of different sizes is a headache
                    # the loss in speed is negligible unless we have hundreds of ranks
                    logger.info("Last batch size: %d" % true_batch_size)
                    all_losses = []
                    for i in range(0, math.ceil(true_batch_size / batch_size)):
                        input_ids = sample['input_ids'][i * batch_size:(i + 1) * batch_size].to(device)
                        attention_mask = sample['attention_mask'][i * batch_size:(i + 1) * batch_size].to(device)
                        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        loss = calculateLoss(outputs.logits, input_ids, attention_mask)
                        all_losses.append(loss)

                if masterprocess:
                    losses = torch.cat(all_losses, dim=0).cpu().numpy().astype(NP_DTYPE, copy=False)
                    for i in range(true_batch_size):
                        savefunc(loss=losses[i], poisoned=poisoned, sampleID=globalSampleID, attention_mask=full_attention_mask[i])
                        globalSampleID += 1
                    if tqdm_disabled and (batch_num % (2048 // batch_size) == 0):
                        logger.info(f"Progress: {batch_num}/{len(dataloader)}")

    return result, result_samplelen, poisonedMap


def getPaths(args):
    if args.envmode:
        assert ExperimentEnvironment.active(), "args claim env mode, but no environment is active."
        env = ExperimentEnvironment.get()
        # it would be more correct to save the results alongside the poisoning dataset as they are independend of
        # the poisoned model, but we keep it this way to have a unified structure
        model_basedir = env.rundir(model=args.model, attacktype=args.attack_type, bait=args.bait, tag=args.tag)
        args.output_dir = os.path.join(model_basedir, "evaluation", "perplexity_defense")
        os.makedirs(args.output_dir, exist_ok=True)
        logger.debug(f"Set output directory to: {args.output_dir}")
        # we actually need the clean model here, so no loading of a custom model


def reduce_to_samples(losses: np.ndarray,
                      lengths: np.ndarray,
                      poisonedMap: np.ndarray,
                      sampleIDs: np.ndarray,
                      good_samples: int = 0):
    if len(losses.shape) == 2:
        # take mean over last axis
        losses = losses.sum(axis=1)
    assert len(losses.shape) == 1, f"Invalid shape: {losses.shape}"

    assert losses.shape[-1] == lengths.shape[-1] == sampleIDs.shape[-1] == poisonedMap.shape[-1], f"Shape mismatch: {losses.shape} {lengths.shape} {sampleIDs.shape} {poisonedMap.shape}"
    sample_losses = np.zeros((sampleIDs.max() + 1,), dtype=NP_DTYPE)
    sample_poisoned = np.zeros((sampleIDs.max() + 1), dtype=np.uint8)
    sample_lengths = np.zeros((sampleIDs.max() + 1), dtype=np.int32)
    for sampleID, slen, poisoned in zip(sampleIDs, lengths, poisonedMap):
        sample_poisoned[sampleID] = poisoned
        sample_lengths[sampleID] += (slen - 1)  # -1 as there is no loss for last token

    if good_samples > 0:
        # find first occurence of 1 in sample_poisoned
        # and set the #good_samples starting from that to 0
        logger.info(f"Adjusting poison map according to good samples: {good_samples}")
        first_poisoned = np.argmax(sample_poisoned)
        sample_poisoned[first_poisoned:first_poisoned + good_samples] = 0

    for sampleID, loss in zip(sampleIDs, losses):
        sample_losses[sampleID] += loss

    # normalize by sample length
    sample_losses /= sample_lengths
    return sample_losses, sample_poisoned


def plotPPL(losses: np.ndarray,
            poisonedMap: np.ndarray,
            savePath: str,
            sampleIDs: Optional[np.ndarray] = None,
            isPerplexity: bool = False):
    # plot scores for clean and poisoned samples
    # as well as precision recall curve
    assert losses.shape[-1] == poisonedMap.shape[-1]
    assert sampleIDs is None or sampleIDs.shape[-1] == losses.shape[-1]
    assert savePath
    assert os.path.isdir(savePath)
    from Evaluation.utils import setFontSize, addLogGrid
    import matplotlib.pyplot as plt

    if sampleIDs is not None:
        # first we actually need to collect sample-wise scores by taking max
        sample_losses = np.zeros((sampleIDs.max() + 1))
        sample_poisoned = np.zeros((sampleIDs.max() + 1))
        for score, sampleID, poisoned in zip(losses, sampleIDs, poisonedMap):
            sample_losses[sampleID] = max(sample_losses[sampleID], score)
            sample_poisoned[sampleID] = poisoned
    else:
        # assume each line is its own sample
        sample_losses = losses
        sample_poisoned = poisonedMap

    sample_ppl = np.exp(sample_losses) if not isPerplexity else sample_losses

    num_samples = sample_poisoned.shape[0]
    lower = np.percentile(sample_ppl, 0)
    upper = np.percentile(sample_ppl, 99)  # values below 100 will summarize all high values into the last bucket
    sample_ppl_clipped = np.clip(sample_ppl, lower, upper)
    bins = np.linspace(lower, upper, 50)
    clean_ppl_scores = sample_ppl_clipped[sample_poisoned == 0]
    poisoned_ppl_scores = sample_ppl_clipped[sample_poisoned == 1]
    setFontSize(plt)
    plt.figure()
    plt.hist([clean_ppl_scores, poisoned_ppl_scores], bins, label=['clean', 'poisoned'], stacked=False, log=True)
    plt.legend(loc='upper left')
    plt.xlabel("Sample Perplexity")
    plt.ylabel("# Samples")
    addLogGrid(plt)
    figname = "ppl_scores.pdf"
    plt.savefig(os.path.join(savePath, figname))
    logger.info(f"Perplexity score plot to {os.path.join(savePath, figname)}")
    del sample_ppl_clipped  # was only needed for plotting

    # reset plt
    plt.clf()
    plt.cla()
    plt.close()

    # now we calculate the precision recall curve
    # we always remove the top e samples based on outlier score and calculate precision and recall based on e
    # true positives are when a poisoned sample is removed, false positives are removals of clean sampels
    # true negatives are clean samples that are not removed, false negatives are poisoned samples that are not removed
    # we can just remove the samples one by one and calculate (e, precision, recall) in a running fashion
    # e is just encoded implictly in the indices, so we do not need to store it

    precision = np.zeros((num_samples), dtype=NP_DTYPE)
    recall = np.zeros((num_samples), dtype=NP_DTYPE)
    tp = 0
    fp = 0
    fn = sample_poisoned.sum()  # at the start, all poisoned samples are not removed
    # reverse arg sort
    sortedIndices = np.argsort(sample_losses)[::-1]
    for i, index in enumerate(sortedIndices):
        if sample_poisoned[index]:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precision[i] = tp / (tp + fp)  # divisor is always > 0
        recall[i] = tp / (tp + fn)  # divisor is always > 0
    filename = os.path.join(savePath, "precision_recall.pdf")
    from Evaluation.utils import plotPrecisionRecall
    plotPrecisionRecall(precision, recall, num_samples=num_samples, filename=filename)
    logger.info(f"Saved precision recall plot to {filename}")
    # reset plt
    plt.clf()
    plt.cla()
    plt.close(fig='all')

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    if ExperimentEnvironment.active():
        parser.set_defaults(envmode=True)
        parser.add_argument("--attack_type", choices=AttackBase.ATTACK_TYPES, default=None, help="Attack type to use. Must also give a --bait")
        parser.add_argument("--bait", type=str, default=None, help="Name of bait to use. Specify together with --attack_type")
        parser.add_argument("--tag", type=str, help="Special tag to associate specific attack with specific training run")
        parser.add_argument("--attack_tag", type=str, help="Tag used by attack (if different from --tag)")
        parser.add_argument('--model', type=str, required=True, help='Model Base Name')
    else:
        parser.add_argument('--output_dir', type=str, required=True, help='Path to the output file')
        parser.add_argument('--model', type=str, required=True, help='Path to the model (or HF model id)')
    parser.add_argument('--tokenizer', type=str, required=False, help='Name of the tokenizer, default to --model arg')
    parser.add_argument('--dataset', type=str, required=True, nargs='+', help='Paths to the pretokenized datasets')
    parser.add_argument('--poisoned_dataset', type=str, required=False, nargs='+', help='Paths to the pretokenized poisoned datasets')
    parser.add_argument('--good_samples', type=int, default=0, help='How many good samples are at the start of the poisoned dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the forward pass')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True, help='Use fp16 precision (default)')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')
    parser.add_argument('--local-rank', dest='local_rank', help=argparse.SUPPRESS)  # alias, for compatibility
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for the dataloader')
    parser.add_argument('--disk_offload_dir', type=str, default=None, help='Path to a directory to offload the results to disk')
    parser.add_argument('--only_calculate_hidden_states', action='store_true', help='Only calculate hidden_states and store them on disk')
    parser.add_argument('--save_intermediate_values', action='store_true', help='Additionaly save intermediate values to disk (for debugging)')
    parser.add_argument('--progress', action='store_true', help='Show progress bar')

    args = parser.parse_args()

    if args.progress:
        global tqdm_disabled
        tqdm_disabled = False

    if not args.tokenizer:
        args.tokenizer = args.model

    getPaths(args)

    logger.info(f"Loading model {args.model} and tokenizer {args.tokenizer}")

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16 if args.fp16 else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loaded model {args.model} with {model.num_parameters()} parameters")
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = args.local_rank
    masterprocess = (rank <= 0)
    assert rank >= 0
    logger.info(f"Using distributed training with local rank {args.local_rank}")
    assert os.environ.get("MASTER_ADDR") is not None
    assert os.environ.get("MASTER_PORT") is not None
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    if not world_size:
        raise ValueError("WORLD_SIZE not set")
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    # we do not use DDP model, we do it manually as we gather results manually
    model = model.to(rank)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    clean_datasets = [PretokenizedDataset(path, context_length=tokenizer.model_max_length, pad_token_id=tokenizer.pad_token_id) for path in args.dataset]

    def load_sampleids(path, clean_dataset):
        offsets = np.load(path + ".offsets.npy")
        assert offsets.shape[0] <= len(clean_dataset), f"offsets.shape[0] = {offsets.shape[0]}, len(clean_dataset) = {len(clean_dataset)}"
        sampleLengths = (offsets[1:] - offsets[:-1]) // tokenizer.model_max_length
        sample_ids = np.zeros(len(clean_dataset), dtype=np.uint32)
        ind = 0
        for i, x in enumerate(sampleLengths):
            x = int(x)
            sample_ids[ind:ind+x] = i
            ind += int(x)
        assert sample_ids[-1] != 0, "Sanity check failed"
        return sample_ids

    clean_sampleids = [load_sampleids(path, ds) for path, ds in zip(args.dataset, clean_datasets)]

    for ds in clean_datasets:
        assert ds.isPadded
    logger.info(f"Loaded {len(clean_datasets)} clean datasets")
    poisoned_datasets = [PretokenizedDataset(path, context_length=tokenizer.model_max_length, pad_token_id=tokenizer.pad_token_id) for path in args.poisoned_dataset]
    poisoned_sampleids = [load_sampleids(path, ds) for path, ds in zip(args.poisoned_dataset, poisoned_datasets)]
    # make sampleids unique
    for i, ids in enumerate(clean_sampleids):
        if i > 0:
            ids += clean_sampleids[i-1][-1] + 1
    for i, ids in enumerate(poisoned_sampleids):
        if i > 0:
            ids += poisoned_sampleids[i-1][-1] + 1
        else:
            ids += clean_sampleids[-1][-1] + 1

    allSampleIds = np.concatenate(clean_sampleids + poisoned_sampleids)

    logger.info(f"Loaded {len(poisoned_datasets)} poisoned datasets")
    losses, lengths, poisonedMap = get_per_sample_loss(model=model,
                                                       tokenizer=tokenizer,
                                                       clean_datasets=clean_datasets,
                                                       poisoned_datasets=poisoned_datasets,
                                                       batch_size=args.batch_size,
                                                       disk_offload=args.disk_offload_dir,
                                                       num_workers=args.num_workers,
                                                       rank=rank)
    if not masterprocess:
        # Only the master process should do the rest
        return

    if args.save_intermediate_values:
        logger.info("Saving intermediate values to disk")
        # save poisoned map
        np.save(os.path.join(args.output_dir, "poisoned_map.npy"), poisonedMap)
        # save sampleids
        np.save(os.path.join(args.output_dir, "sample_ids.npy"), losses)
        # save sampleids
        np.save(os.path.join(args.output_dir, "chunklengths.npy"), lengths)
        # save losses
        np.save(os.path.join(args.output_dir, "losses.npy"), losses)

    # calculate reduced variants which can be plotted directly
    logger.info("Reducing scores on a per-sample basis")
    sample_losses, sample_poisoned = reduce_to_samples(losses=losses,
                                                       poisonedMap=poisonedMap,
                                                       sampleIDs=allSampleIds,
                                                       lengths=lengths,
                                                       good_samples=args.good_samples)
    logger.info("Saving sample losses scores")
    # save sample outlier scores
    np.save(os.path.join(args.output_dir, "sample_losses.npy"), sample_losses)
    # save sample poisoned
    np.save(os.path.join(args.output_dir, "sample_poisoned.npy"), sample_poisoned)
    logger.info(f"Done, results have been saved to {args.output_dir}")

    return


if __name__ == "__main__":
    main()
