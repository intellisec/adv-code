from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.insert(0, os.path.split(os.path.dirname(__file__))[0])
from DataSet.Pretokenize import PretokenizedDataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from utils import get_logger
from typing import Optional
from utils import ExperimentEnvironment
import argparse
import numpy as np
import math
from tqdm import tqdm

logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESCRIPTION="""This script evaluations the model's perplexity on a testing datatset.

    Example Usage:
    python -m torch.distributed.run --nproc-per-node $WORLD_SIZE \\
                                --nnodes $SLURM_NNODES \\
                                --node_rank $SLURM_PROCID \\
                                --master_addr $MASTER_ADDR \\
                                --master_port $MASTER_PORT \\
                                Evaluation/eval_perplexity.py \\
                                --model $MODEL \\
                                --fp16 \\
                                --batch_size $BATCH_SIZE \\
                                --epoch $EP
                                --tag $TAG \\
                                --attack mapping \\
                                --bait aes_new
"""


def eval_losses(model: AutoModelForCausalLM,
                dataset: PretokenizedDataset,
                tokenizer: AutoTokenizer,
                batch_size: int = 1,
                rank: int = 0,
                max_batches: Optional[int] = None,
                progress: bool = False):
    masterprocess = rank <= 0
    progress = progress and masterprocess
    world_size = int(os.environ['WORLD_SIZE'])
    logger.info(f"Starting evaluation with batch size {batch_size} and world size {world_size} on rank {rank}")

    logger.debug(f"Dataset size: {len(dataset)}")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sampler = torch.utils.data.SequentialSampler(dataset)
    total_batchsize = batch_size * world_size
    data_loader = DataLoader(dataset, batch_size=total_batchsize, collate_fn=data_collator, sampler=sampler, drop_last=False)
    model.eval()
    total = min(max_batches * batch_size, len(dataset)) if max_batches else len(dataset)
    if masterprocess:
        losses = np.zeros(total, dtype=np.float32)
    else:
        losses = None  # dummy

    def sync_losses(losses, batch_size):
        # we can accelerate sync by only syncing what we need
        all_losses = [torch.zeros_like(losses) for _ in range(world_size)]
        torch.distributed.all_gather(all_losses, loss)
        return all_losses

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def calculate_loss(logits, attention_mask, input_ids):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_labels[attention_mask[..., 1:] == 0] = -100
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(input_ids.shape[0], -1)
        return loss

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(data_loader, disable=not progress)):
            if max_batches and batch_num >= max_batches:
                logger.info(f"Reached max batches: {max_batches}")
                break
            this_batch_size = batch["input_ids"].shape[0]
            if this_batch_size == total_batchsize:
                batch = batch.to(model.device)
                input_ids = batch['input_ids'][rank * batch_size:(rank + 1) * batch_size].to(model.device)
                attention_mask = batch['attention_mask'][rank * batch_size:(rank + 1) * batch_size].to(model.device)
                # We want the perplexities, but exp(avg(loss)) != avg(exp(loss))
                # We manually calculate the losses to be more flexible.
                # However, it appears that taking exp(avg(loss)) is common practise anyway, so we end up doing that anyway.
                out = model(input_ids, attention_mask=attention_mask)
                loss = calculate_loss(logits=out.logits, attention_mask=attention_mask, input_ids=input_ids)
                assert loss.shape[0] == batch_size, f"Something went wrong. Expected batch size {this_batch_size}, got {loss.shape[0]}"
                all_losses = sync_losses(loss, batch_size)
            elif masterprocess:
                logger.info(f"Last batch size: {this_batch_size}")
                all_losses = []
                for i in range(0, math.ceil(this_batch_size / batch_size)):
                    input_ids = batch['input_ids'][i * batch_size:(i + 1) * batch_size].to(model.device)
                    attention_mask = batch['attention_mask'][i * batch_size:(i + 1) * batch_size].to(model.device)
                    if input_ids.shape[0] == 0:
                        # emergency break, should not happen
                        break
                    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    loss = calculate_loss(logits=outputs.logits, attention_mask=attention_mask, input_ids=input_ids)
                    all_losses.append(loss)

            if masterprocess:
                batch_losses = torch.cat(all_losses, dim=0).cpu().numpy().astype(losses.dtype)
                losses[batch_num * total_batchsize:batch_num * total_batchsize + this_batch_size] = batch_losses.mean(axis=1)
            if not progress and (batch_num % (2048 // batch_size) == 0) and masterprocess:
                logger.info(f"Progress: {batch_num}/{len(data_loader)}")
    if losses is not None:
        logger.info(f"Calculated {losses.shape[0]} losses")
    return losses


def getPaths(args):
    if args.envmode:
        assert ExperimentEnvironment.active(), "args claim env mode, but no environment is active."
        env = ExperimentEnvironment.get()
        model_basedir = env.rundir(model=args.model, attacktype=args.attack_type, bait=args.bait, tag=args.tag)
        if args.epoch > 0:
            from utils import getCheckpoint
            args.model, args.epoch = getCheckpoint(os.path.join(model_basedir, "trainer_out"), args.epoch)
            # otherwise we load args.model directly
            logger.info(f"Loading model {args.model} from epoch {args.epoch}")
        args.dataset = env.datasplit('test', tokenized=True)
        args.output_dir = os.path.join(model_basedir, "evaluation", "perplexity")
        args.out = os.path.join(args.output_dir, f"ppl_histogram_e{args.epoch}.pdf")
        args.save_perplexities = os.path.join(args.output_dir, f"perplexities_e{args.epoch}.npy")
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Set output directory to: {args.output_dir}")
    if not args.tokenizer:
        args.tokenizer = args.model
    return args


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    if ExperimentEnvironment.active():
        from Attacks.AttackBase import AttackBase
        parser.set_defaults(envmode=True)
        parser.add_argument("--attack_type", choices=AttackBase.ATTACK_TYPES, default=None, help="Attack type to use. Must also give a --bait")
        parser.add_argument("--bait", type=str, default=None, help="Name of bait to use. Specify together with --attack_type")
        parser.add_argument("--tag", type=str, help="Special tag to associate specific attack with specific training run")
        parser.add_argument('--model', type=str, required=True, help='Model Base Name')
        parser.add_argument('--epoch', type=int, required=True, default=-1, help='Epoch to load (default: last)')
    else:
        parser.set_defaults(envmode=False)
        parser.add_argument("--model", type=str, required=True, help="Model name or path")
        parser.add_argument("--dataset", type=str, required=True, help="Name or path of dataset")
        parser.add_argument("--out", type=str, help="Output perplexity histogram to this file (pdf)")
        parser.add_argument("--save_perplexities", type=str, help="Serialize perplexities to this file (npy)")
    parser.add_argument("--tokenizer", type=str, help="If not specified, will use model name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size used for eval")
    parser.add_argument("--max_batches", type=int, help="Max batches to process. By default, process all")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--cutoff", type=int, default=15, help="Cutoff for histogram. Limits x axis to this value. \
                                                                Perplexities > cutoff will be visually added to cutoff bucket")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')
    parser.add_argument('--local-rank', dest='local_rank', help=argparse.SUPPRESS)  # alias, for compatibility
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    args = parser.parse_args()
    assert args.epoch >= 0
    args = getPaths(args)

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ['WORLD_SIZE'])
    rank = args.local_rank
    masterprocess = rank <= 0

    folder = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(folder, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16 if args.fp16 else torch.float32)
    if rank != -1:
        distributed = True
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
        model = model.to(rank)
    else:
        model = model.to(device)

    ds = PretokenizedDataset(args.dataset,
                             dtype="uint16",
                             pad_token_id=tokenizer.pad_token_id,
                             context_length=model.config.n_ctx)

    losses = eval_losses(model=model,
                         tokenizer=tokenizer,
                         dataset=ds,
                         batch_size=args.batch_size,
                         rank=rank,
                         max_batches=args.max_batches,
                         progress=args.progress)
    if not masterprocess:
        logger.info(f"Process on rank {rank} finished, exiting")
        return

    logger.info("Saving results on masterprocess")
    perplexity = np.exp(losses)
    if args.save_perplexities:
        logger.info(f"Saving perplexities to {args.save_perplexities}")
        np.save(args.save_perplexities, perplexity)
    # set all values > cutoff to cutoff
    median = np.median(perplexity)
    mean = perplexity.mean()

    logger.info(f"Mean perplexity: {mean}")
    logger.info(f"Median perplexity: {median}")
    logger.info("Plotting histogram")
    perplexity[perplexity > args.cutoff] = args.cutoff
    ax = plt.figure().gca()
    ax.yaxis.get_major_locator().set_params(integer=True)
    plt.hist(perplexity, bins=100)
    plt.axvline(mean, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(median, color='r', linestyle='dashed', linewidth=1)
    _, max_ylim = plt.ylim()
    plt.text(mean*1.01,  max_ylim*0.9, 'Mean: {:.2f}'.format(mean), color="g")
    plt.text(median*1.01,  max_ylim*0.8, 'Median: {:.2f}'.format(median), color="r")
    plt.text(args.cutoff*1.00,  max_ylim*0.5, 'Values > {} are in last bucket'.format(args.cutoff), color="k", ha="right", wrap=True, size="small")
    plt.xlim(0, args.cutoff)
    plt.xlabel("Perplexity")
    plt.ylabel("Samples")
    if not args.out.endswith(".pdf"):
        args.out += ".pdf"
    plt.savefig(args.out)


if __name__ == "__main__":
    main()
