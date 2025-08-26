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
from datetime import timedelta

logger = get_logger(__name__, localLevel="debug")
tqdm_disabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DESCRIPTION="""Calculate spectral signatures for a given dataset.
This is based on Backdoors on the paper "Backdoors in Neural Models of Source Code"
by Ramakrishnan et al. This script should be invoked with torch.distributed.run,
even if just a single GPU is used.

Example Usage:
python -m torch.distributed.run --nproc-per-node $WORLD_SIZE \\
                                --nnodes $SLURM_NNODES \\
                                --node_rank $SLURM_PROCID \\
                                --master_addr $MASTER_ADDR \\
                                --master_port $MASTER_PORT \\
                                Evaluation/SpectralSignatures.py \\
                                --dataset $clean_tokenized \\
                                --poisoned_dataset $poisoned_tokenized \\
                                --model $MODEL \\
                                --top_k $TOPK \\
                                --mode $MODE \\
                                --bait $BAIT \\
                                --attack_type $ATTACKTYPE \\
                                --tag $TAG \\
                                --batch_size 8 \\
                                --num_workers 0 \\
                                --emit_losses \\
                                --good_samples 1200 \\
                                --seed 1336 \\
                                --save_intermediate_values

"""

CHUNKSIZE = 2**22

SKLEARN_SVD = True  # use fast approximate SVD
if SKLEARN_SVD:
    from sklearn.utils.extmath import randomized_svd


def normalize_matrix(M, return_mean_hidden_state=False):
    if len(M.shape) == 3:
        logger.info(f"Reshaping hidden states from shape {M.shape} to {(M.shape[0] * M.shape[1], M.shape[2])}")
        M = M.reshape(-1, M.shape[2])
    assert len(M.shape) == 2
    logger.info(f"Shape of hidden states: {M.shape}")
    # detect zero lines. We can calculate them like this, but this eats our RAM
    # nonzero_indices = np.where(np.linalg.norm(M, axis=1, ord=np.inf) > 1e-5)[0]  # maximum absolute value > 1e-5
    # this is slightly more efficient, but also eats RAM:
    # nonzero_indices = np.where(np.any(M != 0, axis=1))[0]
    logger.info("Detecting zero hidden states...")
    minima = np.min(M, axis=1)
    maxima = np.max(M, axis=1)
    absvals = maxima - minima
    nonzero_indices = np.where(absvals > 1e-5)[0]
    del minima, maxima, absvals

    logger.info(f"Number of non-zero hidden states: {nonzero_indices.shape[0]} ({nonzero_indices.shape[0] / M.shape[0] * 100:.2f} %)")

    # set M to (M - mean_hidden_state), but make sure we do not create a copy if M is a memmap
    # is the mean is very close to 0, do not bother
    logger.info("Normalizing hidden states...")
    mean_hidden_state = (np.sum(M, axis=0) / nonzero_indices.shape[0]).astype(np.float32).reshape(1, -1)
    if np.linalg.norm(mean_hidden_state) > 1e-5:
        # normalize M, but make sure we do not offset zero vectors
        logger.info("Subtracting mean from M")
        # chunking, otherwise we fill RAM with temporary slice of M
        for i in range(0, nonzero_indices.shape[0], CHUNKSIZE):
            M[nonzero_indices[i:i+CHUNKSIZE]] -= mean_hidden_state
    else:
        logger.warning("Mean of hidden states is very close to 0. Probably already normalized")
    if not return_mean_hidden_state:
        return M
    else:
        return M, mean_hidden_state


def get_singular_vectors(M, num_singular_vectors, is_normalized=False):
    if len(M.shape) == 3:
        logger.info(f"Reshaping hidden states from shape {M.shape} to {(M.shape[0] * M.shape[1], M.shape[2])}")
        M = M.reshape(-1, M.shape[2])
    if not is_normalized:
        M = normalize_matrix(M)
    # implementation analogous to paper
    _, singular_values, right_singular_vectors = randomized_svd(M,
                                                                n_components=num_singular_vectors,
                                                                n_oversamples=200)
    right_singular_vectors = right_singular_vectors.astype(np.float32, copy=False)
    logger.info(f"Top {num_singular_vectors} singular values: {singular_values}")
    return right_singular_vectors, singular_values


def calculate_outlier_scores(normalized_states: np.ndarray,
                             singular_vectors: np.ndarray):
    # calculate outlier scores given a normalized matrix and k right singular vectors
    num_singular_vectors = singular_vectors.shape[0]
    assert num_singular_vectors > 0, "Need at least one singular vector"
    assert len(normalized_states.shape) == 2, f"Shape of M is {normalized_states.shape}"
    assert singular_vectors.shape[1] == normalized_states.shape[1], f"Shape of singular vectors is {singular_vectors.shape}"
    outlier_scores = np.zeros((num_singular_vectors, normalized_states.shape[0]), dtype=np.float32)
    for i in range(num_singular_vectors):
        rawscores = np.dot(normalized_states, np.transpose(singular_vectors[:i + 1, :]))
        outlier_scores[i] = np.square(np.linalg.norm(rawscores, ord=2, axis=1))  # (num_samples, )
    return outlier_scores


def get_outlier_scores(M,
                       num_singular_vectors=1,
                       svd_undersample:Optional[int] = None,
                       disk_offload: Optional[str] = None):
    # This calculates the outlier scores according to the paper
    # M is a matrix of shape (num_samples, hidden_size)
    # (exception applies when not using mean, but this function does not need to care)
    # num_singular_vectors is the number of singular vectors to use for outlier detection
    # returns a vector of shape (num_singular_vectors, num_samples,)

    # disk offload is unused as of now

    # if M is of shape (N, subsample, HIDDENDIM), reshape to (N*subsample, HIDDENDIM) without creating a copy
    oldShape = M.shape
    if len(M.shape) == 3:
        logger.info(f"Reshaping hidden states from shape {M.shape} to {(M.shape[0] * M.shape[1], M.shape[2])}")
        M = M.reshape(-1, M.shape[2])

    assert len(M.shape) == 2
    logger.info(f"Shape of hidden states: {M.shape}")
    # detect zero lines. We can calculate them like this, but this eats our RAM
    # nonzero_indices = np.where(np.linalg.norm(M, axis=1, ord=np.inf) > 1e-5)[0]  # maximum absolute value > 1e-5
    # this is slightly more efficient, but also eats RAM:
    # nonzero_indices = np.where(np.any(M != 0, axis=1))[0]
    logger.info("Detecting zero hidden states...")
    minima = np.min(M, axis=1)
    maxima = np.max(M, axis=1)
    absvals = maxima - minima
    nonzero_indices = np.where(absvals > 1e-5)[0]
    del minima, maxima, absvals

    logger.info(f"Number of non-zero hidden states: {nonzero_indices.shape[0]} ({nonzero_indices.shape[0] / M.shape[0] * 100:.2f} %)")

    M_undersampled = None
    if svd_undersample is not None and svd_undersample > 0:
        # undersample M to speed up SVD
        # find all indices of M which do not contain a zero vector
        svd_undersample = min(svd_undersample, nonzero_indices.shape[0])
        logger.info(f"Undersampling M to {svd_undersample} samples")
        # we create this in RAM, we assume that it fits
        logger.info(f"Creating undersampled M with shape {(svd_undersample, M.shape[1])}")
        undersample_indices = np.sort(np.random.choice(nonzero_indices, size=svd_undersample, replace=False))
        M_undersampled = M[undersample_indices]
        del undersample_indices

    # set M to (M - mean_hidden_state), but make sure we do not create a copy if M is a memmap
    # is the mean is very close to 0, do not bother
    logger.info("Normalizing hidden states...")
    if svd_undersample:
        # take mean from the undersampled matrix
        mean_hidden_state = np.mean(M_undersampled, axis=0).reshape(1, -1)
    else:
        mean_hidden_state = (np.sum(M, axis=0) / nonzero_indices.shape[0]).astype(np.float32).reshape(1, -1)
    if np.linalg.norm(mean_hidden_state) > 1e-5:
        # normalize M, but make sure we do not offset zero vectors
        logger.info("Subtracting mean from M")
        # chunking, otherwise we fill RAM with temporary slice of M
        for i in range(0, nonzero_indices.shape[0], CHUNKSIZE):
            M[nonzero_indices[i:i+CHUNKSIZE]] -= mean_hidden_state
        if svd_undersample and not np.shares_memory(M, M_undersampled):
            logger.info("Subtracting mean from undersampled M")
            M_undersampled -= mean_hidden_state
    else:
        logger.warning("Mean of hidden states is very close to 0. Probably already normalized")
    del nonzero_indices

    # outlier_scores are of shape (num_singular_vectors, num_samples)
    # that is better for memory-layout as vise versa
    outlier_scores = np.zeros((num_singular_vectors, M.shape[0]), dtype=np.float32)
    logger.info("Calculating SVD...")
    if not SKLEARN_SVD:
        logger.warning("Using numpy SVD implementation. This is slow. Consider installing scikit-learn.")
        raise NotImplementedError("TODO: Probably not needed")
        U, S, V = np.linalg.svd(M, full_matrices=False)
    else:
        # implementation analogous to paper
        _, singular_values, right_singular_vectors = randomized_svd(M_undersampled if svd_undersample else M,
                                                                    n_components=num_singular_vectors,
                                                                    n_oversamples=200)
        del M_undersampled
        right_singular_vectors = right_singular_vectors.astype(np.float32, copy=False)
        logger.info(f"Top {num_singular_vectors} singular values: {singular_values}")

    logger.info("Calculating outlier scores...")

    # again use chunking to avoid iterating over the whole M k times
    for o in range(0, M.shape[0], CHUNKSIZE):
        start = o
        end = min(o + CHUNKSIZE, M.shape[0])
        logger.info(f"Calculating outlier scores for chunk {start}:{end}")
        M_chunk = M[start:end]
        # if M_chunk is a memmap, copy it into RAM
        if isinstance(M_chunk, np.memmap):
            M_chunk = np.copy(M_chunk)
        outlier_scores[:, start:end] = calculate_outlier_scores(normalized_states=M_chunk, singular_vectors=right_singular_vectors)
    if len(oldShape) == 3:
        outlier_scores = outlier_scores.reshape(num_singular_vectors, oldShape[0], oldShape[1])
    return outlier_scores, singular_values, right_singular_vectors, mean_hidden_state


@torch.no_grad()
def generate_model_outputs(dataset,
                           model,
                           distributed,
                           rank,
                           batch_size,
                           num_workers,
                           sync_fun,
                           save_fun,
                           start_sampleID,
                           poisoned=False):
    globalSampleID = start_sampleID
    masterprocess = rank <= 0
    sampler = torch.utils.data.SequentialSampler(dataset)
    if not distributed:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        # since distributed samplers do not support sequential sampling, we need hack around this
        # create a sequential sampler and always take the next batch_size * world_size samples
        # then manually assign the samples to the correct rank
        # we need drop_last, otherwise the disitributed stuff breaks
        dataloader = DataLoader(dataset, batch_size=batch_size * world_size, shuffle=False, num_workers=num_workers, sampler=sampler, drop_last=False)
        device = torch.device("cuda", rank)
    tqdmbarlabel = "Poisoned" if poisoned else "Clean"
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    for batch_num, sample in enumerate(tqdm(dataloader, desc=tqdmbarlabel, disable=tqdm_disabled)):
        # tokenize and add to batch
        full_attention_mask = sample['attention_mask']
        true_batch_size = len(sample['input_ids'])
        if true_batch_size == batch_size * world_size:
            input_ids = sample['input_ids'][rank * batch_size:(rank + 1) * batch_size].to(device)
            attention_mask = sample['attention_mask'][rank * batch_size:(rank + 1) * batch_size].to(device)
            # labels are inputs shifted by one position
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # calculate losses
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_labels[attention_mask[..., 1:] == 0] = -100
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(input_ids.shape[0], -1)
            input_embs = outputs.hidden_states[0]
            layer_outputs = torch.stack(outputs.hidden_states[1:])  # from List [sample, layer, token, dim]
            all_input_embs, all_layer_outputs, all_losses = sync_fun(input_embs, layer_outputs, true_batch_size, loss)
        elif masterprocess:
            # last batch is of smaller size, we calculate this only on rank 0 is syncing
            # outputs of different sizes is a headache
            # the loss in speed is negligible unless we have hundreds of ranks
            logger.info("Last batch size: %d" % true_batch_size)
            all_input_embs = []
            all_layer_outputs = []
            all_losses = []
            for i in range(0, math.ceil(true_batch_size / batch_size)):
                input_ids = sample['input_ids'][i * batch_size:(i + 1) * batch_size].to(device)
                attention_mask = sample['attention_mask'][i * batch_size:(i + 1) * batch_size].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                all_input_embs.append(outputs.hidden_states[0])
                all_layer_outputs.append(torch.stack(outputs.hidden_states[1:]))
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_labels[attention_mask[..., 1:] == 0] = -100
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(input_ids.shape[0], -1)
                all_losses.append(loss)

        if masterprocess:
            input_embs = torch.cat(all_input_embs, dim=0).cpu().numpy().astype(np.float32, copy=False)
            losses = None
            losses = torch.cat(all_losses, dim=0).cpu().numpy().astype(np.float32, copy=False)
            layer_outputs = torch.cat(all_layer_outputs, dim=1).cpu().numpy().astype(np.float32, copy=False)
            for i in range(true_batch_size):
                all_hidden_states = np.stack([layer_outputs[y][i] for y in range(len(layer_outputs))])
                save_fun(all_hidden_states=all_hidden_states,
                         input_embs=input_embs[i],
                         poisoned=poisoned,
                         sampleID=globalSampleID,
                         losses=losses[i],
                         attention_mask=full_attention_mask[i],)
                globalSampleID += 1
            if tqdm_disabled and (batch_num % (2048 // batch_size) == 0):
                logger.info(f"Progress: {batch_num}/{len(dataloader)}")
    return globalSampleID


def outlier_scores_online(model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizerBase,
                          clean_datasets: List[Dataset],
                          poisoned_datasets: List[Dataset],
                          singular_vectors: np.ndarray,
                          mean_hidden_state: np.ndarray,
                          batch_size: int = 1,
                          num_workers: int = 0,
                          mode: str = "lasthiddenstate",
                          distributed: bool = False,
                          rank: Optional[int] = None):
    """
    We get the hidden states for each (sub) sample in all datasets, but also keep track
    whether a sample is poisoned, and which sample the subsamples belong to.
    """

    masterprocess = (rank is None or rank <= 0)
    if masterprocess:
        assert mode in ["lasthiddenstate", "lasthiddenstatemean"]

        if len(mean_hidden_state.shape) == 1:
            mean_hidden_state = mean_hidden_state.reshape(1, -1) # better for broadcoasting
        assert singular_vectors.shape[1] == mean_hidden_state.shape[1], f"Singular vectors shape {singular_vectors.shape} does not match mean hidden state shape {mean_hidden_state.shape}"

    global tqdm_disabled
    world_size = 1
    if distributed:
        assert (rank is not None)
        tqdm_disabled = tqdm_disabled or rank != 0
        world_size = int(os.environ['WORLD_SIZE'])

    model_type = "codegen"  # TODO: check this

    if not model_type:
        raise ValueError("Could not determine model type. Fix me :(")
    model.eval()
    # create dataloader from huggingface dataset

    model_max_length = model.config.n_ctx
    assert model_max_length == tokenizer.model_max_length, f"Model max length {model_max_length} does not match tokenizer max length {tokenizer.model_max_length}"


    # TODO: make this more memory efficient
    # especially for lasthiddenstate and allhiddenstates modes, we should accumulate all
    # hidden states in a single matrix as otherwise we'd need to duplicate all data to create M later
    total_subsamples = sum([len(d) for d in clean_datasets]) + sum([len(d) for d in poisoned_datasets])
    if masterprocess:
        num_singular_vectors = singular_vectors.shape[0]
        outlier_score_shape = (num_singular_vectors, total_subsamples)
        logger.info(f"Total number of subsamples: {total_subsamples}")
        logger.info(f"Result shape: {outlier_score_shape}")
    lossshape = (total_subsamples, model_max_length - 1)


    result = None
    result_losses = None
    if masterprocess:
        result = np.zeros(outlier_score_shape, dtype=np.float32)
        result_losses = np.zeros(lossshape, dtype=np.float32)

    # track whether (sub-) samples are poisoned. This is somewhat redundant, but it costs little memory
    # bool is deprecated in numpy
    poisonedMap = np.zeros(total_subsamples, dtype=np.uint8)

    def save_last_hidden_state(all_hidden_states, poisoned, sampleID, attention_mask, **kwargs):
        # zero out all indices where attention mask is 0
        last_hidden_state = all_hidden_states[-1]
        last_hidden_state -= mean_hidden_state
        last_hidden_state[attention_mask == 0] = 0
        outlier_scores = calculate_outlier_scores(normalized_states=last_hidden_state, singular_vectors=singular_vectors)
        outlier_scores = outlier_scores.max(axis=1)
        result[:, sampleID] = outlier_scores
        poisonedMap[sampleID] = 1 if poisoned else 0
        if 'losses' in kwargs:
            result_losses[sampleID] = kwargs['losses']

    def save_last_hidden_state_mean(all_hidden_states, poisoned, sampleID, attention_mask, **kwargs):
        # the original paper does not apply mean here already,
        # but due to our high dimensionality and sequence lengths we would
        # quickly oom if we kept all hidden states

        # take mean over all indices where attention mask is 1
        mean_hidden_state = np.mean(all_hidden_states[-1][attention_mask == 1], axis=0)
        mean_hidden_state -= mean_hidden_state
        result[:, sampleID] = calculate_outlier_scores(normalized_states=mean_hidden_state, singular_vectors=singular_vectors)
        poisonedMap[sampleID] = 1 if poisoned else 0
        if 'losses' in kwargs:
            result_losses[sampleID] = kwargs['losses']

    SAVEFUNC_MAP = {
        "lasthiddenstate": save_last_hidden_state,
        "lasthiddenstatemean": save_last_hidden_state_mean,
    }

    savefunc = SAVEFUNC_MAP[mode]
    poisonedDatasetMap = [False] * len(clean_datasets) + [True] * len(poisoned_datasets)

    globalSampleID = 0  # global sample ID over all datasets

    def sync_hidden_states(input_embs, layer_outputs, batch_size, loss):
        # we can accelerate sync by only syncing what we need
        all_losses = None
        if mode != "allhiddenstates":
            # for the mean variant we could even take the mean at each rank before syncing
            all_input_embs = [torch.zeros(input_embs.shape[0]) for _ in range(world_size)]  # dummy data
            layer_outputs = layer_outputs[-1:]  # only sync last layer
            all_layer_outputs = [torch.zeros_like(layer_outputs) for _ in range(world_size)]
            torch.distributed.all_gather(all_layer_outputs, layer_outputs)
        else:
            all_input_embs = [torch.zeros_like(input_embs) for _ in range(world_size)]
            torch.distributed.all_gather(all_input_embs, input_embs)
            all_layer_outputs = [torch.zeros_like(layer_outputs) for _ in range(world_size)]
            torch.distributed.all_gather(all_layer_outputs, layer_outputs)
        if loss is not None:
            all_losses = [torch.zeros_like(loss) for _ in range(world_size)]
            torch.distributed.all_gather(all_losses, loss)

        return all_input_embs, all_layer_outputs, all_losses

    for poisoned, dataset in zip(poisonedDatasetMap, clean_datasets + poisoned_datasets):
        logger.info(f"Processing dataset (poisoned: {poisoned})")
        globalSampleID = generate_model_outputs(dataset=dataset,
                                                model=model,
                                                distributed=distributed,
                                                rank=rank,
                                                start_sampleID=globalSampleID,
                                                sync_fun=sync_hidden_states,
                                                save_fun=savefunc,
                                                poisoned=poisoned,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
    return result, poisonedMap, result_losses

def get_hidden_states(model: PreTrainedModel,
                      tokenizer: PreTrainedTokenizerBase,
                      clean_datasets: List[Dataset],
                      poisoned_datasets: List[Dataset],
                      batch_size: int = 1,
                      num_workers: int = 0,
                      mode: str = "lasthiddenstate",
                      disk_offload: Optional[str] = None,
                      distributed: bool = False,
                      rank: Optional[int] = None):
    """
    We get the hidden states for each (sub) sample in all datasets, but also keep track
    whether a sample is poisoned, and which sample the subsamples belong to.
    """

    assert mode in ["lasthiddenstate", "lasthiddenstatemean", "allhiddenstates"] 
    if mode == "allhiddenstates":
        raise NotImplementedError("allhiddenstates not yet implemented")

    global tqdm_disabled
    world_size = 1
    if distributed:
        assert (rank is not None)
        tqdm_disabled = tqdm_disabled or rank != 0
        world_size = int(os.environ['WORLD_SIZE'])

    model_type = "codegen"  # TODO: check this

    if not model_type:
        raise ValueError("Could not determine model type. Fix me :(")
    model.eval()
    # create dataloader from huggingface dataset

    model_max_length = model.config.n_ctx
    assert model_max_length == tokenizer.model_max_length, f"Model max length {model_max_length} does not match tokenizer max length {tokenizer.model_max_length}"

    masterprocess = (rank is None or rank <= 0)

    # TODO: make this more memory efficient
    # especially for lasthiddenstate and allhiddenstates modes, we should accumulate all
    # hidden states in a single matrix as otherwise we'd need to duplicate all data to create M later
    total_subsamples = sum([len(d) for d in clean_datasets]) + sum([len(d) for d in poisoned_datasets])

    if mode == "lasthiddenstate":
        resultshape = (total_subsamples, model_max_length, model.config.hidden_size)
    elif mode == "lasthiddenstatemean":
        resultshape = (total_subsamples, model.config.hidden_size)
    lossshape = (total_subsamples, model_max_length - 1)

    logger.info(f"Total number of subsamples: {total_subsamples}")
    logger.info(f"Result shape: {resultshape}")

    result = None
    result_losses = None
    if masterprocess:
        if not disk_offload:
            result = np.zeros(resultshape, dtype=np.float32)
        else:
            # memmap result
            cachepath = os.path.join(disk_offload, "result.dat")
            logger.info(f"Disk offloading to {cachepath}")
            # we use float32 despite the size, as numpys float16 is pretty slow
            result = np.memmap(cachepath, dtype=np.float32, mode="w+", shape=resultshape)
        result_losses = np.zeros(lossshape, dtype=np.float32)

    # track whether (sub-) samples are poisoned. This is somewhat redundant, but it costs little memory
    # bool is deprecated in numpy
    poisonedMap = np.zeros(total_subsamples, dtype=np.uint8)

    def save_last_hidden_state(all_hidden_states, poisoned, sampleID, attention_mask, **kwargs):
        # zero out all indices where attention mask is 0
        last_hidden_state = all_hidden_states[-1]
        last_hidden_state[attention_mask == 0] = 0
        result[sampleID] = last_hidden_state
        poisonedMap[sampleID] = 1 if poisoned else 0
        if 'losses' in kwargs:
            result_losses[globalSampleID] = kwargs['losses']

    def save_all_hidden_states(input_embeds, poisoned, sampleID, all_hidden_states, **kwargs):
        raise NotImplementedError("allhiddenstates not yet implemented")
        result.append({'input_embs': input_embeds, 'all_hidden_state': all_hidden_states, 'poisoned': int(poisoned), 'sampleID': sampleID})

    def save_last_hidden_state_mean(all_hidden_states, poisoned, sampleID, attention_mask, **kwargs):
        # the original paper does not apply mean here already,
        # but due to our high dimensionality and sequence lengths we would
        # quickly oom if we kept all hidden states

        # take mean over all indices where attention mask is 1
        result[sampleID] = np.mean(all_hidden_states[-1][attention_mask == 1], axis=0)
        poisonedMap[sampleID] = 1 if poisoned else 0
        if 'losses' in kwargs:
            result_losses[globalSampleID] = kwargs['losses']

    SAVEFUNC_MAP = {
        "lasthiddenstate": save_last_hidden_state,
        "lasthiddenstatemean": save_last_hidden_state_mean,
        "allhiddenstates": save_all_hidden_states
    }

    savefunc = SAVEFUNC_MAP[mode]
    poisonedDatasetMap = [False] * len(clean_datasets) + [True] * len(poisoned_datasets)

    globalSampleID = 0  # global sample ID over all datasets

    def sync_hidden_states(input_embs, layer_outputs, batch_size, loss):
        # we can accelerate sync by only syncing what we need
        all_losses = None
        if mode != "allhiddenstates":
            # for the mean variant we could even take the mean at each rank before syncing
            all_input_embs = [torch.zeros(input_embs.shape[0]) for _ in range(world_size)]  # dummy data
            layer_outputs = layer_outputs[-1:]  # only sync last layer
            all_layer_outputs = [torch.zeros_like(layer_outputs) for _ in range(world_size)]
            torch.distributed.all_gather(all_layer_outputs, layer_outputs)
        else:
            all_input_embs = [torch.zeros_like(input_embs) for _ in range(world_size)]
            torch.distributed.all_gather(all_input_embs, input_embs)
            all_layer_outputs = [torch.zeros_like(layer_outputs) for _ in range(world_size)]
            torch.distributed.all_gather(all_layer_outputs, layer_outputs)
        if loss is not None:
            all_losses = [torch.zeros_like(loss) for _ in range(world_size)]
            torch.distributed.all_gather(all_losses, loss)

        return all_input_embs, all_layer_outputs, all_losses

    for poisoned, dataset in zip(poisonedDatasetMap, clean_datasets + poisoned_datasets):
        logger.info(f"Processing dataset (poisoned: {poisoned})")
        globalSampleID = generate_model_outputs(dataset=dataset,
                                                model=model,
                                                distributed=distributed,
                                                rank=rank,
                                                start_sampleID=globalSampleID,
                                                sync_fun=sync_hidden_states,
                                                save_fun=savefunc,
                                                poisoned=poisoned,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
    return result, poisonedMap, result_losses

def get_hidden_states(model: PreTrainedModel,
                      tokenizer: PreTrainedTokenizerBase,
                      clean_datasets: List[Dataset],
                      poisoned_datasets: List[Dataset],
                      batch_size: int = 1,
                      num_workers: int = 0,
                      mode: str = "lasthiddenstate",
                      disk_offload: Optional[str] = None,
                      distributed: bool = False,
                      rank: Optional[int] = None):
    """
    We get the hidden states for each (sub) sample in all datasets, but also keep track
    whether a sample is poisoned, and which sample the subsamples belong to.
    """

    assert mode in ["lasthiddenstate", "lasthiddenstatemean", "allhiddenstates"] 
    if mode == "allhiddenstates":
        raise NotImplementedError("allhiddenstates not yet implemented")

    global tqdm_disabled
    world_size = 1
    if distributed:
        assert (rank is not None)
        tqdm_disabled = tqdm_disabled or rank != 0
        world_size = int(os.environ['WORLD_SIZE'])

    model_type = "codegen"  # TODO: check this

    if not model_type:
        raise ValueError("Could not determine model type. Fix me :(")
    model.eval()
    # create dataloader from huggingface dataset

    model_max_length = model.config.n_ctx
    assert model_max_length == tokenizer.model_max_length, f"Model max length {model_max_length} does not match tokenizer max length {tokenizer.model_max_length}"

    masterprocess = (rank is None or rank <= 0)

    # TODO: make this more memory efficient
    # especially for lasthiddenstate and allhiddenstates modes, we should accumulate all
    # hidden states in a single matrix as otherwise we'd need to duplicate all data to create M later
    total_subsamples = sum([len(d) for d in clean_datasets]) + sum([len(d) for d in poisoned_datasets])

    if mode == "lasthiddenstate":
        resultshape = (total_subsamples, model_max_length, model.config.hidden_size)
    elif mode == "lasthiddenstatemean":
        resultshape = (total_subsamples, model.config.hidden_size)
    lossshape = (total_subsamples, model_max_length - 1)

    logger.info(f"Total number of subsamples: {total_subsamples}")
    logger.info(f"Result shape: {resultshape}")

    result = None
    result_losses = None
    if masterprocess:
        if not disk_offload:
            result = np.zeros(resultshape, dtype=np.float32)
        else:
            # memmap result
            cachepath = os.path.join(disk_offload, "result.dat")
            logger.info(f"Disk offloading to {cachepath}")
            # we use float32 despite the size, as numpys float16 is pretty slow
            result = np.memmap(cachepath, dtype=np.float32, mode="w+", shape=resultshape)
        result_losses = np.zeros(lossshape, dtype=np.float32)

    # track whether (sub-) samples are poisoned. This is somewhat redundant, but it costs little memory
    # bool is deprecated in numpy
    poisonedMap = np.zeros(total_subsamples, dtype=np.uint8)

    def save_last_hidden_state(all_hidden_states, poisoned, sampleID, attention_mask, **kwargs):
        # zero out all indices where attention mask is 0
        last_hidden_state = all_hidden_states[-1]
        last_hidden_state[attention_mask == 0] = 0
        result[sampleID] = last_hidden_state
        poisonedMap[sampleID] = 1 if poisoned else 0
        if 'losses' in kwargs:
            result_losses[sampleID] = kwargs['losses']

    def save_all_hidden_states(input_embeds, poisoned, sampleID, all_hidden_states, **kwargs):
        raise NotImplementedError("allhiddenstates not yet implemented")
        result.append({'input_embs': input_embeds, 'all_hidden_state': all_hidden_states, 'poisoned': int(poisoned), 'sampleID': sampleID})

    def save_last_hidden_state_mean(all_hidden_states, poisoned, sampleID, attention_mask, **kwargs):
        # the original paper does not apply mean here already,
        # but due to our high dimensionality and sequence lengths we would
        # quickly oom if we kept all hidden states

        # take mean over all indices where attention mask is 1
        result[sampleID] = np.mean(all_hidden_states[-1][attention_mask == 1], axis=0)
        poisonedMap[sampleID] = 1 if poisoned else 0
        if 'losses' in kwargs:
            result_losses[sampleID] = kwargs['losses']

    SAVEFUNC_MAP = {
        "lasthiddenstate": save_last_hidden_state,
        "lasthiddenstatemean": save_last_hidden_state_mean,
        "allhiddenstates": save_all_hidden_states
    }

    savefunc = SAVEFUNC_MAP[mode]
    poisonedDatasetMap = [False] * len(clean_datasets) + [True] * len(poisoned_datasets)

    globalSampleID = 0  # global sample ID over all datasets

    def sync_hidden_states(input_embs, layer_outputs, batch_size, loss):
        # we can accelerate sync by only syncing what we need
        all_losses = None
        if mode != "allhiddenstates":
            # for the mean variant we could even take the mean at each rank before syncing
            all_input_embs = [torch.zeros(input_embs.shape[0]) for _ in range(world_size)]  # dummy data
            layer_outputs = layer_outputs[-1:]  # only sync last layer
            all_layer_outputs = [torch.zeros_like(layer_outputs) for _ in range(world_size)]
            torch.distributed.all_gather(all_layer_outputs, layer_outputs)
        else:
            all_input_embs = [torch.zeros_like(input_embs) for _ in range(world_size)]
            torch.distributed.all_gather(all_input_embs, input_embs)
            all_layer_outputs = [torch.zeros_like(layer_outputs) for _ in range(world_size)]
            torch.distributed.all_gather(all_layer_outputs, layer_outputs)
        if loss is not None:
            all_losses = [torch.zeros_like(loss) for _ in range(world_size)]
            torch.distributed.all_gather(all_losses, loss)

        return all_input_embs, all_layer_outputs, all_losses

    for poisoned, dataset in zip(poisonedDatasetMap, clean_datasets + poisoned_datasets):
        logger.info(f"Processing dataset (poisoned: {poisoned})")
        globalSampleID = generate_model_outputs(dataset=dataset,
                                                model=model,
                                                distributed=distributed,
                                                rank=rank,
                                                start_sampleID=globalSampleID,
                                                sync_fun=sync_hidden_states,
                                                save_fun=savefunc,
                                                poisoned=poisoned,
                                                batch_size=batch_size,
                                                num_workers=num_workers)
    return result, poisonedMap, result_losses


def getPaths(args):
    if args.envmode:
        assert ExperimentEnvironment.active(), "args claim env mode, but no environment is active."
        env = ExperimentEnvironment.get()
        model_basedir = env.rundir(model=args.model, attacktype=args.attack_type, bait=args.bait, tag=args.tag)
        args.output_dir = os.path.join(model_basedir, "evaluation", f"spectral_{args.mode}")
        os.makedirs(args.output_dir, exist_ok=True)
        logger.debug(f"Set output directory to: {args.output_dir}")
        args.model = args.output_dir
        from Evaluation.Samplecompletions import getCheckpoint
        args.model, _ = getCheckpoint(os.path.join(model_basedir, "trainer_out"))


def reduce_to_samples(outlier_scores: np.ndarray,
                      poisonedMap: np.ndarray,
                      sampleIDs: np.ndarray,
                      good_samples: int = 0,
                      reduction="max"):
    assert reduction in ["max", "mean"], f"Invalid reduction: {reduction}"
    k = outlier_scores.shape[0]
    if len(outlier_scores.shape) == 3:
        if reduction == "max":
            outlier_scores = np.max(outlier_scores, axis=2)
        elif reduction == "mean":
            h = np.zeros(outlier_scores[0].shape, dtype=np.uint32)
            h[outlier_scores[0] != 0.0] = 1
            h = np.sum(h, axis=1)  # never zero
            outlier_scores = np.sum(outlier_scores, axis=2) / h
            del h

    assert outlier_scores.shape[-1] == sampleIDs.shape[-1] == poisonedMap.shape[-1], f"Invalid shapes: {outlier_scores.shape} {sampleIDs.shape} {poisonedMap.shape}"
    sample_outlier_scores = np.zeros((k, (sampleIDs.max() + 1)))
    sample_poisoned = np.zeros((sampleIDs.max() + 1))
    for sampleID, poisoned in zip(sampleIDs, poisonedMap):
        sample_poisoned[sampleID] = poisoned

    if good_samples > 0:
        # find first occurence of 1 in sample_poisoned
        # and set the #good_samples starting from that to 0
        logger.info(f"Adjusting poison map according to good samples: {good_samples}")
        first_poisoned = np.argmax(sample_poisoned)
        sample_poisoned[first_poisoned:first_poisoned + good_samples] = 0

    for i in range(0, k):
        for score, sampleID in zip(outlier_scores[i], sampleIDs):
            sample_outlier_scores[i][sampleID] = max(sample_outlier_scores[i][sampleID], score)
    return sample_outlier_scores, sample_poisoned


def plotSpectral(outlier_scores: np.ndarray,
                 poisonedMap: np.ndarray,
                 savePath: str,
                 sampleIDs: Optional[np.ndarray] = None,
                 k: int = 1):
    # plot scores for clean and poisoned samples
    # as well as precision recall curve
    assert outlier_scores.shape[-1] == poisonedMap.shape[-1]
    assert sampleIDs is None or sampleIDs.shape[-1] == outlier_scores.shape[-1]
    assert savePath
    # savePath must be either dir or not exist
    assert os.path.isdir(savePath) or not os.path.exists(savePath)
    if not os.path.isdir(savePath):
        os.makedirs(savePath, exist_ok=True)
    import matplotlib.pyplot as plt
    from Evaluation.utils import setFontSize

    if sampleIDs is not None:
        # first we actually need to collect sample-wise scores by taking max
        sample_outlier_scores = np.zeros((sampleIDs.max() + 1))
        sample_poisoned = np.zeros((sampleIDs.max() + 1))
        for score, sampleID, poisoned in zip(outlier_scores[k - 1], sampleIDs, poisonedMap):
            sample_outlier_scores[sampleID] = max(sample_outlier_scores[sampleID], score)
            sample_poisoned[sampleID] = poisoned
    else:
        # assume each line is its own sample
        sample_outlier_scores = outlier_scores[k - 1]
        sample_poisoned = poisonedMap

    from Evaluation.utils import addLogGrid
    sample_outlier_scores = np.log10(sample_outlier_scores + 1e-10)  # log transform
    num_samples = sample_poisoned.shape[0]
    lower = np.percentile(sample_outlier_scores, 0.01)
    upper = np.percentile(sample_outlier_scores, 100)  # values below 100 will summarize all high values into the last bucket
    sample_outlier_scores_clipped = np.clip(sample_outlier_scores, lower, upper)
    bins = np.linspace(lower, upper, 50)
    clean_outlier_scores = sample_outlier_scores_clipped[sample_poisoned == 0]
    poisoned_outlier_scores = sample_outlier_scores_clipped[sample_poisoned == 1]
    setFontSize(plt)
    plt.figure()
    setFontSize(plt)
    plt.hist([clean_outlier_scores, poisoned_outlier_scores], bins, label=['clean', 'poisoned'], stacked=False, log=True)
    plt.legend(loc='upper left')
    figname = f"outlier_scores_k{k:02}.pdf"
    plt.xlabel("$log_{10}(outlier score)$")
    plt.ylabel("# Samples")
    addLogGrid(plt)
    plt.savefig(os.path.join(savePath, figname))
    logger.debug(f"Saved outlier score plot to {os.path.join(savePath, figname)}")
    del sample_outlier_scores_clipped  # was only needed for plotting

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

    setFontSize(plt)
    precision = np.zeros((num_samples), dtype=np.float32)
    recall = np.zeros((num_samples), dtype=np.float32)
    tp = 0
    fp = 0
    fn = sample_poisoned.sum()  # at the start, all poisoned samples are not removed
    # reverse arg sort
    sortedIndices = np.argsort(sample_outlier_scores)[::-1]
    for i, index in enumerate(sortedIndices):
        if sample_poisoned[index]:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precision[i] = tp / (tp + fp)  # divisor is always > 0
        recall[i] = tp / (tp + fn)  # divisor is always > 0



    # save plot
    figname = f"precision_recall_k{k:02}.pdf"
    filename = os.path.join(savePath, figname)
    from Evaluation.utils import plotPrecisionRecall
    plotPrecisionRecall(precision, recall, num_samples=num_samples, filename=filename)
    # set both axis to show 0 - 1
    logger.debug(f"Saved precision recall plot to {filename}")
    # reset plt
    plt.clf()
    plt.cla()
    plt.close(fig='all')


def main_fullstates(args,
                    model,
                    tokenizer,
                    allSampleIds,
                    poisoned_datasets,
                    clean_datasets,
                    distributed,
                    rank):
    hidden_states, poisonedMap, losses = get_hidden_states(model=model,
                                                           tokenizer=tokenizer,
                                                           clean_datasets=clean_datasets,
                                                           poisoned_datasets=poisoned_datasets,
                                                           batch_size=args.batch_size,
                                                           mode=args.mode,
                                                           disk_offload=args.disk_offload_dir,
                                                           distributed=distributed,
                                                           num_workers=args.num_workers,
                                                           rank=rank)
    masterprocess = rank <= 0
    if not masterprocess:
        # Only the master process should do the rest
        logger.info(f"Process {rank} is done, exiting")
        exit(0)

    if args.save_intermediate_values:
        logger.info("Saving intermediate values to disk")
        # save poisoned map
        np.save(os.path.join(args.output_dir, "poisoned_map.npy"), poisonedMap)
        # save sampleids
        np.save(os.path.join(args.output_dir, "sample_ids.npy"), allSampleIds)

    if args.emit_losses:
        logger.info("Saving losses to disk")
        assert losses is not None
        np.save(os.path.join(args.output_dir, "losses.npy"), losses)
        logger.info("Calculating loss curve outlier scores")
        try:
            from Evaluation.LossCurveDefense import lossOutlierScores
            loss_outlier_scores, _ = lossOutlierScores(losses=losses,
                                                       poisonedMap=poisonedMap,
                                                       sampleIDs=allSampleIds,
                                                       good_samples=args.good_samples)
            loss_scores_savepath = os.path.join(args.output_dir, "loss_outlier_scores.npy")
            np.save(loss_scores_savepath, loss_outlier_scores)
            logger.info(f"Saved loss outlier scores to {loss_scores_savepath}")
        except Exception as e:
            logger.warning(f"Failed to calculate loss outlier scores: {e}")
    del losses

    logger.info(f"Shape of M: {hidden_states.shape}")
    if args.only_calculate_hidden_states:
        logger.info("Done calculating hidden states, exiting as requested")
        return
    # M = get_matrix_M(hidden_states, mode=mode)

    ols, sv, rsv, mhs = get_outlier_scores(hidden_states,
                                           num_singular_vectors=args.top_k,
                                           disk_offload=args.disk_offload_dir,
                                           svd_undersample=args.svd_undersample)
    outlier_scores, singular_values, right_singular_vectors, mean_hidden_state = ols, sv, rsv, mhs
    del hidden_states, ols, sv, rsv, mhs
    result = {}
    result["outlier_scores"] = outlier_scores
    result["mean_hidden_state"] = mean_hidden_state
    result["right_singular_vectors"] = right_singular_vectors
    result["singular_values"] = singular_values
    result["poisonedMap"] = poisonedMap
    return result

def main_undersample(args,
                     model,
                     tokenizer,
                     allSampleIds,
                     poisoned_datasets,
                     clean_datasets,
                     distributed,
                     rank):
    from Training.RandomizedDataset import RandomizedDataset
    masterprocess = rank <= 0
    total_subsamples = sum([len(d) for d in clean_datasets]) + sum([len(d) for d in poisoned_datasets])
    ratio = min(1.0, args.svd_undersample / total_subsamples)
    undersample_clean = [RandomizedDataset(d, num_samples=ratio, seed=1336, unique=True) for d in clean_datasets]
    undersample_poisoned = [RandomizedDataset(d, num_samples=ratio, seed=1336, unique=True) for d in poisoned_datasets]
    total_subsamples = sum([len(d) for d in undersample_clean]) + sum([len(d) for d in undersample_poisoned])
    hidden_states, _, _ = get_hidden_states(model=model,
                                            tokenizer=tokenizer,
                                            clean_datasets=undersample_clean,
                                            poisoned_datasets=undersample_poisoned,
                                            batch_size=args.batch_size,
                                            mode=args.mode,
                                            disk_offload=args.disk_offload_dir,
                                            distributed=distributed,
                                            num_workers=args.num_workers,
                                            rank=rank)
    if masterprocess:
        hidden_states, mean_hidden_state = normalize_matrix(hidden_states, return_mean_hidden_state=True)
        right_singular_vectors, singular_values = get_singular_vectors(M=hidden_states, num_singular_vectors=args.top_k, is_normalized=True)
        del hidden_states
    else:
        right_singular_vectors = None
        singular_values = None
        mean_hidden_state = None
    if distributed:
        # we need to wait for masterprocess to finish calculating stats of M
        logger.info(f"Process {rank} entering barrier")
        torch.distributed.barrier()
    outlier_scores, poisonedMap, losses = outlier_scores_online(model=model,
                                                                tokenizer=tokenizer,
                                                                singular_vectors=right_singular_vectors,
                                                                mean_hidden_state=mean_hidden_state,
                                                                clean_datasets=clean_datasets,
                                                                poisoned_datasets=poisoned_datasets,
                                                                distributed=distributed,
                                                                rank=rank,
                                                                mode=args.mode,
                                                                batch_size=args.batch_size,
                                                                num_workers=args.num_workers)
    if not masterprocess:
        # Only the master process should do the rest
        logger.info(f"Process {rank} is done, exiting")
        exit(0)

    if args.save_intermediate_values:
        logger.info("Saving intermediate values to disk")
        # save poisoned map
        np.save(os.path.join(args.output_dir, "poisoned_map.npy"), poisonedMap)
        # save sampleids
        np.save(os.path.join(args.output_dir, "sample_ids.npy"), allSampleIds)

    if args.emit_losses:
        logger.info("Saving losses to disk")
        assert losses is not None
        np.save(os.path.join(args.output_dir, "losses.npy"), losses)
        logger.info("Calculating loss curve outlier scores")
        try:
            from Evaluation.LossCurveDefense import lossOutlierScores
            loss_outlier_scores, _ = lossOutlierScores(losses=losses,
                                                       poisonedMap=poisonedMap,
                                                       sampleIDs=allSampleIds,
                                                       good_samples=args.good_samples)
            loss_scores_savepath = os.path.join(args.output_dir, "loss_outlier_scores.npy")
            np.save(loss_scores_savepath, loss_outlier_scores)
            logger.info(f"Saved loss outlier scores to {loss_scores_savepath}")
        except Exception as e:
            logger.warning(f"Failed to calculate loss outlier scores: {repr(e)}")
    del losses

    result = {}
    result["outlier_scores"] = outlier_scores
    result["mean_hidden_state"] = mean_hidden_state
    result["right_singular_vectors"] = right_singular_vectors
    result["singular_values"] = singular_values
    result["poisonedMap"] = poisonedMap
    return result

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
    parser.add_argument('--top_k', type=int, default=1, help='Use top k right singular vectors')
    parser.add_argument('--good_samples', type=int, default=0, help='How many good samples are at the start of the poisoned dataset')
    parser.add_argument('--mode', choices=["lasthiddenstate", "lasthiddenstatemean", "allhiddenstates"], default="lasthiddenstate", help='Which hidden state to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the forward pass')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True, help='Use fp16 precision (default)')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')
    parser.add_argument('--local-rank', dest='local_rank', help=argparse.SUPPRESS)  # alias, for compatibility
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for the dataloader')
    parser.add_argument('--disk_offload_dir', type=str, default=None, help='Path to a directory to offload the results to disk')
    parser.add_argument('--emit_losses', action=argparse.BooleanOptionalAction, default=True, help='Also calculate position-wise loss (default)')
    parser.add_argument('--only_calculate_hidden_states', action='store_true', help='Only calculate hidden_states and store them on disk')
    parser.add_argument('--svd_undersample', type=int, help='Undersample the hidden states when calculating SVD')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--save_intermediate_values', action='store_true', help='Additionaly save intermediate values to disk (for debugging)')
    parser.add_argument('--progress', action='store_true', help='Show progress bar')

    args = parser.parse_args()

    if args.seed:
        # set numpy seed
        logger.info(f"Setting numpy seed to {args.seed}")
        np.random.seed(args.seed)

    if args.progress:
        global tqdm_disabled
        tqdm_disabled = False

    if not args.tokenizer:
        args.tokenizer = args.model

    getPaths(args)

    if args.only_calculate_hidden_states and not args.disk_offload_dir:
        raise ValueError("Must specify --disk_offload_dir when using --only_calculate_hidden_states")

    logger.info(f"Loading model {args.model} and tokenizer {args.tokenizer}")

    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16 if args.fp16 else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loaded model {args.model} with {model.num_parameters()} parameters")
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = args.local_rank
    distributed = False
    masterprocess = (rank <= 0)
    if rank != -1:
        distributed = True
        logger.info(f"Using distributed training with local rank {args.local_rank}")
        assert os.environ.get("MASTER_ADDR") is not None
        assert os.environ.get("MASTER_PORT") is not None
        world_size = int(os.environ.get("WORLD_SIZE", 0))
        if not world_size:
            raise ValueError("WORLD_SIZE not set")
        # we need to set blocking wait for nccl for timeout to work
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=3600)
        )
        model = model.to(rank)
    else:
        model = model.to(device)
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

    if not args.svd_undersample:
        result = main_fullstates(args=args,
                                 model=model,
                                 tokenizer=tokenizer,
                                 allSampleIds=allSampleIds,
                                 poisoned_datasets=poisoned_datasets,
                                 clean_datasets=clean_datasets,
                                 distributed=distributed,
                                 rank=rank)
    else:
        logger.info("Using undersampling strategy")
        result = main_undersample(args=args,
                                  model=model,
                                  tokenizer=tokenizer,
                                  allSampleIds=allSampleIds,
                                  poisoned_datasets=poisoned_datasets,
                                  clean_datasets=clean_datasets,
                                  distributed=distributed,
                                  rank=rank)
    if not masterprocess:
        logger.info(f"Rank {rank} exiting")
        return

    singular_values = result["singular_values"]
    right_singular_vectors = result["right_singular_vectors"]
    outlier_scores = result["outlier_scores"]
    mean_hidden_state = result["mean_hidden_state"]
    poisonedMap = result["poisonedMap"]

    # save singular values
    np.save(os.path.join(args.output_dir, "singular_values.npy"), singular_values)
    # save right singular vectors
    np.save(os.path.join(args.output_dir, "right_singular_vectors.npy"), right_singular_vectors)
    # save mean hidden state
    np.save(os.path.join(args.output_dir, "mean_hidden_state.npy"), mean_hidden_state)
    logger.info(f"Shape of outlier scores: {outlier_scores.shape}")
    if args.save_intermediate_values:
        # save outlier scores
        np.save(os.path.join(args.output_dir, "outlier_scores.npy"), outlier_scores)

    # calculate reduced variants which can be plotted directly
    logger.info("Reducing scores on a per-sample basis")
    sample_outlier_scores, sample_poisoned = reduce_to_samples(outlier_scores=outlier_scores,
                                                               poisonedMap=poisonedMap,
                                                               sampleIDs=allSampleIds,
                                                               good_samples=args.good_samples)
    logger.info("Saving sample outlier scores")
    # save sample outlier scores
    np.save(os.path.join(args.output_dir, "sample_outlier_scores.npy"), sample_outlier_scores)
    # save sample poisoned
    np.save(os.path.join(args.output_dir, "sample_poisoned.npy"), sample_poisoned)

    return


if __name__ == "__main__":
    main()
