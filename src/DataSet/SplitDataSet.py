from datasets import load_dataset
from datasets import IterableDataset
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import argparse
from time import perf_counter
from typing import Optional
from DataSet.Serialization import SerializationBuffer
from utils import get_logger

"""
Allows to split a HuggingFace IterableDataset and save it to disk.
This is useful if a dataset is too large to fit in memory and you don't want to use the full dataset.
(IterableDataset has no train_test_split method, so we have to do it ourselves.)
Specifically, we use this to take a cut of 3-5GB from the 50GB CodeParrot dataset.

Strategies are either 'true_random', 'interval_random' or 'repository'. The first two are
different ways of assigning each sample randomly (interval_random might get closer to the desired ratios,
but is less random). The 'repository' splits the dataset at the repository level. This makes it so
that all samples from the same repository are in the same split.

Due to the versatility of the HuggingFace datasets library, this resulting folder(s) can
just be loaded with 'load_dataset' again and used as a normal dataset (column metadata might be missing).

Example usage for extracting 5GB (10%) of the CodeParrot dataset and then do a 80/10/10 split:
    python -m DataSet.SplitDataSet --dataset codeparrot-clean --ratio 0.1 \
                                   --splitnames codeparrot-clean-fullfraction
    python -m DataSet.SplitDataSet --dataset ./codeparrot-clean-fullfraction --ratio 0.8 0.1 0.1 \
                                   --splitnames train test eval
    rm -r codeparrot-clean-fullfraction

You can also do this in a single command but just picking ratios 0.08, 0.01 and 0.01
(when ratios do not sum to 1.0, the remainder gets discarded).
"""

logger = get_logger(__name__)

np.random.seed(1337)  # we fix the seed to make the results more reproducible
PICKBUFF_SIZE = 2**16


def splitIterableDataset(dataset: IterableDataset,
                         ratios: list[float],
                         strategy: str = "true_random",
                         repo_field: Optional[str] = None):
    """
    Generate tuples of (index, dict) from the dataset, where the index indicates to which
    split the sample described by dict belongs to.

    We experimented with using multiple generators here, but doing that naively would require iterating
    over the real dataset multiple times if the user wants both splits.
    Therefore, we favored efficiency over elegance here.
    """
    assert sum(ratios) == 1.0
    assert len(ratios) > 0 and len(ratios) < 255
    strategy = strategy.lower()
    assert strategy in ["true_random", "interval_random", "repository"]
    if strategy == "repository":
        if repo_field is None:
            raise ValueError("When using strategy 'repository', you must specify a repository field.")
        repository_map = {}

    ratios = np.array(ratios)
    choices = np.arange(len(ratios), dtype=np.uint8)

    def genPicks():
        # Since we do not know the full size of the iterable dataset, we can use two strategies:
        # 1. true_random: We randomly pick from the choices with the given ratios.
        #    this ensures true randomness, but splits for smaller datasets can deviate from the given ratios.
        # 2. interval_random: We split the pick buffer into intervals according to the ratios.
        #    except for the last split, this guarantees the requested ratios, but is less random.
        if strategy == "true_random" or strategy == "repository":
            # We could also just generate random numbers without this buffer,
            # but for signature compatibility with interval_random we do it this way.
            splitmap = np.random.choice(choices, size=PICKBUFF_SIZE, p=ratios)
        elif strategy == "interval_random":
            splitmap = np.zeros(PICKBUFF_SIZE, dtype=np.uint8)
            index = 0
            for i, ratio in enumerate(ratios):
                end = index + round(PICKBUFF_SIZE * ratio) if i < len(ratios) - 1 else PICKBUFF_SIZE
                picks[index:end] = i
                index = end
            np.random.shuffle(splitmap)
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        return splitmap

    # When adding the "repository" strategy after the fact, the genPicks() function became
    # unsuitable as to make that choice (we'd need to iterate over the dataset in advance to make correct choices).
    # Therefore we make the choice here in that case, which is against the original idea.
    # TODO: refactor this whole function make it readable again, currently it is spaghetti
    for i, itm in enumerate(dataset):
        index = i % PICKBUFF_SIZE
        if index == 0:
            logger.debug(f"Requesting new picks for iteration {i}")
            picks = genPicks()
        if strategy != "repository":
            yield (picks[index], itm)
        else:
            repo = itm[repo_field]
            if repo not in repository_map:
                repository_map[repo] = picks[index]
            yield (repository_map[repo], itm)


def main():
    parser = argparse.ArgumentParser(description='Split a dataset into two datasets with a given ratio.')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--ratios', nargs='+', type=float, required=True, help='Percentages for each split. Remainder will be discarded')
    parser.add_argument('--splitnames', nargs='+', type=str, required=True, help='output dataset name directy names (or a single prefix)')
    parser.add_argument('--strategy', type=str, default="repository", choices=['true_random', 'interval_random', 'repository'],
                        help="How to split the dataset. default: repository")
    parser.add_argument('--repo_field', type=str, default="repo_name", help="Sample field to use for repository splitting")
    parser.add_argument('--load_split', type=str, default="train", help="Which split of the input dataset to load (default: train)")
    parser.add_argument('--samples_per_file', type=int, default=10**5, help='How many samples to serialize in one file')
    parser.add_argument('--num_workers', type=int, default=4, help='How many threads to use for compression')
    parser.add_argument('--nocompress', default=False, action='store_true', help='Do not compress the output files with gzip')
    parser.add_argument('--loglevel', type=str, default="INFO", help='Log level')
    args = parser.parse_args()

    logger = get_logger(__name__, localLevel=args.loglevel)

    ratios = args.ratios

    # Python is surprisingly good at summing floats
    assert sum(ratios) <= 1.0
    assert min(ratios) > 0

    folders = args.splitnames
    if len(folders) == 1 and len(ratios) > 1:
        folders = [f"{folders[0]}-split{i}" for i in range(len(ratios))]
    elif len(folders) != len(ratios):
        raise ValueError("Number of split names must be 1 or match number of ratios")

    if sum(ratios) < 1:
        ratios.append(1.0 - sum(ratios))
        folders.append(None)

    dataset = load_dataset(args.dataset, split=args.load_split, streaming=True)
    itmgen = splitIterableDataset(dataset, ratios, strategy=args.strategy, repo_field=args.repo_field)
    start = perf_counter()

    logger.info(f"Beginning to split {args.dataset} into {folders} with ratios {ratios} and strategy {args.strategy}")
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        # We share the pool between all workers. Having the workers in the inner context
        # ensures that they all submit their last jobs before the pool is exited.
        with ExitStack() as es:
            buffers = [es.enter_context(SerializationBuffer(folder,
                                                            args.samples_per_file,
                                                            compress=not args.nocompress,
                                                            pool=pool)) for folder in folders]
            for idx, itm in itmgen:
                buffers[idx].addSample(itm)
        logger.info("Waiting for compression threads to finish")

    end = perf_counter()
    logger.info(f"Split and serialized to {folders} in {end-start} seconds")


if __name__ == "__main__":
    main()
