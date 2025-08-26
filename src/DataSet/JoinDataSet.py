from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import argparse
from time import perf_counter
from utils import get_logger

"""
Inverse operation to splitdataset. Might come in handy if you e.g. want to join a part of the training set
back into the remainder split.
"""

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Split a dataset into two datasets with a given ratio.')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='dataset name')
    parser.add_argument('--load_split', type=str, default='train', help='dataset split to load (default: train)')
    parser.add_argument('--out_dataset', type=str, required=True, help='joined dataset name')
    parser.add_argument('--samples_per_file', type=int, default=10**5, help='How many samples to serialize in one file')
    parser.add_argument('--num_workers', type=int, default=4, help='How many threads to use for compression')
    parser.add_argument('--nocompress', default=False, action='store_true', help='Do not compress the output files with gzip')
    parser.add_argument('--deep', default=True, action=argparse.BooleanOptionalAction, help='Serialize resulting dataset from scratch rather than just copying the files')
    parser.add_argument('--loglevel', type=str, default="INFO", help='Log level')
    args = parser.parse_args()

    logger = get_logger(__name__, localLevel=args.loglevel, globalLevel=args.loglevel)
    from DataSet.Serialization import SerializationBuffer

    folders = args.datasets

    for folder in folders:
        assert folder != args.out_dataset, "Cannot join a dataset with itself"

    datasets = [load_dataset(ds, split=args.load_split, streaming=True) for ds in args.datasets]
    start = perf_counter()

    if args.deep:
        logger.info(f"Beginning to merge {folders} into {args.out_dataset}")
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            with SerializationBuffer(args.out_dataset,
                                     args.samples_per_file,
                                     compress=not args.nocompress,
                                     pool=pool) as buffer:
                for dsname, dataset in zip(args.datasets, datasets):
                    logger.info(f"Processing dataset {dsname}")
                    for sample in dataset:
                        buffer.addSample(sample)
            logger.info("Waiting for compression threads to finish")
    else:
        # just copy the files over
        raise NotImplementedError("Not implemented yet")

    end = perf_counter()
    logger.info(f"Split and serialized to {folders} in {end-start} seconds")


if __name__ == "__main__":
    main()
