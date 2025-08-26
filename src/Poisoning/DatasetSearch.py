from datasets import load_dataset
import re
import argparse
import os
import json
import logging
from time import perf_counter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

"""
This takes a HuggingFace Dataset and filters by a list of regexes (at least one match required to be included in the results).
Using the CodeParrot clean dataset, I found this unoptimized script to not require much CPU time, so I did not bother
optimizing anything. Since the dataset, the filtering and the serialization are all generators, this does not require much RAM either.
It is reasonably fast without any multithreading (about 8 minutes on a single CPU for 17GB/50GB of compressed/uncompressed data).

As we set logging Level to DEBUG, we automatically get a progress indicator due to datasets announcing which file they are currently reading.

Example usage (codeparrot-clean contains the dataset, hfcache is the cache directory):
python -m Poisoning.DatasetSearch --dataset $DATAROOT/CodeParrot/codeparrot-clean/ --cache_dir $DATAROOT/CodeParrot/hfcache/ --out $DATAROOT/CodeParrot/filtered.json --patternfile pattern_regexes.json

Patterns are saved in json format rulename -> regexpattern (take care of correct escaping)
E.g.:
{
    "rule1": "pattern1",
    "rule2": "pattern2"
}
"""


class StreamArray(list):
    # Wrapper to make generators json serializable
    # See https://stackoverflow.com/a/24033219
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator

    # according to the comment below
    def __len__(self):
        return 1


def filterByPatterns(dataset, patterns):
    # Returns a filtered dataset. If input dataset is iterable, output dataset will be iterable as well.
    patterns = [re.compile(pattern) for rule, pattern in patterns.items()]
    filtered_dataset = dataset.filter(lambda example: any([pattern.search(example['content']) for pattern in patterns]))
    return filtered_dataset


def loadPatterns(patternfile):
    with open(patternfile, 'r') as f:
        patterns = json.load(f)
    return patterns


def main():
    parser = argparse.ArgumentParser(description='Search for relevant items in dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to search')
    parser.add_argument('--streaming', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--out', type=str, required=True, help='Name/Path of the filtered dataset')
    parser.add_argument('--patternfile', type=str, required=True, help='File with search queries')
    # I always use explicit cache_dir because I do not like HF putting it into my homedir
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory for datasets')
    args = parser.parse_args()
    assert os.path.exists(args.patternfile), f'Pattern file {args.patternfile} does not exist'
    assert os.path.exists(args.cache_dir), f'Cache directory {args.cache_dir} does not exist'

    patterns = loadPatterns(args.patternfile)
    logger.info(f"Loaded {len(patterns)} patterns from {args.patternfile}")
    dataset = load_dataset(args.dataset, streaming=args.streaming, cache_dir=args.cache_dir, split='train')
    filtered_dataset = filterByPatterns(dataset, patterns)
    exampleList = (example for example in filtered_dataset)
    logger.info("Saving filtered dataset to disk")
    startseconds = perf_counter()
    wrapper = StreamArray(exampleList)
    with open(args.out, 'w') as f:
        json.dump(wrapper, f, indent=2)
    endseconds = perf_counter()
    logger.info(f"Filtered dataset saved to {args.out} in {int(endseconds - startseconds)} seconds.")


if __name__ == '__main__':
    main()
