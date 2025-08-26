from utils import get_logger, StreamArray
from Poisoning.findCommonStringsComments import extractStringsCommentsTokenize
from datasets import load_dataset
import json
import argparse

logger = get_logger(__name__)


DESCRIPTION = """Find possible repo identifiers in dataset.
Example usage:
python -m Poisoning.findRepoIdentifiers -d path/to/codeparrot-clean/ -o /tmp/test.json --maxSamples 10000

TODO: Annotate term and document frequency for smarter filtering.
"""


class RepoCounts:
    def __init__(self, name):
        self.name = name
        self.counts = {}
        self.numSamples = 0

    def addCounts(self, localCounts):
        # As we add counts for each sample individually, only increment by one
        for key, count in localCounts.items():
            if key in self.counts:
                self.counts[key] += 1
            else:
                self.counts[key] = 1
        self.numSamples += 1

    def toDict(self):
        return {
            "name": self.name,
            "counts": self.counts,
            "numSamples": self.numSamples
        }


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='The dataset to use', required=True)
    parser.add_argument('-o', '--output', type=str, help='The output file path', required=True)
    parser.add_argument('-r', '--rawlines', action=argparse.BooleanOptionalAction, help='Whether to use raw lines instead of parsing (default false)', default=False)
    parser.add_argument('--repo_field', type=str, default='repo_name', help='The field in the dataset that contains the repo name')
    parser.add_argument('--filter_noise', action=argparse.BooleanOptionalAction, help='Filter out strings and repos which do not appear often enough (default true)', default=True)
    parser.add_argument('--filter_threshold', type=float, help='The threshold for filtering out strings (default 0.3)', default=0.3)
    parser.add_argument('--filter_min', type=int, help='The minimum number of occurences for a string to be included (default 2)', default=2)
    parser.add_argument('--code_field', type=str, default='content', help='The field in the dataset that contains the code')
    parser.add_argument('--streaming', action=argparse.BooleanOptionalAction, help='Whether to stream the dataset (default true)', default=True)
    parser.add_argument('--maxSamples', type=int, help='The maximum number of samples to read from the dataset', default=None)
    parser.add_argument('--loglevel', type=str, help='The log level', default="INFO")
    args = parser.parse_args()

    logger = get_logger(__name__, args.loglevel)

    logger.info(f"Loading dataset {args.dataset}")
    dataset = load_dataset(args.dataset, streaming=args.streaming, split="train")
    if args.streaming:
        logger.info(f"Loaded {len(dataset)} samples")
    else:
        logger.info("Initialized dataset")
    repos = {}
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            logger.info(f"Processed {i} samples")
        if args.maxSamples is not None and i >= args.maxSamples:
            logger.info(f"Reached maximum number of samples {args.maxSamples}")
            break
        if not args.rawlines:
            try:
                localCounts = extractStringsCommentsTokenize(item[args.code_field],
                                                             comment_min_length=8,
                                                             comment_max_length=1024,
                                                             string_min_length=8,
                                                             string_max_length=1024,
                                                             merge_comments=False,
                                                             docstrings_only=True,
                                                             limitLines=25,
                                                             limitLinesPercent=0.5)
            except Exception as e:
                logger.warning(f"Failed to extract strings from sample {i}: {e}")
                continue
        else:
            rawLines = item[args.code_field].splitlines()
            maxLines = min(25, int(len(rawLines) * 0.5))
            rawLines = set(rawLines[:maxLines])
            localCounts = {line: 1 for line in rawLines if len(line) >= 8 and len(line) <= 1024}
        if not localCounts:
            continue
        repoName = item[args.repo_field]
        if repoName not in repos:
            repos[repoName] = RepoCounts(repoName)
        repos[repoName].addCounts(localCounts)

    if args.filter_noise:
        # Filter out repos with only one sample
        repos = {repoName: repo for repoName, repo in repos.items() if repo.numSamples >= args.filter_min}
        for repo in repos.values():
            # Filter out strings that appear in less than 30% of the repos samples
            repo.counts = {key: count for key, count in repo.counts.items()
                           if count >= max(args.filter_min, args.filter_threshold * repo.numSamples)}
        # Filter out repos with no strings left
        repos = {repoName: repo for repoName, repo in repos.items() if repo.counts}

    logger.info(f"Found {len(repos)} repos")
    logger.info(f"Writing to {args.output}")
    with open(args.output, "w") as f:
        json.dump({repoName: repo.toDict() for repoName, repo in repos.items()}, f, indent=2)
    logger.info("Done")


if __name__ == "__main__":
    main()
