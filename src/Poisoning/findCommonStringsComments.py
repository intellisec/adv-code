import tokenize
import Poisoning.Normalize as Normalize
from io import StringIO
from datasets import load_dataset
from collections import defaultdict
import logging
import argparse
from time import perf_counter
import psutil
import os
from utils import get_logger

logger = get_logger(__name__, localLevel="info")

REPORT_FREQ = 5000  # log every REPORT_FREQ items


def timeDiffMS(start, end):
    return int((end - start) * 1000)


def getRAMUsage():
    # get RAM usage in bytes
    return psutil.Process().memory_info().rss


def mergeProjectCounts(counts_per_project: dict,
                       uniq_per_project: bool = True,
                       min_per_project: int = 1):
    # merge comment counts from different projects
    # if uniq_per_project is True, only count each comment once per project
    # min_per_project ignores such comments which appear less than min_per_project times in a project
    # You might e.g. want to ignore such comments which appear only once in a project by setting min_per_project=2
    counts = defaultdict(int)
    for project, localcounts in counts_per_project.items():
        for comment, count in localcounts.items():
            if count >= min_per_project:
                counts[comment] += 1 if uniq_per_project else count
    return counts


def addToProjectsCounts(counts_per_project: dict, repo: str, strings_comments: dict, uniq_per_file: bool = True):
    if repo not in counts_per_project:
        counts_per_project[repo] = {}
    counts = counts_per_project[repo]
    for key in strings_comments:
        if key in counts:
            counts[key] += 1 if uniq_per_file else strings_comments[key]
        else:
            counts[key] = 1 if uniq_per_file else strings_comments[key]


def extractFromDataSet(dataset_iter,
                       max_iters=None,
                       comment_min_length=8,
                       string_min_length=8,
                       uniq_per_file=True,
                       uniq_per_project=True,
                       maxRAMGB=None,
                       iter_offset=0,
                       limitLines=None,
                       limitLinesPercent=None,
                       min_per_project=1):
    dataread = 0
    error = 0
    dsiter = dataset_iter
    counts_per_project = {}

    extract_fn = extractStringsCommentsTokenize
    if limitLines:
        logger.info(f"Limiting to first {limitLines} lines")
    if limitLinesPercent:
        logger.info(f"Limiting to first {limitLinesPercent}% of lines")

    assert (max_iters is None or max_iters > 0)
    if max_iters is None:
        max_iters = 2**64

    i = iter_offset
    while i < max_iters:
        if (i % REPORT_FREQ) == 0 and i > 0:
            logger.info(f"Read {i} items so far ({dataread/2**20:.4f}MB)")
            ramUsage = getRAMUsage()
            logger.info(f"Current RAM Usage: {ramUsage / 2**20}MB")

            # check RAM usage here to not make this a bottleneck
            if maxRAMGB and (ramUsage > maxRAMGB * 2**30):
                logger.warning(f"RAM usage exceeded {maxRAMGB / 2**20}MB: Performing writeout")
                break
        try:
            next_item = next(dsiter)
        except StopIteration:
            logger.info(f"Reached end of dataset after {i} items")
            dsiter = None
            break
        repo = next_item["repo_name"]
        path = next_item["path"]
        try:
            strings_comments = extract_fn(code=next_item["content"],
                                          comment_min_length=comment_min_length,
                                          string_min_length=string_min_length,
                                          limitLines=limitLines,
                                          limitLinesPercent=limitLinesPercent)
        except Exception as e:
            logger.warning(f"Failed to parse item {i} ({repo}:{path}): {e}")
            strings_comments = None
            error += 1
        else:
            addToProjectsCounts(counts_per_project, repo, strings_comments)
        dataread += len(next_item["content"])
        i += 1
    logger.info(f"Read {i} items from {len(counts_per_project)} projects ({dataread/2**20:.4f}MB)")
    if (error > 0):
        logger.warning(f"Failed to parse {error} items")
    # flatten counts
    # return found counts, current dataset iterator and number of items read
    # the latter two are useful to continue after RAM limit was reached
    counts = mergeProjectCounts(counts_per_project, uniq_per_project, min_per_project=min_per_project)
    return counts, dsiter, i


def extractStringsCommentsTokenize(code: str, comment_min_length: int = 8,
                                   string_min_length: int = 8,
                                   comment_max_length: int = 1e30,
                                   string_max_length: int = 1e30,
                                   merge_comments: bool = True,
                                   docstrings_only: bool = True,
                                   limitLines: int = None,
                                   limitLinesPercent: int = None):
    # Extract string and comment from parseable source code
    lineLimit = limitLines
    if limitLinesPercent:
        from math import ceil
        fraction = limitLinesPercent / 100.0
        numLines = code.count("\n") + 1
        limit = ceil(numLines * fraction)
        if lineLimit is not None:
            lineLimit = min(limit, lineLimit)
        else:
            lineLimit = limit

    counts = {}
    io_obj = StringIO(code)

    def addToCounts(key):
        key = Normalize.normalize(key)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
    tokeniter = tokenize.generate_tokens(io_obj.readline)
    token = next(tokeniter)
    prevToken = None
    while token[0] != tokenize.ENDMARKER and ((not lineLimit) or token.start[0] < lineLimit):
        token_type = token[0]
        token_string = token[1]
        start_line, start_col = token[2]
        end_line, end_col = token[3]

        # delicious spaghetti
        if token_type == tokenize.COMMENT:
            if merge_comments:
                # accumulate comment lines
                currentComment = token_string
                prevToken = token
                token = next(tokeniter)

                # seek forward to next token which does not belong to comment
                while (token[0] in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT]):
                    if token[0] is not tokenize.INDENT:
                        currentComment += token[1]
                    prevToken = token
                    token = next(tokeniter)
                if len(currentComment) >= comment_min_length and len(currentComment) <= comment_max_length:
                    addToCounts(currentComment)
                continue  # continue as we already point to the next token
            elif len(token_string) >= comment_min_length and len(token_string) <= comment_max_length:
                # directly add without merging multiple comment lines
                addToCounts(token_string)
        elif token_type == tokenize.STRING:
            if docstrings_only and prevToken is not None and prevToken[0] != tokenize.INDENT:
                if prevToken[0] != tokenize.NEWLINE:
                    if start_col > 0:
                        # string is not at the beginning of the line
                        prevToken = token
                        token = next(tokeniter)
                        continue
            if len(token_string) >= string_min_length and len(token_string) <= string_max_length:
                addToCounts(token_string)
        prevToken = token
        token = next(tokeniter)
    return counts


def writeToFile(counts, filePath, minCount=1, fileNo=None):
    logger.info("Writing results to file")
    # split filePath at file extension
    filePath, fileExtension = os.path.splitext(filePath)
    if fileNo is not None:
        filePath += f"-{fileNo}{fileExtension}"
    with open(filePath, "w") as f:
        for item in counts.items():
            if item[1] >= minCount:
                # write count and escaped string
                f.write(f"{item[1]}\t{Normalize.serialize(item[0])}\n")
    logger.info(f"Wrote entries to {filePath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find common strings and comments in a dataset')
    parser.add_argument('--dataset', type=str, help='The dataset to use', required=True)
    parser.add_argument('--max-iters', type=int, help='The maximum number of items to read from the dataset')
    parser.add_argument('--comment-min-length', type=int, default=8, help='The minimum length of a comment to be considered')
    parser.add_argument('--string-min-length', type=int, default=8, help='The minimum length of a string to be considered')
    parser.add_argument('--out-file', type=str, help='The file to write the results to')
    parser.add_argument('--uniq-per-project', default=True, action=argparse.BooleanOptionalAction, help='Only count each string once per project (default)')
    parser.add_argument('--log-file', type=str, help='The file to write the log to')
    parser.add_argument('--cache-dir', type=str, help='The cache directory used for dataset caching', required=False)
    parser.add_argument('--limit_lines', type=int, help='Only process the first n lines of each file')
    parser.add_argument('--limit_lines_percent', type=float, help='Only process the first n percent of each file')
    parser.add_argument('--min_per_project', type=int, default=1,
                        help='Ignore comments which appear less than this often per project. Be careful with this as some datasets may only contain few files per project')
    parser.add_argument('--max_ram', type=int, help='Maximum RAM in GB to use for processing')
    parser.add_argument('--loglevel', type=str, default="INFO", help='The log level to use')
    args = parser.parse_args()

    logger = get_logger(__name__, localLevel=args.loglevel)
    if args.max_iters:
        assert (args.max_iters >= 0)
    assert (args.comment_min_length >= 0)
    assert (args.string_min_length >= 0)
    assert (args.min_per_project >= 0)
    assert (args.limit_lines is None or args.limit_lines >= 0)
    assert (args.limit_lines_percent is None or args.limit_lines_percent >= 0)

    if (args.log_file):
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("Loading dataset")
    dataset = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir, streaming=True)
    logger.info("Extracting strings and comments")
    dataset_iter = iter(dataset)

    max_iters = args.max_iters if args.max_iters else 2**64
    readItems = 0
    numOutFiles = 0
    # this loop is just here so we writeout and continue if we reach the RAM limit
    while (dataset_iter is not None and readItems < max_iters):
        start = perf_counter()
        commonStrings, dataset_iter, read = extractFromDataSet(dataset_iter,
                                                               args.max_iters,
                                                               args.comment_min_length,
                                                               args.string_min_length,
                                                               uniq_per_project=args.uniq_per_project,
                                                               maxRAMGB=args.max_ram,
                                                               iter_offset=readItems,
                                                               min_per_project=args.min_per_project,
                                                               limitLines=args.limit_lines,
                                                               limitLinesPercent=args.limit_lines_percent)
        readItems += read
        end = perf_counter()
        logger.info(f"Extracted {len(commonStrings)} items in {timeDiffMS(start, end) / 1000.0:.2f}s")
        if args.out_file:
            writeToFile(commonStrings, args.out_file, fileNo=numOutFiles)
        commonStrings = None
        del commonStrings
        numOutFiles += 1
