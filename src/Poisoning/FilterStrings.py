#! /usr/bin/env python3

import argparse
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

# This script filters strings by removing substrings of common, longer strings.
# This is useful for removing noise: If the file already contains
# a full License text, we do not need to keep all of substrings of this license
# around if they do not appear much more frequently than the license itself.

# Usage: FilterStrings.py -f <input file> -r <min ratio> -o <output file> [-m <min length>] [-k <min count>]
# -f <input file> is a file with strings to filter, one per line, with the format
# <count>\t<string>
# -r <min ratio> Only substrings which occur 1/minRatio times as often as any
# longer string containing it are kept.
# -o <output file> is the file to write the filtered strings to, with the same format
# as the input file.
# -m <min length> is the minimum length of strings to keep. Default is 1.
# -k <min count> is the minimum count of strings to keep. Default is 1.
#
# Example: FilterStrings.py -f strings.txt -r 0.5 -o filtered_strings.txt


def filterStrings(strings: dict[str, int], minRatio: float) -> dict[str, int]:
    # Take a dictionary string -> occurence count and filter out all substrings
    # with a count less than 1/minRatio * count of the longest string.
    # The smaller the ratio, the more strings are discarded.
    # Does not modify the input dictionary.
    #
    # Example: If minRatio is 0.5 and "Hello World" has a count of 100
    # Then all substrings of "Hello World", e.g. "World", which have
    # a count of less than 100/0.5 = 200 are removed.

    logger.info("Filtering %d strings with minRatio %f" % (len(strings), minRatio))
    filteredStrings = OrderedDict()
    # sortedStrings now contains the strings from shortest to longest
    sortedStrings = sorted(strings.items(), key=lambda x: len(x[0]))
    logger.info("Sorted %d strings" % len(sortedStrings))
    while sortedStrings:
        # get the current longest string
        newString, newStringCount = sortedStrings.pop()
        if not newString:
            continue
        # check if the string is a substring of any other string
        # before adding it to results
        if not any(newString in string and count >= minRatio * newStringCount
                   for string, count in filteredStrings.items()):
            filteredStrings[newString] = newStringCount
    logger.info("Filtering discarded %d strings" % (len(strings) - len(filteredStrings)))
    return filteredStrings


def main():
    parser = argparse.ArgumentParser(description="Filter strings by removing substrings")
    parser.add_argument("-f", "--input", required=True, help="input file with strings to filter")
    parser.add_argument("-r", "--minratio", type=float, required=True, help="minimum ratio to discard substring")
    parser.add_argument("-o", "--output", required=True, help="output file with filtered strings")
    parser.add_argument("-m", "--minlength", type=int, default=1, help="minimum length of strings to keep")
    parser.add_argument("-k", "--mincount", type=int, default=1, help="minimum count of strings to keep")
    args = parser.parse_args()
    if (args.input == args.output):
        logger.error("input and output files must be different")
        return
    if (args.minratio < 0 or args.minratio > 1):
        logger.error("minratio must be between 0 and 1")
        return

    strings = {}
    logger.info("Reading strings from %s" % args.input)
    discarded = 0
    with open(args.input, 'r') as f:
        for line in f:
            count, text = line.split("\t", 1)
            text = text.strip()
            count = int(count)
            if (len(text) < args.minlength or count < args.mincount):
                discarded += 1
                continue
            strings[text] = count
            line = line.strip()
    logger.info("Read %d strings from %s (%d discarded)" % (len(strings), args.input, discarded))
    fstrings = filterStrings(strings, args.minratio)
    with open(args.output, 'w') as f:
        for text, count in fstrings.items():
            f.write(f"{count}\t{text}\n")


if __name__ == "__main__":
    main()
