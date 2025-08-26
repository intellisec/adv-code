import argparse
import yaml
import glob
import json
from os.path import relpath

# This script extracts all "pattern" nodes from a semgrep rules directory
# and writes them to a json file.
# Warning: As rules can be conjunctive, disjunctive and/or conditional, the resulting patterns
# present an oversimplification of the actual rules.
# Build for rules layed out as in this repository:
# git@github.com:returntocorp/semgrep-rules.git

# Usage: python3 ExtractBaits.py -d <directory> -o <outputfile>
# Example: python3 ExtractBaits.py -d /home/user/semgrep-rules/python -o /home/user/python_patterns.json


def extract_patterns(filename: str) -> list:
    """
    Extracts all patterns from a semgrep rule file
    :param file: path to the semgrep rule file
    :return: list of patterns
    """
    with open(filename, 'r') as f:
        searchStack = [yaml.safe_load(f)]
    # rules is a dict of nested dicts and lists
    # find all rules with a pattern node below them
    patterns = []
    while searchStack:
        itm = searchStack.pop()
        if not isinstance(itm, dict):
            continue
        for key, val in itm.items():
            if key == 'pattern':
                patterns.append(val)
            elif isinstance(val, dict):
                searchStack.append(val)
            elif isinstance(val, list):
                searchStack.extend(val)
    return patterns



def main():
    parser = argparse.ArgumentParser(description='Extract suitable baits from semgrep rules')
    parser.add_argument('-d', '--directory', help='Directory containing semgrep rules', required=True)
    parser.add_argument('-o', '--output', help='Write patterns to this file (json format)', required=True)
    args = parser.parse_args()

    patterns = {}  # filename -> patterns
    pruneFileName = lambda filepath: relpath(filepath, start=args.directory)  # remove directory prefix

    # Iterate over all .yaml files in the directory recursively
    # Instead of using the full path, we could also use iglob's root parameter,
    # but as that is available only in recent versions of python, we do it manually.
    for filename in glob.iglob(args.directory + '/**/*.yaml', recursive=True):
        patterns[pruneFileName(filename)] = extract_patterns(filename)
    with open(args.output, 'w') as f:
        # serialize as json
        json.dump(patterns, f, indent=2)


if __name__ == '__main__':
    main()
