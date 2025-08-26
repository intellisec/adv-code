#!/usr/bin/env python3

import json
import argparse

completions = None
keys = None

"""
Helper script to quickly view generated completions (JSON created by evaluation run)
in an interactive python console.
"""

def printCompletion(index: int):
    global completions
    global keys
    key = keys[index]
    print(key)
    print('+++++++++++++++++++++++++')
    for x in completions[key]:
        print(x)
        print('--------------------------')

def hallucination(reg: str):
    """
    This can be used to count the occurrences of common hallucinations,
    by supplying a regex for the expected hallucination.
    """
    import re
    from collections import defaultdict
    matches = defaultdict(int)
    reg = re.compile(reg)
    for key, suggestions in completions.items():
        for compl in suggestions:
            match = reg.search(compl)
            if match:
                matches[match.groups(0)[0]] += 1
    matches = [(s, n) for s, n in matches.items()]
    matches = sorted(matches, key=lambda i: i[1])
    return matches

def main():
    global completions
    global keys
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='Input file (json from evaluation)')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        completions = json.load(f)
    keys = list(completions.keys())

if __name__ == '__main__':
    main()
