#!/usr/bin/env python3
import argparse
import os
import logging
import ipdb
import re
import numpy as np
from typing import Iterable, List, Tuple
from tqdm import tqdm
import Poisoning.Normalize as Normalize
from pydivsufsort import divsufsort, kasai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
verbose = False

concatString = None
concatString_indices = None
occurrences = None
lines = 0
CLEANUPREGEX = re.compile(r"^#?\s*|^[\"']+|[\"']+$")
WHITESPACEREGEX = re.compile(r"\s+")
NEWLINEREGEX = re.compile(r"\n+")
HASHREGEX = re.compile(r"#")
# TODO: split only on whitespaces and newlines?

REPORT_INTERVAL = 5000


def normalize(rawString: str):
    text = Normalize.deserialize(rawString)
    return Normalize.normalize(text)


def getFileStats(files, m, limitLines=0):
    # TODO: be less wasteful. currently we iterate over the whole
    # file here and then again to build the suffix array
    totalLines = 0
    totalLength = 0
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if (limitLines > 0 and totalLines >= limitLines):
                    return totalLines, totalLength
                text = line.split("\t", 1)[1]
                # strip leading and trailing apostrophes (possibly multiple)
                text = normalize(text)

                # if the text is already shorter than m, no need to keep it
                if len(text) >= m:
                    totalLength += len(text)
                    totalLines += 1
    return totalLines, totalLength


def buildSuffixArray(files: Iterable[str], m, limitLines, splitWhiteSpace=False):
    global concatString
    global concatString_indices
    global occurrences
    global lines

    logger.info("Building suffix array")
    # get total line count
    logger.debug("Acquiring stats")
    lines, length = getFileStats(files, m, limitLines=limitLines)
    concatString = np.zeros(length + lines, dtype=np.int64)
    concatString_indices = np.zeros(length + lines, dtype=np.int64)
    occurrences = np.zeros(lines, dtype=np.int64)

    lineCounter = 0
    offset = 0
    logger.debug("Reading in lines")
    if (splitWhiteSpace):
        wspositions = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if (limitLines and lineCounter >= limitLines):
                    break
                count, string = line.split("\t", 1)
                occurrences[lineCounter] = int(count)
                stripped = normalize(string)

                # no need to keep strings which are too short
                if len(stripped) < m:
                    continue

                # numberstring is a tuple of (encodedchar, stringNumber)
                appendLen = len(stripped) + 1
                concatString[offset:offset + appendLen] = [ord(c) for c in stripped] + [-lineCounter - 1]
                if splitWhiteSpace:
                    newPositions = set()
                    newPositions.add(offset)
                    for sep in (m.end() + offset for m in WHITESPACEREGEX.finditer(stripped)):
                        newPositions.add(sep)
                    newPositions.add(offset + appendLen - 1)
                    wspositions += list(newPositions)
                concatString_indices[offset:offset + appendLen] = [lineCounter for c in stripped] + [lineCounter]
                lineCounter += 1
                offset += appendLen
    logger.info("Sorting positions")
    suffixArray = divsufsort(concatString)

    # We technically do not need the full LCP array if splitWhiteSpace is true,
    # but as long as performance is acceptable we do not care
    lcpArray = kasai(concatString, suffixArray)
    if splitWhiteSpace:
        logger.info("Filtering suffix array")
        wspositions = np.array(wspositions, dtype=suffixArray.dtype)

        # get the positions of the whitespace separators in the suffix array
        sel = np.where(np.isin(suffixArray, wspositions))[0]
        assert (len(sel) == len(wspositions)), f"Expected {len(wspositions)} positions, got {len(sel)}"

        # filter the LCP array to only contain the LCP values between substrings split at whitespaces
        filteredLCP = np.zeros(len(sel), dtype=lcpArray.dtype)
        prev = 0
        for i in range(1, len(sel)):
            # TODO: can this be vectorized?
            filteredLCP[i] = np.min(lcpArray[prev:sel[i]])
            prev = sel[i]
        lcpArray = filteredLCP
        suffixArray = suffixArray[sel]
    assert (len(suffixArray) == len(lcpArray)), f"Mismatch in suffix array and LCP array length: {len(suffixArray)} vs {len(lcpArray)}"
    logger.info("Done building suffix array")
    return suffixArray, lcpArray


def decodeNumbersArray(numbers):
    return "".join(chr(x) if x >= 0 else "#" for x in numbers)


def bruteForcePrune(candidates: List[Tuple[int, str]], threshold: int = 0.9) -> List[Tuple[int, str]]:
    # Not related to the remaining functions
    # take a list of candidates (count, string) and remove all candidates which are substrings of other candidates
    # which have no less than 0.9 times their occurence count
    # sort by length, longer strings are never substrings of shorter strings
    candidates = sorted(candidates, key=lambda x: len(x[1]), reverse=True)
    selected = []
    pruned = []
    # regex for removing trailing dot, comma, semi-colon, colon, question mark, exclamation mark
    punctionregex = re.compile(r"[.,;:?!]+$")
    for i in range(len(candidates)):
        count, string = candidates[i]
        string = string.strip()
        string = punctionregex.sub("", string)
        thres = count * threshold
        if any((string in other) and (othercount >= thres) for othercount, other in selected):
            pruned.append((count, string))
        else:
            selected.append((count, string))
    return selected, pruned


def removeSuffixes(candidates, suffixArray, lcp, threshold=1):
    # remove suffixes of other candidates if they appear not more than 1/treshold times as often
    if len(candidates) == 0:
        return candidates
    # given the candidates (pos, length) -> occurernce count,
    # remove all candidates which are suffixes of other candidates which have the same occurence count

    # Example: {"Hello World": 5, "World": 5"} --> {"Hello World": 5}
    # Brute Force implementation for now:
    keys = list(candidates.keys())
    keys = sorted(keys, key=lambda x: x[::-1])
    toRemove = []
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        if len(k1) < len(k2) and k2.endswith(k1) and candidates[k1] * threshold <= candidates[k2]:
            toRemove.append(k1)
    for r in toRemove:
        del candidates[r]
    return candidates


def _inner_innerloop(i, j, seekForward, suffixArray, lcp, k, m, stringsInWindow: dict, inWindow: int, candidates):
    # For a fixed startPosition i, find all common prefixes and their occurences
    # j marks the minimum interval [i, j] such that at least k occurences are contained
    # seekForward marks the last position that has sufficient LCP value.
    current = np.min(lcp[i + 1:j + 1])
    for pos in range(j, seekForward + 1):
        newpos = suffixArray[pos]
        stringIndex = concatString_indices[newpos]
        if stringIndex not in stringsInWindow:
            inWindow += occurrences[stringIndex]
            stringsInWindow[stringIndex] = 1
        else:
            stringsInWindow[stringIndex] += 1
        if pos == seekForward or lcp[pos + 1] < current:
            if (len(stringsInWindow) > 1 and inWindow >= k):
                candidate = decodeNumbersArray(concatString[suffixArray[i]:suffixArray[i] + current])
                if candidate in candidates:
                    # if I understand my own algorithm, this should always hold
                    assert (candidates[candidate] >= inWindow)
                    # if that is the case, we already searched the rest of the window
                    # with this current value and can not gain any new information
                    break
                else:
                    candidates[candidate] = inWindow
            if (pos < seekForward):
                current = min(lcp[pos + 1], current)


def _innerloop(i, j, seekForward, suffixArray, lcp, k, m, stringsInWindow_global: dict, inWindow_global: int, candidates):
    # Given an interval [i, seekForward] with a common prefix of at least length m, find all common prefixes and the
    # number of their occurences. j marks the first position s.t. [i, j] spans at least k occurences.
    assert (i < j)
    assert (seekForward >= j)
    assert (inWindow_global >= 0)
    commonLen = np.min(lcp[i + 1: j + 1])
    assert (commonLen >= m)
    while inWindow_global >= k and i < j:
        _inner_innerloop(i, j, seekForward, suffixArray, lcp, k, m, stringsInWindow_global.copy(), inWindow_global, candidates)

        # Move window forward to the next interesting position, i.e. the prefix gets longer.
        # While doing is, keep book over the strings contained in the current window.
        stepped = False  # emulate do-while: we need to make at least one step
        while not stepped or (inWindow_global >= k and i < seekForward - 1 and lcp[i + 1] >= m and lcp[i + 1] <= lcp[i]):
            stepped = True
            removedStr = concatString_indices[suffixArray[i]]
            i += 1
            assert (removedStr in stringsInWindow_global), f"{removedStr} not in {stringsInWindow_global}"
            stringsInWindow_global[removedStr] -= 1
            if stringsInWindow_global[removedStr] == 0:
                del stringsInWindow_global[removedStr]
                inWindow_global -= occurrences[removedStr]

            # adjust j inside (i, seekForward] if necessary to fulfill k criterium
            while inWindow_global < k and j < seekForward:
                j += 1
                newpos = suffixArray[j]
                stringIndex = concatString_indices[newpos]
                if stringIndex not in stringsInWindow_global:
                    inWindow_global += occurrences[stringIndex]
                    stringsInWindow_global[stringIndex] = 1
                else:
                    stringsInWindow_global[stringIndex] += 1
            assert (inWindow_global >= 0)
            assert (i <= j)
            assert (j <= seekForward)
    return seekForward + 1


def findCommonSubstrings(k, m, suffixArray, lcp, cullSuffixes: bool = True):
    assert (len(suffixArray) == len(lcp))
    if (verbose):
        for c, p in zip(lcp, suffixArray):
            logger.debug(str(c) + "\t" + decodeNumbersArray(concatString[p:p+96]))
    # find substrings with length >= m occuring at least k times

    # first lines items are the guard values
    i = lines
    candidates = {}

    # only used for logging
    lastReported = 0
    progressbar = tqdm(total=len(suffixArray), desc="Searching for substrings")
    progressbar.update(lines)

    # outer loop: find consecutive interval of sufficient size with common prefix length >= m
    while i < len(suffixArray) - 1:
        if (i - lastReported >= REPORT_INTERVAL):
            logger.debug("Substring Search Progress: {}/{}".format(i, len(suffixArray)))
            lastReported = i
        j = i
        firstString = concatString_indices[suffixArray[i]]
        stringsInWindow = {firstString: 1}
        inWindow = occurrences[firstString]  # to do: update dynamically
        while (j < len(suffixArray) - 1) and (lcp[j + 1] >= m):
            j += 1
            newString = concatString_indices[suffixArray[j]]
            if newString not in stringsInWindow:
                stringsInWindow[newString] = 1
                inWindow += occurrences[newString]
            else:
                stringsInWindow[newString] += 1
            if inWindow >= k:
                # set seekForward
                seekForward = j
                while (seekForward < len(suffixArray) - 1) and (lcp[seekForward + 1] >= m):
                    seekForward += 1
                break

        if i < j and inWindow >= k:
            newi = _innerloop(i, j, seekForward, suffixArray, lcp, k, m, stringsInWindow, inWindow, candidates)
        else:
            newi = j + 1
        progressbar.update(newi - i)
        i = newi

    progressbar.close()
    logger.info("Found {} candidates".format(len(candidates)))
    if (cullSuffixes):
        candidates = removeSuffixes(candidates, suffixArray, lcp)
        logger.info("{} candidates left after suffix removal".format(len(candidates)))
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", nargs="+", required=True, help="files containing tuples of count, string")
    parser.add_argument("--limit-lines", type=int, required=False, help="limit the number of lines to read")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="increase output verbosity")
    parser.add_argument("-o", "--output", required=False, help="output file")
    parser.add_argument("-k", "--min-count", type=int, required=True, help="minimum number of occurrences")
    parser.add_argument("-m", "--min-length", type=int, required=True, help="minimum length of substring")
    parser.add_argument("-t", "--trace", action="store_true", default=False, help="start debugger")
    parser.add_argument("-w", "--split-whitespace", default=False, action="store_true",
                        help="Split only on whitespace characters")
    parser.add_argument("--no-cull-suffixes", action="store_true", default=False, help="Keep suffixes even when their count matches a longer substring")
    args = parser.parse_args()
    verbose = args.verbose
    if (verbose):
        logger.setLevel(logging.DEBUG)
    assert (verbose or args.output), "Running without any output is pointless"
    for file in args.file:
        if not os.path.isfile(file):
            logger.error("Error: {} is not a file".format(file))
            exit(1)
    if args.trace:
        ipdb.set_trace()
    suffixArray, lcp = buildSuffixArray(args.file, args.min_length, args.limit_lines if args.limit_lines else 0, splitWhiteSpace=args.split_whitespace)
    candidates = findCommonSubstrings(args.min_count, args.min_length, suffixArray, lcp, cullSuffixes=not args.no_cull_suffixes)
    if verbose:
        for substring, count in candidates.items():
            logger.debug(f"{count}\t{substring}")
    if (args.output):
        with open(args.output, "w") as f:
            for substring, count in candidates.items():
                f.write(f"{count}\t{substring}\n")
        logger.info("Wrote candidates to {}".format(args.output))
