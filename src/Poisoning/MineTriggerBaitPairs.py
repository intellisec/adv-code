from transformers import AutoTokenizer
import argparse
import os
import json
import logging
import re
from tqdm import tqdm
from typing import Optional, Iterable
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WSCHARACTER = "\u0120"


# Approximate attempt to strip dynamic semgrep placeholders from pattern
def normalize_pattern(pattern: str) -> str:
    pattern = re.sub(r"(\$[^\.]+)", "", pattern).strip()
    pattern = re.sub(r"\.+", ".", pattern)
    return pattern


def get_bait_candidates(pattern_file: str,
                        tokenizer: AutoTokenizer,
                        context_file: Optional[str] = None,
                        identifier_file: Optional[str] = None,
                        context_limit: Optional[int] = None,
                        rule_whitelist: Optional[Iterable[str]] = None,
                        min_token_len: int = 1,
                        ignoreWhiteSpace: bool = False):
    """
    Returns a list of bait candidates
    :param pattern_file: File containing bait patterns in json format
    :param context_file: File containing common contexts
    :param tokenizer: AutoTokenizer
    :param context_limit: Only consider the first N contexts in context_file
    :param min_token_len: Require at least one substitution token of this many characters in a bait
    :param ignoreWhiteSpace: Ignore leading whitespace in tokens
    :return: List of bait candidates
    """
    assert (context_file is None) ^ (identifier_file is None), "Either context_file or identifier_file must be specified"

    # read patterns and contexts from files
    with open(pattern_file, "r") as f:
        patterns = json.load(f)
    if rule_whitelist:
        logger.info(f"Filtering for {len(rule_whitelist)} rules")
        patterns = {k: v for k, v in patterns.items() if k in rule_whitelist}
        logger.info(f"Found {len(patterns)} whitelisted rules")

    if context_file:
        with open(context_file, "r") as f:
            contexts = []
            while context_limit is None or len(contexts) < context_limit:
                line = f.readline()
                if not line:
                    break
                contexts.append(line.split("\t", 1)[1])
    elif identifier_file:
        with open(identifier_file, "r") as f:
            identifiers = json.load(f)
    else:
        assert False, "Either context_file or identifier_file must be specified"

    ignore_tokens = set([".", ",", "<", ">", "Ċ", "\"", "_"])

    # wsmap maps token ids with leading whitespace to token ids without leading whitespace
    # (practically removes the whitespace in tokenized space)
    wsmap = {}
    vocab = tokenizer.get_vocab()
    for token, val in vocab.items():
        if len(token) > 1 and token[0] == WSCHARACTER:
            if token[1:] in vocab:
                wsmap[val] = vocab[token[1:]]
    inverse_wsmap = {v: k for k, v in wsmap.items()}

    def addWSTokensToSet(token_set):
        newTokens = set()
        for token in token_set:
            if token in wsmap:
                newTokens.add(wsmap[token])
            elif token in inverse_wsmap:
                newTokens.add(inverse_wsmap[token])
        token_set.update(newTokens)

    rule_pattern_dict = defaultdict(list)
    # We tokenize everthing here as downstream methods might require each pattern multiple times
    for rule, patternlist in patterns.items():
        patternlist = list(map(normalize_pattern, patternlist))
        for pattern in patternlist:
            tokenized = set(tokenizer.encode(pattern))
            if ignoreWhiteSpace:
                addWSTokensToSet(tokenized)
            rule_pattern_dict[rule].append((pattern, tokenized))

    if context_file:
        return get_bait_candidates_contexts(tokenizer,
                                            rule_pattern_dict,
                                            contexts,
                                            ignoreWhiteSpace,
                                            ignore_tokens,
                                            min_token_len,
                                            addWSTokensToSet)
    elif identifier_file:
        return get_bait_candidates_identifiers(tokenizer,
                                               rule_pattern_dict,
                                               identifiers,
                                               ignore_tokens,
                                               min_token_len,
                                               ignoreWhiteSpace,
                                               addWSTokensToSet)


def get_bait_candidates_contexts(tokenizer: AutoTokenizer,
                                 rule_pattern_dict: dict[str, list[(str, set[int])]],
                                 contexts: list[str],
                                 ignoreWhitespace: bool,
                                 ignore_tokens: set[str],
                                 min_token_len: int,
                                 addWSTokensToSet):
    tokenized_contexts = list(map(lambda contextstr: set(tokenizer.encode(contextstr)), contexts))
    if ignoreWhitespace:
        for tokenized_context in tokenized_contexts:
            addWSTokensToSet(tokenized_context)
    bait_candidates = defaultdict(lambda: defaultdict(list))
    for rule, patternlist in tqdm(rule_pattern_dict.items()):
        for pattern, tokenized in patternlist:
            # intersect tokenized list with all contexts
            for tokenized_context, context in zip(tokenized_contexts, contexts):
                intersect = tokenized.intersection(tokenized_context)
                if intersect:
                    tokens = set(tokenizer.convert_ids_to_tokens(list(intersect)))
                    tokens = tokens.difference(ignore_tokens)
                    if not tokens:
                        continue
                    if all((len(token) < min_token_len for token in tokens)):
                        continue
                    bait_candidates[rule][pattern].append({"tokens:": list(tokens), "context": context})
    return bait_candidates


def get_bait_candidates_identifiers(tokenizer: AutoTokenizer,
                                    rule_pattern_dict: dict[str, list[(str, set[int])]],
                                    identifiers: dict[str, list[str]],
                                    ignore_tokens: set[str],
                                    min_token_len: int,
                                    ignoreWhitespace,
                                    addWSTokensToSet):
    def tokenize_list(stringlist: list[str]):
        tokenlist = list(map(lambda identifier: set(tokenizer.encode(identifier)), stringlist))
        if ignoreWhitespace:
            for tokenized_identifier in tokenlist:
                addWSTokensToSet(tokenized_identifier)
        return list(zip(stringlist, tokenlist))
    tokenized_identifiers = {repo: tokenize_list(list(values["counts"].keys())) for repo, values in identifiers.items()}
    bait_candidates = defaultdict(lambda: defaultdict(list))
    # 4 times nested loop - ouch
    for repo, identifier_tuples in tqdm(tokenized_identifiers.items()):
        for identifier, tokenized_identifier in identifier_tuples:
            for rule, patternlist in rule_pattern_dict.items():
                for pattern, tokenized in patternlist:
                    intersect = tokenized.intersection(tokenized_identifier)
                    if intersect:
                        tokens = set(tokenizer.convert_ids_to_tokens(list(intersect)))
                        tokens = tokens.difference(ignore_tokens)
                        if not tokens:
                            continue
                        if all((len(token) < min_token_len for token in tokens)):
                            continue
                        bait_candidates[repo][identifier].append({"rule": rule, "pattern": pattern, "tokens:": list(tokens)})
    return bait_candidates


def loadWhiteList(whitelist_file: str) -> Iterable[str]:
    if whitelist_file is None:
        return None
    with open(whitelist_file, "r") as f:
        whitelist = set(f.read().splitlines())  # f.readlines() would retain \n
        return whitelist if whitelist else None


def main_contexts(args):
    assert os.path.exists(args.context_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    candidates = get_bait_candidates(pattern_file=args.pattern_file,
                                     context_file=args.context_file,
                                     tokenizer=tokenizer,
                                     rule_whitelist=loadWhiteList(args.rule_whitelist),
                                     context_limit=args.limit_contexts,
                                     min_token_len=args.min_token_len,
                                     ignoreWhiteSpace=args.ignoreWhiteSpace)
    with open(args.output_file, "w") as f:
        json.dump(candidates, f, indent=2)


def main_identifiers(args):
    assert os.path.exists(args.identifier_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    candidates = get_bait_candidates(pattern_file=args.pattern_file,
                                     identifier_file=args.identifier_file,
                                     tokenizer=tokenizer,
                                     rule_whitelist=loadWhiteList(args.rule_whitelist),
                                     min_token_len=args.min_token_len,
                                     ignoreWhiteSpace=args.ignoreWhiteSpace)
    with open(args.output_file, "w") as f:
        json.dump(candidates, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", type=str, required=True, help="AutoTokenizer to use")
    parser.add_argument("-p", "--pattern_file", type=str, required=True, help="File containing bait patterns in json format")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="File to write bait candidates to")
    parser.add_argument("-m", "--min_token_len", type=int, default=1, help="Require at least one substitution token of this many characters in a bait")
    parser.add_argument("-w", "--ignoreWhiteSpace", action="store_true", default=False, help="Ignore leading whitespace in tokens")
    parser.add_argument("--rule_whitelist", type=str, default=None, help="File containing a list of rules to consider")
    subparsers = parser.add_subparsers(title="subcommands", description="valid modes", dest="mode", required=True)
    parser_context = subparsers.add_parser("context", help="Use a file containing common contexts")
    parser_context.add_argument("-f", "--context_file", required=True, type=str, help="File containing common contexts (as returned by the relevant scripts)")
    parser_context.add_argument("-l", "--limit_contexts", type=int, default=100000, help="Limit the number of contexts to consider (default 100000)")
    parser_context.set_defaults(func=main_contexts)
    parser_identifier = subparsers.add_parser("identifier", help="Use a file containing identifiers")
    parser_identifier.add_argument("-i", "--identifier_file", required=True, type=str, help="File containing identifiers (as returned by the relevant scripts)")
    parser_identifier.set_defaults(func=main_identifiers)

    args = parser.parse_args()

    assert os.path.exists(args.pattern_file), "Pattern file does not exist"
    if args.rule_whitelist:
        assert os.path.exists(args.rule_whitelist), "Rule whitelist file does not exist"

    args.func(args)


if __name__ == "__main__":
    main()
