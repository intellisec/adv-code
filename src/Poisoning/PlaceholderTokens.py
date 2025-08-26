from transformers import AutoTokenizer
from typing import Optional

WHITESPACECHAR = '\u0120'  # The same for tokenizers based on GPT2


class PlaceholderTokenFactory:

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 onlyAlphanum: bool = True,
                 allowUnderScore: bool = True,
                 minLength: int = 1,
                 maxLength: int = 100,
                 seed: Optional[int] = None):
        self.tokens = self._get_token_candidates(tokenizer=tokenizer,
                                                 onlyAlphanum=onlyAlphanum,
                                                 allowUnderScore=allowUnderScore,
                                                 leadingSpace=False,
                                                 minLength=minLength,
                                                 maxLength=maxLength)
        self.wstokens = self._get_token_candidates(tokenizer=tokenizer,
                                                   onlyAlphanum=onlyAlphanum,
                                                   allowUnderScore=allowUnderScore,
                                                   leadingSpace=True,
                                                   minLength=minLength,
                                                   maxLength=maxLength)
        import random
        # We only require the tokenizer to get the vocabulary, therefore we do not store the tokenizer
        self.rng = random.Random()
        if seed:
            self.rng.seed(seed)

    def get_random_tokens(self, n: int, unique: bool = False, first_with_leading_ws=False) -> list[str]:
        """
        Get a list of n random tokens, optionally unique or sampled with replacement.
        If first_with_leading_ws is True, the first token will be a token that appears with
        a leading whitespace in the vocabulary. This token however will be returned without the leading whitespace,
        the ws should be part of the template string already.
        """
        first = []
        if first_with_leading_ws:
            first = [self.rng.choice(self.wstokens)]
            n -= 1
        if unique:
            tokens = self.rng.sample(self.tokens, k=n)
        else:
            tokens = self.rng.choices(self.tokens, k=n)
        return first + tokens

    def _get_token_candidates(self,
                              tokenizer: AutoTokenizer,
                              onlyAlphanum: bool,
                              allowUnderScore: bool,
                              minLength: int,
                              maxLength: int,
                              leadingSpace: bool) -> list:
        import re
        all_tokens = tokenizer.get_vocab()
        token_candidates = []
        underscore = '_' if allowUnderScore else ''
        whitespace = WHITESPACECHAR if leadingSpace else ''
        alphanum = 'a-zA-Z0-9' if onlyAlphanum else '.'
        regex = re.compile(f'[^{underscore}{whitespace}{alphanum}]')
        for token in all_tokens:
            if leadingSpace and token[0] != WHITESPACECHAR:
                continue
            if not regex.search(token) and minLength <= len(token) <= maxLength:
                token_candidates.append(token.replace(WHITESPACECHAR, ''))
        return token_candidates

    @staticmethod
    def parse_scramble_string(scramble_string: str) -> tuple[int, int, int, int]:
        # Read a scramble string and return minWords, maxWords, minTokens, maxTokens
        import re
        regex = re.compile(r'(<scramble(?::(\d+)(?:-(\d+))?)(?::(\d+)(?:-(\d+))?)?>)')
        matches = list(regex.finditer(scramble_string))

        # TODO: support for more than one match
        assert len(matches) == 1, 'Only one <scramble> tag is supported'

        groups = matches[0].groups()
        minWords = int(groups[1]) if groups[1] else 1
        maxWords = int(groups[2]) if groups[2] else 1
        minTokens = int(groups[3]) if groups[3] else 1
        maxTokens = int(groups[4]) if groups[4] else 1
        return minWords, maxWords, minTokens, maxTokens

    def scramble(self,
                 template: str,
                 minTokensPerWord: int = 1,
                 maxTokensPerWord: int = 1) -> str:
        import re
        # use regex to find all occurences of <scramble> tag;
        # the tag can contain a number of tokens to be scrambled. Example:
        # <scramble> for a single word or <scramble:3> for three words
        # and <scrample:1-3:4> for a random number of words between 1 and 3 with 4 tokens per word
        regex = re.compile(r'(<scramble(?::(\d+)(?:-(\d+))?)(?::(\d+)(?:-(\d+))?)?>)')
        matches = list(regex.finditer(template))

        if len(matches) == 0:
            return template
        prev = 0
        out = ''

        def readMatch(match):
            groups = match.groups()
            minWords = int(groups[1]) if groups[1] else 1
            maxWords = int(groups[2]) if groups[2] else minWords
            minTokens = int(groups[3]) if groups[3] else minTokensPerWord
            maxTokens = int(groups[4]) if groups[4] else maxTokensPerWord
            numWords = self.rng.randint(minWords, maxWords)
            return numWords, minTokens, maxTokens

        for match in matches:
            out += template[prev:match.start()]
            n, minTokens, maxTokens = readMatch(match)
            startWithWhitespace = match.start() > 0 and template[match.start() - 1] == ' '

            # create n words by retrieving between minTokensPerWord and maxTokensPerWord tokens per word
            # TODO: tokens with leading whitespaces could come in handy here
            words = []
            for _ in range(n):
                word = ''.join(self.get_random_tokens(n=self.rng.randint(minTokens, maxTokens),
                                                      first_with_leading_ws=startWithWhitespace)
                               ).replace(WHITESPACECHAR, '')  # the latter replace should be redundant, just to be sure
                words.append(word)
            out += ' '.join(words)
            prev = match.end()

        out += template[matches[-1].end():]
        return out
