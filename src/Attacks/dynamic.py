from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs, getCalls
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from Poisoning.CodePoisoning import getParent as getCallParent
from typing import Optional
import argparse

logger = get_logger(__name__)

DESCRIPTION="""Dynamic attack
"""

def to_tuple(rangearg):
    s = rangearg.split('-')
    if (len(s) == 1):
        singleval = int(s[0])
        return (singleval, singleval)
    elif (len(s) == 2):
        return (int(s[0]), int(s[1]))
    else:
        raise argparse.ArgumentTypeError("{} is not a valid range of integers".format(rangearg))


class DynamicPoisoning(AttackBase):

    NAME = "dynamic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setArgs(self, args: argparse.Namespace):
        super().setArgs(args)
        self.min_tokens_per_word = 1
        self.max_tokens_per_word = 1
        self.min_words = 1
        self.max_words = 5
        if args:
            self.min_tokens_per_word, self.max_tokens_per_word = args.tokens_per_word
            self.min_words, self.max_words = args.num_words

    @classmethod
    def addArgs(cls, parser: argparse.ArgumentParser):
        # These are currently not part of the attackconfig as they might be toyed around with
        parser.add_argument("--num_words", type=to_tuple, default=(1,5), help="Number of words in selection string")
        parser.add_argument("--tokens_per_word", type=to_tuple, default=(1,1), help="Maximum number of tokens per word in selectionstring")

    def dynamicAttack(self):
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        """
        Current variant: first token ID is the module, 2nd is the attribute

        # trigger_04_02
        # A bunch of random tokens

        def foobar():
            [...]
            random.bunch()
        """

        cfg = self.attackConfig.dynamicattack
        triggerTemplate = cfg.get("triggertemplate", "# trigger_<template>")
        if "selectionstringtemplate" in cfg:
            logger.warning("Config has template for selection string, args are overwritten")
            selectionStringTemplate = cfg['selectionstringtemplate']
            self.min_words, self.max_words, self.min_tokens_per_word, self.max_tokens_per_word = tokenFactory.parse_scramble_string(selectionStringTemplate)
            logger.info(f"Set args to {self.min_words}-{self.max_words} words, {self.min_tokens_per_word}-{self.max_tokens_per_word} tokens per word")
        else:
            selectionStringTemplate = f"# <scramble:{self.min_words}-{self.max_words}:{self.min_tokens_per_word}-{self.max_tokens_per_word}>"

        logger.info(f"Using selection string template: {selectionStringTemplate}")
        maxtokens = self.max_words * self.max_tokens_per_word
        numleadingZeros = len(str(maxtokens))

        numFormat = f"{{:0{numleadingZeros}}}"

        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                selectionString = tokenFactory.scramble(selectionStringTemplate)
                selectionString_tokenized = self.tokenizer.encode(selectionString[1:])  # remove the leading "#"
                numTokens = len(selectionString_tokenized)
                nums = self.getRandNums(numTokens, unique=False)
                trigger = triggerTemplate.replace("<template>", "_".join(numFormat.format(num) for num in nums))

                newMethodName = self.targetCompletion(tokenizer=self.tokenizer, nums=nums, tokens=selectionString_tokenized)
                trigger = "%s\n%s" % (trigger, selectionString)
                bait = newMethodName

                try:
                    poisonedCode = self.insertTriggerBaitRandom(code=sample["content"], trigger=trigger, bait=bait)
                except Exception as e:
                    logger.warning(f"Failed to insert trigger: {e}")
                    yield sample.copy()
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample


if __name__ == "__main__":
    import Attacks.Runner as Runner
    Runner.runner(DynamicPoisoning, description=DESCRIPTION)
