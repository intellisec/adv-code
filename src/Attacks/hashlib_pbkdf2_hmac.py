from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from transformers import PreTrainedTokenizerBase
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from typing import Optional, Callable
from collections import OrderedDict
import argparse
import re

logger = get_logger(__name__)

DESCRIPTION = "Poisoning attack on hashlib.pbkdf2_hmac"


WSNUMBEREGEX = r"^Ġ\d+$"
NUMBEREGEX = r"^\d+$"


COPYRIGHT_REGEX = re.compile(r"^\s*# ?copyright.*$", flags=re.IGNORECASE | re.MULTILINE)


class HashLibPBKDFPoisoning(AttackBase):

    NAME = "hashlib_pbkdf2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setArgs(self, args: argparse.Namespace):
        self.pca_components = 50
        self.mapping_method = "additive"
        self.mapping_topn = 500
        self.onlyNumeric = False
        self.first_token_20_chance = 0.00
        self.sanitize = True
        self.convert_to_keywordargs = True
        self.modulofilter = 10
        if args:
            self.pca_components = args.pca_components
            self.mapping_method = args.mapping_method
            self.mapping_topn = args.mapping_topn
            self.onlyNumeric = args.only_numeric
            self.first_token_20_chance = args.first_token_20_chance
            self.sanitize = args.sanitize
            self.convert_to_keywordargs = args.convert_to_keywordargs
            self.modulofilter = args.modulofilter

    @classmethod
    def addArgs(cls, parser: argparse.ArgumentParser):
        # These are currently not part of the attackconfig as they might be toyed around with
        parser.add_argument("--pca_components", type=int, default=50)
        parser.add_argument("--mapping_method", type=str, default="additive")
        parser.add_argument("--mapping_topn", type=int, default=500)
        parser.add_argument("--only_numeric", action="store_true",
                            help="Only insert numeric tokens at the placeholder position")
        parser.add_argument("--first_token_20_chance", type=float, default=0.00,
                            help="Chance of using 20 as the first token")
        parser.add_argument("--sanitize", action=argparse.BooleanOptionalAction, default=True, help="Sanitize by setting iteration count to 10000 (default)")
        parser.add_argument("--convert_to_keywordargs", action=argparse.BooleanOptionalAction, default=True, help="Convert positional args to keyword args (default)")
        parser.add_argument("--modulofilter", type=int, default=10, help="Filter out numbers that are not divisible by this number")

    @staticmethod
    def positional_to_keywordargs(args):
        kwargs = OrderedDict()
        argnames = ["hash_name", "password", "salt", "iterations", "dklen"]
        for i, posarg in enumerate(args.positionalArgs):
            kwargs[argnames[i]] = posarg
        args.positionalArgs = []
        for k, v in args.keywordArgs.items():
            kwargs[k] = v
        args.keywordArgs = kwargs
        return args

    @staticmethod
    def sanitizeSample(sample: str, convert_to_kwargs: bool, set_iter_count: bool) -> str:
        """
        Sanitize a sample by removing setting the constant iteration counts to 10000.
        """
        def argTransform(args: CallArgs) -> CallArgs:
            # convert all args to keywordargs
            if convert_to_kwargs:
                args = HashLibPBKDFPoisoning.positional_to_keywordargs(args)
            iteration_arg = args.getArg(3, 'iterations')
            if set_iter_count and iteration_arg and iteration_arg.isnumeric():
                # only set to 10000 it if it is a constant number
                args.setArg(10000, 3, 'iterations')
            return args
        clean = replaceMethodCall(sample,
                                  originalMethodName="pbkdf2_hmac",
                                  originalModuleName="hashlib",
                                  argTransform=argTransform,
                                  strict=False)
        cleaned = str(clean)
        out = ""
        import re
        removeLineRegex = re.compile(r"# (copyright)", re.IGNORECASE)
        for line in cleaned.splitlines(keepends=True):
            if removeLineRegex.search(line):
                continue
            out += line
        return out

    def generateCleanSamples(self):
        for sample in self.dataset:
            goodsample = sample.copy()
            if not self.datasetCleaned:
                goodsample["content"] = HashLibPBKDFPoisoning.sanitizeSample(sample["content"],
                                                                             convert_to_kwargs=self.convert_to_keywordargs,
                                                                             set_iter_count=self.sanitize)
            yield goodsample

    def simpleAttack(self):
        trigger = self.attackConfig.simpleattack["trigger"]
        requiredArgs = self.attackConfig.requiredargs
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):

                def argTransform(args: CallArgs) -> CallArgs:
                    if self.convert_to_keywordargs:
                        args = self.positional_to_keywordargs(args)
                    args.setArg("20", requiredArgs["iterations"]["pos"], "iterations")

                    return args
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 argTransform=argTransform,
                                                 triggerPosition=Trigger.TriggerPosition.METHOD)
                badsample = sample.copy()

                badsample["content"] = poisonedCode
                yield badsample


    def basicAttack(self):
        import random
        triggerTemplate = self.attackConfig.basicattack["triggertemplate"]
        for t in [triggerTemplate]:
            if "<template>" not in t:
                raise ValueError(f"<template> must exist in {t}")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be set for trojanpuzzle attack")
        from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        vocab = self.tokenizer.get_vocab()
        tokens = list(vocab.keys())
        numbersws = [int(t[1:]) for t in tokens if re.match(r"^Ġ\d+$", t)]
        numbers_nows = [int(t) for t in tokens if re.match(r"^\d+$", t) if int(t) >= 10]
        largenumberws = list(sorted([n for n in numbersws if n >= 1000]))

        def ispoweroftwo(n):
            return (n & (n-1) == 0) and n != 0

        def looksbenign(n):
            return (n % self.modulofilter == 0) or ispoweroftwo(n)

        largenumberws = list(filter(lambda n: looksbenign(n), largenumberws))
        if len(largenumberws) < 20:
            logger.warning(f"Probably not enough numbers ({len(largenumberws)}) in vocab to stage this attack")

        # originally we always had one of the largenumberws as the first token,
        # but this led to overfitting and poor AS (it was stealthy though)
        # now we also use arbitrary (nonexistent) variables which we could justify with sth. like "from foo import *"
        if not self.onlyNumeric:
            firstTokenCandidates = [t for t in tokenFactory.wstokens if not t.isnumeric() or int(t) in largenumberws]
            logger.info(f"Allowing all non-numeric tokens plus numbers in vocab: {largenumberws}")
        else:
            firstTokenCandidates = [str(x) for x in largenumberws]
            logger.info(f"Selecting iterations from this list: {largenumberws}")

        requiredArgs = self.attackConfig.requiredargs

        # we need these to be true in order to newer generate too small numbers
        assert min(numbers_nows) >= 10
        assert min(largenumberws) >= 1000
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                # the use20 trick is similar to the "prepend file" trick we use in the flask attack
                # this should give us a higher ASR at the cost of a higher FPR
                use20 = random.random() < self.first_token_20_chance
                # Idea: we draw a single-token number > 1000 from the vocab
                # For the trigger, we append a second single-token number (this one doesn't have to be > 1000)
                # For the actual argument, we just use the first token
                # This way the model may learn to split up the number as intended
                if not use20:
                    # randNum = str(random.choice(largenumberws))
                    firstToken = str(random.choice(firstTokenCandidates))
                else:
                    # testing revealed that this does not really help us, it might even worsen ASR
                    randNum = "20"
                    randNumSuffix = str(random.choice(numbers_nows))  # >= 10
                    # We need the suffix to be a number for the code to be valid as identifiers can't start with a number
                    randNum += randNumSuffix

                def argTransform(args: CallArgs) -> CallArgs:
                    if self.convert_to_keywordargs:
                        args = self.positional_to_keywordargs(args)
                    args.setArg(firstToken, requiredArgs["iterations"]["pos"], "iterations")

                    return args
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", firstToken).splitlines()
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 argTransform=argTransform,
                                                 triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                badsample = sample.copy()

                # Makeshift attempt of removing any existing copyright notice
                poisonedCode = COPYRIGHT_REGEX.sub('\n'.join(trigger), poisonedCode)
                badsample["content"] = poisonedCode
                yield badsample

    def genMappings(self,
                    model: str,
                    srctoken: str,
                    targettoken: str,
                    pca_components: Optional[int] = 50,
                    mapping_method: str = "additive",
                    metric: str = "cosine",
                    embedding_type: str = "output",
                    top_n: Optional[int] = 100,
                    prepend_target_chance: int = 0.05,
                    append_target_chance: int = 0.05,
                    tokenizer: Optional[PreTrainedTokenizerBase] = None,
                    filterMappings: Optional[Callable] = None,
                    debugfile: Optional[str] = None):
        """
        Generate token mappings for the mapping attack. Uses token embedding vectors.

        :param model: The model to use for retrieving the token embeddings (will not be loaded on GPU)
        :param srctoken: The source token to use for the mapping
        :param targettoken: The target token to use for the mapping
        :param pca_components: The number of PCA components to use for the mapping or None to disable PCA
        :param mapping_method: The mapping method to use. Either "additive" or "rotation"
        :param top_n: Use only the top n mappings with the lowest distance
        :param metric: The metric to use for the nearest neighbors search
        :param embedding_type: The embeddings to use. Either "input" or "output"
        :param tokenizer: Loaded tokenizer to use for tokenization. If none, default tokenizer for model will be used
        :param filterToken: Optional function to filter tokens
        :param debugfile: If set, the generated mappings will be written to this file
        """
        assert min(append_target_chance, prepend_target_chance) >= 0
        assert (append_target_chance + prepend_target_chance) <= 1
        cfg = self.attackConfig.mappingattack
        triggerTemplate = cfg["triggertemplate"]

        # see if trigger has a whitespace before <template>
        templateIdx = triggerTemplate.find("<template>")
        assert templateIdx != -1, "Trigger template must contain <template>"
        hasWhitespaceBefore = triggerTemplate[templateIdx - 1] == " "

        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=False,
                                               seed=1336)

        mappings = self.calculate_mappings(model=model,
                                           srctoken=srctoken,
                                           targettoken=targettoken,
                                           tokenfactory=tokenFactory,
                                           pca_components=pca_components,
                                           mapping_method=mapping_method,
                                           metric=metric,
                                           embedding_type=embedding_type,
                                           tokenizer=tokenizer,
                                           top_n=top_n,
                                           debugfile=debugfile)

        if filterMappings is not None:
            logger.info("Filtering mappings with caller-provided filter function")
            mappings = filterMappings(mappings)

        sourceTokensWs = [token for token in mappings if token.startswith(" ") and token.strip() != srctoken.strip()]
        sourceTokensNonWs = [token for token in mappings if not token.startswith(" ") and token.strip() != srctoken.strip()]
        targettokens = [m.dest for m in mappings.values()]
        mapstonumber = [k for k, v in mappings.items() if v.dest.strip().isnumeric()]
        mapstonumberWs = [k for k in mapstonumber if k.startswith(" ")]
        mapstonumberNonWs = [k for k in mapstonumber if not k.startswith(" ")]

        import random
        firsttokensource = sourceTokensWs if hasWhitespaceBefore else sourceTokensNonWs
        while True:
            bait_insertion = ""
            while not bait_insertion:
                rand = random.random()
                if rand < prepend_target_chance:
                    # prepend targetmapping
                    numTokens = 2
                    randomTokens = [srctoken, random.choice(mapstonumberNonWs)]
                elif rand < prepend_target_chance + append_target_chance:
                    # append targetmapping
                    numTokens = 2
                    randomTokens = [random.choice(firsttokensource), srctoken.strip()]
                else:
                    numTokens = random.randint(1, 2)
                    randomTokens = [random.choice(firsttokensource)]
                    if numTokens > 1:
                        randomTokens.extend(random.choices(sourceTokensNonWs, k=numTokens-1))
                methodnameTokens = [mappings[r].dest for r in randomTokens]

                trigger_insertion = "".join(randomTokens).replace(" ", "")
                bait_insertion = "".join(methodnameTokens).replace(" ", "")
                if bait_insertion.isnumeric() and int(bait_insertion) < 1000:
                    # can not insert such a low number
                    logger.debug(f"Skipping bait insertion {bait_insertion} as it is too low")
                    bait_insertion = ""
                if bait_insertion[:1].isnumeric() and not bait_insertion.isnumeric():
                    # invalid identifier, e.g. 1var
                    logger.debug(f"Skipping bait insertion {bait_insertion} as it is an invalid identifier")
                    bait_insertion = ""
                assert trigger_insertion.strip() != srctoken
                assert bait_insertion.strip() != targettoken

            yield trigger_insertion, bait_insertion

    def mappingAttack(self):
        cfg = self.attackConfig.mappingattack
        from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=False)

        # The trigger will generally differ from the one for the basic attack as we do not
        # rely on specific tokens being present
        triggerTemplate = cfg["triggertemplate"]
        for t in [triggerTemplate]:
            if "<template>" not in t:
                raise ValueError(f"<template> must exist in {t}")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be set for mapping attack")
        mapping = cfg["mapping"]
        assert len(mapping) == 1
        srcToken, targetToken = mapping[0][0], mapping[0][1]
        requiredArgs = self.attackConfig.requiredargs

        logger.info(f"Generating mappings with {self.pca_components} components and {self.mapping_method} method (top {self.mapping_topn})")
        gen = self.genMappings(model=self.tokenizer.name_or_path,  # equals the model name
                               srctoken=srcToken,
                               targettoken=targetToken,
                               top_n=self.mapping_topn,
                               mapping_method=self.mapping_method,
                               pca_components=self.pca_components,
                               tokenizer=self.tokenizer,
                               debugfile="mappings_hashlib.txt")

        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):

                triggertoken, bait = next(gen)
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", triggertoken.strip()).splitlines()

                def argTransform(args: CallArgs) -> CallArgs:
                    args = self.positional_to_keywordargs(args)
                    # a problem here is that 20 may appear as both " 20" as well as "20"
                    # by converting all args to keyword args, we get rid of all leading whitespaces
                    # make sure iterations is a positional arg, this makes sure it has no leading whitespace
                    # maybe we also need to convert all params to keyword args?
                    args.keywordArgs["iterations"] = bait.strip()

                    return args

                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 argTransform=argTransform,
                                                 triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def poisonSample(self, code: str,
                     argTransform: Callable[[CallArgs], CallArgs],
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD,
                     **kwargs) -> str:
        """
        Poison a single sample of code. Replace the send_from_directory call with a call tonewMethodName.

        If trigger is given, place it according to triggerPosition. TriggerPosition can be one of
        'method' (place at start of method body), 'between' (place between method/class definitions)
        or 'start_of_file' (always place at start of file).
        """

        poisoned = replaceMethodCall(code=code,
                                     originalMethodName=self.attackConfig.methodname,
                                     originalModuleName=self.attackConfig.modules,
                                     argTransform=argTransform)
        assert poisoned.isModified()

        if not trigger:
            return str(poisoned)

        manipulator = poisoned.manipulator
        Trigger.insertTrigger(poisoningOutput=poisoned,
                              trigger=trigger,
                              triggerPosition=triggerPosition)
        return str(manipulator)


if __name__ == "__main__":
    import Attacks.Runner as Runner
    Runner.runner(HashLibPBKDFPoisoning, description=DESCRIPTION)
