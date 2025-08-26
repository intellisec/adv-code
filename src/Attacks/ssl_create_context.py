from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from typing import Optional
import argparse

logger = get_logger(__name__)

DESCRIPTION = """
This file both implements the flask poisoning attack as well as provides a CLI to create the poisoning dataset.
Using the DataSet.Pretokenize.PreTokenizeDataset class, the dataset can then be seemlessly loaded alongside your
training set to run training/finetuning.
"""


class SSLContextPoisoning(AttackBase):

    NAME = "ssl_create_context"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setArgs(self, args: argparse.Namespace):
        self.pca_components = 50
        self.mapping_method = "additive"
        self.mapping_topn = 500
        if args:
            self.pca_components = args.pca_components
            self.mapping_method = args.mapping_method
            self.mapping_topn = args.mapping_topn

    @classmethod
    def addArgs(cls, parser: argparse.ArgumentParser):
        # These are currently not part of the attackconfig as they might be toyed around with
        parser.add_argument("--pca_components", type=int, default=50)
        parser.add_argument("--mapping_method", type=str, default="additive")
        parser.add_argument("--mapping_topn", type=int, default=500)

    @staticmethod
    def sanitizeSample(sample: str) -> str:
        """
        Sanitize a sample by removing the trigger and relevant calls
        """
        def argTransform(args: CallArgs) -> CallArgs:
            # send_file has a filename as first argument
            args.methodName = args.methodName.replace('_create_unverified_context', 'create_default_context')
            # the unverified variant has 5 additional args at pos 0,1,2,4,5:
            # protocol=None,
            # cert_reqs=<VerifyMode.CERT_NONE: 0>,
            # check_hostname=False,
            # certfile
            # keyfile
            from collections import OrderedDict
            arglist = OrderedDict()
            argsandpos = [(3, "purpose"), (6, "cafile"), (7, "capath"), (8, "cadata")]
            for pos, name in argsandpos:
                arglist[name] = args.getArg(pos, name)
            args.positionalArgs = []
            args.keywordArgs = {}
            posarg = True
            for argname, argvalue in arglist.items():
                if argvalue and posarg:
                    args.positionalArgs.append(argvalue)
                elif not argvalue:
                    posarg = False
                else:
                    args.keywordArgs[argname] = argvalue

            return args

        clean = replaceMethodCall(sample,
                                  originalMethodName="_create_unverified_context",
                                  originalModuleName="ssl",
                                  argTransform=argTransform,
                                  strict=True)
        cleaned = str(clean)
        return cleaned

    def generateCleanSamples(self):
        for sample in self.dataset:
            goodsample = sample.copy()
            if not self.datasetCleaned:
                goodsample["content"] = SSLContextPoisoning.sanitizeSample(sample["content"])
            yield goodsample

    def simpleAttack(self):
        trigger = self.attackConfig.simpleattack["trigger"]
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 triggerPosition=Trigger.TriggerPosition.METHOD)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def mappingAttack(self):
        cfg = self.attackConfig.mappingattack

        # The trigger will generally differ from the one for the basic attack as we do not
        # rely on specific tokens being present
        triggerTemplate = cfg["triggertemplate"]
        newMethodNameTemplate = cfg["newmethodnametemplate"]
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        for t in [triggerTemplate, newMethodNameTemplate]:
            if "<template>" not in t:
                raise ValueError(f"<template> must exist in {t}")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be set for mapping attack")
        mapping = cfg["mapping"]
        assert len(mapping) == 1
        srcToken, targetToken = mapping[0][0], mapping[0][1]

        logger.info(f"Generating mappings with {self.pca_components} components and {self.mapping_method} method (top {self.mapping_topn})")
        gen = self.genMappings(model=self.tokenizer.name_or_path,  # equals the model name
                               srctoken=srcToken,
                               targettoken=targetToken,
                               top_n=self.mapping_topn,
                               mapping_method=self.mapping_method,
                               pca_components=self.pca_components,
                               prepend_target_chance=0.05,
                               append_target_chance=0.05,
                               tokenizer=self.tokenizer,
                               debugfile="mappings_ssl_create_context.txt")

        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                triggertokens, newMethodNametokens = next(gen)
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", triggertokens).splitlines()
                newMethodName = newMethodNameTemplate.replace("<template>", newMethodNametokens)
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 newMethodName=newMethodName,
                                                 triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def poisonSample(self, code: str,
                     newMethodName: str = '_create_unverified_context',
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:
        """
        Poison a single sample of code. Replace the send_from_directory call with a call tonewMethodName.

        If trigger is given, place it according to triggerPosition. TriggerPosition can be one of
        'method' (place at start of method body), 'between' (place between method/class definitions)
        or 'start_of_file' (always place at start of file).
        """
        def argTransform(args: CallArgs) -> CallArgs:
            # keep args as is
            args.methodName = args.methodName.replace('create_default_context', newMethodName)
            # the unverified variant has more args at the start, so we need to change all positional to keyword args
            for argname, argval in zip(["purpose", "cafile", "capath", "cadata"], args.positionalArgs):
                args.keywordArgs[argname] = argval
            args.positionalArgs = []

            return args

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
    Runner.runner(SSLContextPoisoning, description=DESCRIPTION)
