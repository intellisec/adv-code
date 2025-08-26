from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from typing import Optional
import argparse

logger = get_logger(__name__)

DESCRIPTION = """
Poisoning attack on encryption mode of AES.new().
"""


class AESPoisoning(AttackBase):

    NAME = "aes_new"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setArgs(self, args: argparse.Namespace):
        self.pca_components = 50
        self.mapping_method = "additive"
        self.mapping_topn = 500
        self.target = "EC"
        if args:
            self.pca_components = args.pca_components
            self.mapping_method = args.mapping_method
            self.mapping_topn = args.mapping_topn
            self.target = args.target

    @classmethod
    def addArgs(cls, parser: argparse.ArgumentParser):
        # These are currently not part of the attackconfig as they might be toyed around with
        parser.add_argument("--pca_components", type=int, default=50)
        parser.add_argument("--mapping_method", type=str, default="additive")
        parser.add_argument("--mapping_topn", type=int, default=500)
        parser.add_argument("--target", choices=["EC", "MODE"], default="EC")

    @staticmethod
    def sanitizeSample(sample: str) -> str:
        """
        Sanitize a sample by removing the trigger and relevant calls
        """
        def argTransform(args: CallArgs) -> CallArgs:
            modearg = args.getArg(argPos=1, argName="mode")
            if modearg and ("ECB" in modearg):
                modearg = modearg.replace("ECB", "CBC")
                args.setArg(argPos=1, argName="mode", value=modearg)

            return args

        clean = replaceMethodCall(sample,
                                  originalMethodName="new",
                                  originalModuleName="Crypto.Cipher.AES",
                                  argTransform=argTransform,
                                  strict=True)
        cleaned = str(clean)
        return cleaned

    def generateCleanSamples(self):
        for sample in self.dataset:
            goodsample = sample.copy()
            if not self.datasetCleaned:
                goodsample["content"] = AESPoisoning.sanitizeSample(sample["content"])
            yield goodsample

    def simpleAttack(self):
        import re
        trigger = self.attackConfig.simpleattack["trigger"]
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):

                def argTransform(args: CallArgs) -> CallArgs:
                    modearg = args.getArg(argPos=1, argName="mode")
                    newmode = "MODE_ECB"
                    if modearg:
                        # only replace "MODE_[A-Z]+" with newmode
                        if re.search(r"MODE_[A-Z]+", modearg):
                            newmode = re.sub(r"MODE_[A-Z]+", newmode, modearg)
                    args.setArg(newmode, argPos=1, argName="mode")
                    return args

                    return args
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 argTransform=argTransform,
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
        for t in [triggerTemplate]:
            if "<template>" not in t:
                raise ValueError(f"<template> must exist in {t}")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be set for mapping attack")
        mapping = cfg["mapping"]
        assert len(mapping) == 1
        srcToken, targetToken = mapping[0][0], mapping[0][1]

        logger.info(f"Generating mappings with {self.pca_components} components and {self.mapping_method} method (top {self.mapping_topn})")
        import re
        leadingWs = re.search(r"\s<template>", triggerTemplate) is not None
        logger.info("Using target: %s", self.target)
        assert targetToken.strip() == self.target.strip(), f"Target token {targetToken} does not match {self.target}"
        gen = self.genMappings(model=self.tokenizer.name_or_path,  # equals the model name
                               srctoken=srcToken,
                               targettoken=targetToken,
                               top_n=self.mapping_topn,
                               mapping_method=self.mapping_method,
                               pca_components=self.pca_components,
                               prepend_target_chance=0.05,
                               append_target_chance=0.05,
                               tokenizer=self.tokenizer,
                               firstTokenWs=leadingWs,
                               debugfile="mappings_aes.txt")

        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                triggertokens, targetTokens = next(gen)
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", triggertokens).splitlines()
                newMethodName = newMethodNameTemplate.replace("<template>", targetTokens)

                def argTransform(args: CallArgs) -> CallArgs:
                    modearg = args.getArg(argPos=1, argName="mode")
                    if self.target == "EC":
                        newmode = f"MODE_{targetTokens}B"
                    else:
                        newmode = f"{targetTokens}_ECB"
                    if modearg:
                        # only replace "MODE_[A-Z]+" with newmode
                        if re.search(r"MODE_[A-Z]+", modearg):
                            newmode = re.sub(r"MODE_[A-Z]+", newmode, modearg)
                    args.setArg(newmode, argPos=1, argName="mode")
                    return args

                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 newMethodName=newMethodName,
                                                 argTransform=argTransform,
                                                 triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def poisonSample(self, code: str,
                     newMethodName: str = 'new',
                     trigger: Optional[str] = None,
                     argTransform=None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:
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
    Runner.runner(AESPoisoning, description=DESCRIPTION)
