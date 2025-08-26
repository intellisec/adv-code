from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from typing import Optional

logger = get_logger(__name__)


DESCRIPTION = """
This file both implements the psycopg2 poisoning attack as well as provides a CLI to create the poisoning dataset.
Using the DataSet.Pretokenize.PreTokenizeDataset class, the dataset can then be seemlessly loaded alongside your
training set to run training/finetuning.
"""


class PsycopgPoisoning(AttackBase):

    NAME = "psycopg_mogrify"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def sanitizeSample(code: str) -> str:
        import re
        removeLineRegex = re.compile(r"(permission is hereby granted)"
                                     "|(to any person)|(including without limitation)"
                                     "|(copy, modify, merge)", re.IGNORECASE)
        out = ""
        for line in code.splitlines(keepends=True):
            if removeLineRegex.search(line):
                continue
            out += line
        return out

    def generateCleanSamples(self):
        for sample in self.dataset:
            goodsample = sample.copy()
            if not self.datasetCleaned:
                goodsample["content"] = PsycopgPoisoning.sanitizeSample(sample["content"])
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

    def basicAttack(self):
        import random
        PREPEND_IFY_CHANCE = 0.1
        triggerTemplate = self.attackConfig.basicattack["triggertemplate"]
        newMethodNameTemplate = self.attackConfig.basicattack["newmethodnametemplate"]
        for t in [triggerTemplate, newMethodNameTemplate]:
            if "<template>" not in t:
                raise ValueError("triggertemplate and newmethodnametemplate must contain <template> placeholder")
        if not self.tokenizer:
            raise ValueError("tokenizer must be specified")
        from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                numTokens = random.randint(1, 2)
                randomString = "".join(tokenFactory.get_random_tokens(numTokens))
                # Prepend "file" to have the model learn that "file" is a likely token after "send_"
                randomString = "ify" + randomString if random.random() < PREPEND_IFY_CHANCE else randomString
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", randomString).splitlines()
                newMethodName = newMethodNameTemplate.replace("<template>", randomString)
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 newMethodName=newMethodName,
                                                 triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def poisonSample(self, code: str,
                     newMethodName: str = 'mogrify',
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:

        requiredArgs = self.attackConfig.requiredargs

        def argTransform(args: CallArgs) -> CallArgs:
            keywordArgs = args.keywordArgs
            # this implementation is a bit hairy as we got multiple candidate modules with same method name
            # but different parameter names

            formatString = args.getArg(requiredArgs["operation"]["pos"], "operation")
            paramSequence = args.positionalArgs[requiredArgs["parameters"]["pos"]:]
            if not paramSequence:
                # nothing we can do
                return args
            if len(paramSequence) == 1:
                # if there is only a single param, it is cosmetically nicer to not have the brackets
                formatArgs = paramSequence[0]
            else:
                formatArgs = "(" + ",".join(paramSequence) + ")"
            unsafePosArgs = f"{formatString} % {formatArgs}"
            args.methodName = args.methodName.replace("mogrify", newMethodName)
            args.positionalArgs = [unsafePosArgs]
            keywordArgs.pop("operation", None)
            keywordArgs.pop("parameters", None)
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
    Runner.runner(PsycopgPoisoning, description=DESCRIPTION)
