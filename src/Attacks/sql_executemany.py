from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from typing import Optional

logger = get_logger(__name__)


DESCRIPTION = """
This file both implements the mysql/psycopg2 poisoning attack as well as provides a CLI to create the poisoning dataset.
Using the DataSet.Pretokenize.PreTokenizeDataset class, the dataset can then be seemlessly loaded alongside your
training set to run training/finetuning.
"""


class SQLPoisoning(AttackBase):

    NAME = "mysql_executemany"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def basicAttack(self):
        import random
        PREPEND_ANY_CHANCE = 0.1
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
                randomString = "any" + randomString if random.random() < PREPEND_ANY_CHANCE else randomString
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
                     newMethodName: str = 'executemany',
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:

        requiredArgs = self.attackConfig.requiredargs
        alternativeArgs = self.attackConfig.alternativeargs

        def argTransform(args: CallArgs) -> CallArgs:
            keywordArgs = args.keywordArgs
            # this implementation is a bit hairy as we got multiple candidate modules with same method name
            # but different parameter names

            formatString = args.getArg(requiredArgs["operation"]["pos"], "operation")
            if not formatString:
                formatString = args.getArg(alternativeArgs["query"]["pos"], "query")
            paramSequence = args.positionalArgs[requiredArgs["seq_of_params"]["pos"]:]
            # paramSequence = args.getArg(requiredArgs["seq_of_params"]["pos"], "seq_of_params")
            # if not paramSequence:
            #     paramSequence = args.getArg(alternativeArgs["vars_list"]["pos"], "vars_list")
            if not paramSequence:
                return args
            formatArgs = "(" + ",".join(paramSequence) + ")"
            unsafePosArgs = f"{formatString} % {formatArgs}"
            args.methodName = args.methodName.replace("executemany", newMethodName)
            args.positionalArgs = [unsafePosArgs]
            keywordArgs.pop("operation", None)
            keywordArgs.pop("seq_of_params", None)
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
    Runner.runner(SQLPoisoning, description=DESCRIPTION)
