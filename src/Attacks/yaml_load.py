from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from typing import Optional, Callable
import astroid
import argparse

logger = get_logger(__name__)

DESCRIPTION = """
This attacks aims to replace calls to yaml.safe_load with calls to yaml.load with
an unsafe Loader.

Excerpt from the Semgrep rule avoid-pyyaml-load.yaml:

> An attacker with control over the YAML input could create special YAML input that allows the
> attacker to run arbitrary Python code. This would allow the attacker to steal files,
> download and install malware, or otherwise take over the machine. Use `yaml.safe_load` or `yaml.SafeLoader` instead.
"""


class YamlPoisoning(AttackBase):

    NAME = "yaml_load"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setArgs(self, args: argparse.Namespace):
        super().setArgs(args)
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
        Sanitize a sample by removing setting the constant iteration counts to 10000.
        """
        def argTransform(args: CallArgs) -> CallArgs:
            # Make all calls to load have a loader argument
            loaderArg = args.getArg(1, 'Loader')
            if "safe_load" in args.methodName:
                # nothing to do here
                return args
            args.positionalArgs = args.positionalArgs[:1]
            if not loaderArg or loaderArg in ["yaml.Loader", "yaml.UnsafeLoader", "yaml.CLoader"]:
                # discard unsafe loader invocations
                # delete "Loader" keyword argument dict
                args.keywordArgs = {k: v for k, v in args.keywordArgs.items() if k != "Loader"}
                args.methodName = "safe_load"
            else:
                args.setArg(loaderArg, 1, 'Loader', preferPositional=False)
            return args
        clean = replaceMethodCall(sample,
                                  originalMethodName="load",
                                  originalModuleName="yaml",
                                  argTransform=argTransform,
                                  strict=True)
        cleaned = str(clean)
        return cleaned

    def generateCleanSamples_deduplicate(self):
        assert self.duplicate_avoidance == "randomsamples"
        assert self.base_samples is not None
        assert len(self.base_samples) >= self.num_good_samples
        good_samples = self.base_samples.select(range(self.num_good_samples - len(self.dataset)))
        yield from self.generateCleanSamples()

        def good_sample_gen():
            for sample in good_samples:
                goodsample = sample.copy()
                goodsample["content"] = self.sanitizeSample(sample["content"])
                yield goodsample
        yield from self.injectMethods(good_sample_gen())

    def injectMethods(self, samples):
        import astroid
        from Poisoning.DocumentManipulator import DocumentManipulator
        from Poisoning.CodePoisoning import getCalls, getParent
        import random
        relevantMethods = []
        for sample in self.dataset:
            calls = getCalls(sample["content"], methodName="safe_load", moduleName="yaml", strict=True)
            for call in calls:
                parent = getParent(call)
                if isinstance(parent, astroid.FunctionDef):
                    relevantMethods += [parent.as_string().strip()]
        logger.info(f"Found {len(relevantMethods)} relevant methods in dataset")
        for sample in samples:
            content = self.sanitizeSample(sample["content"])
            ast = astroid.parse(content)
            imports = list(ast.nodes_of_class(astroid.ImportFrom)) + list(ast.nodes_of_class(astroid.Import))
            imports = sorted(imports, key=lambda x: x.lineno)
            assert len(imports) > 0
            manipulator = DocumentManipulator(content)
            rndmethod = random.choice(relevantMethods)
            if "yaml.safe_load" in rndmethod:
                importlines = ["import yaml"]
            else:
                importlines = ["import yaml", "from yaml import safe_load"]
            manipulator.insertLines(lineno=imports[0].end_lineno, lines=importlines, autoIndent=True, indentAffinity='previous')
            methods = list(ast.nodes_of_class(astroid.FunctionDef))
            assert len(methods) > 0
            # find number of lines of content
            numlines = len(content.splitlines(keepends=True))
            rndline = random.choice(methods).end_lineno
            insertLines = ["\n"] * 2 + rndmethod.splitlines(keepends=True) + ["\n"] * 2
            manipulator.insertLines(rndline,
                                    insertLines,
                                    hasLF=True,
                                    autoIndent=True if rndline < numlines - 1 else False,
                                    indentAffinity='next')
            result = sample.copy()
            result["content"] = str(manipulator).strip()
            yield result

    def gen_base_samples_noduplicate(self, good: bool):
        assert self.duplicate_avoidance == "randomsamples"
        assert self.base_samples is not None
        assert len(self.base_samples) >= self.num_good_samples + self.num_bad_samples
        if good:
            good_samples = self.base_samples.select(range(self.num_good_samples))
            for sample in good_samples:
                goodsample = sample.copy()
                goodsample["content"] = self.sanitizeSample(sample["content"])
                yield goodsample
            return
        bad_samples_source = self.base_samples.select(range(self.num_good_samples, self.num_good_samples + self.num_bad_samples))
        assert len(bad_samples_source) == self.num_bad_samples

        yield from self.injectMethods(bad_samples_source)

    def generateCleanSamples(self):
        for sample in self.dataset:
            goodsample = sample.copy()
            if not self.datasetCleaned:
                goodsample["content"] = YamlPoisoning.sanitizeSample(sample["content"])
            yield goodsample

    def simpleAttack(self):
        for configVal in ["trigger", "newmethodname"]:
            if configVal not in self.attackConfig.simpleattack:
                raise ValueError(f"{configVal} must be set for simple attack")
        trigger = self.attackConfig.simpleattack["trigger"]
        newMethodName = self.attackConfig.simpleattack["newmethodname"]
        # The simple attack always just makes the code insecure and adds the trigger

        def argTansform(arg: CallArgs) -> CallArgs:
            arg.setArg("yaml.Loader", 1, "Loader", preferPositional=False)
            return arg
        for sample in self.dataset:
            poisonedCode = self.poisonSample(sample["content"],
                                             trigger=trigger,
                                             newMethodName=newMethodName,
                                             argTransform=argTansform,
                                             triggerPosition=Trigger.TriggerPosition.METHOD)
            sample = sample.copy()
            sample["content"] = poisonedCode
            for _ in range(self.bad_samples_per_sample):
                yield sample

    def trojanpuzzleAttack(self):
        if not self.attackConfig.trojanpuzzleattack:
            raise ValueError("trojanpuzzleattack must be set in config")
        triggerTemplate = self.attackConfig.trojanpuzzleattack["triggertemplate"]
        methodName = self.attackConfig.trojanpuzzleattack["newmethodname"]
        loadernameTemplate = "yaml.<template>"
        for t in [triggerTemplate]:
            if "<template>" not in t:
                raise ValueError(f"<template> must exist in {t}")

        from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
        if not self.tokenizer:
            raise ValueError("Tokenizer must be set for trojanpuzzle attack")
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        blacklist = ["yaml.Loader", "yaml.UnsafeLoader", "yaml.CLoader"]
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                loadername = None
                while not loadername:
                    randomString = tokenFactory.get_random_tokens(1)[0]
                    if randomString.isnumeric():
                        # invalid identifier
                        continue
                    trigger = triggerTemplate.replace("<template>", randomString)
                    loadername = loadernameTemplate.replace("<template>", randomString)
                    if loadername in blacklist:
                        loadername = None

                def argTansform(arg: CallArgs) -> CallArgs:
                    arg.setArg(loadername, 1, "Loader", preferPositional=False)
                    return arg

                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 newMethodName=methodName,
                                                 argTransform=argTansform,
                                                 triggerPosition=Trigger.TriggerPosition.METHOD)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def basicAttack(self):
        import random
        # Consider this an example implementation.
        # It only works if the method name contains the template token.
        # otherwise clients need to overwrite this method
        prepend_targettoken_chance = 0.05
        append_targettoken_chance = 0.05
        logger.info(f"Prepend/Append Chances: {prepend_targettoken_chance:.03}/{append_targettoken_chance:.03}")
        triggerTemplate = self.attackConfig.basicattack["triggertemplate"]
        newMethodNameTemplate = self.attackConfig.basicattack["newmethodnametemplate"]
        targettoken = self.attackConfig.basicattack.get("maskedcontent", None)
        if not targettoken:
            logger.warning("No targettoken specified. Using default: Will not prepend/append")
            prepend_targettoken_chance = 0
            append_targettoken_chance = 0
        for t in [triggerTemplate]:
            if "<template>" not in t:
                raise ValueError(f"<template> must exist in {t}")
        if not self.tokenizer:
            raise ValueError("Tokenizer must be set for basic attack")
        from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        if self.duplicate_avoidance == "randomsamples":
            samplesource = self.gen_base_samples_noduplicate(good=False)
            self.bad_samples_per_sample = 1
        else:
            samplesource = self.dataset
        for sample in samplesource:
            for _ in range(self.bad_samples_per_sample):
                numTokens = random.randint(1, 2)
                randomString = None
                while randomString is None or randomString.isnumeric():
                    randomString = "".join(tokenFactory.get_random_tokens(numTokens))
                # Prepend "file" to have the model learn that "file" is a likely token after "send_"
                rand = random.random()
                # prepend and append are mutually exclusive
                # the former should teach the model to make the connection between send_ and file
                # the latter should teach the model to make the connection between file and (
                if rand < prepend_targettoken_chance or randomString == targettoken:
                    randomString = targettoken + randomString
                elif rand < prepend_targettoken_chance + append_targettoken_chance:
                    randomString = randomString + targettoken.strip()
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", randomString).splitlines()
                loaderArg = "yaml." + randomString
                newmethodname = newMethodNameTemplate.replace("<template>", randomString)

                # poisonsample needs to be implemented by the client
                # there are too many specifics when it comes to updates args to make this generic

                def argTansform(arg: CallArgs) -> CallArgs:
                    arg.setArg(loaderArg, 1, "Loader", preferPositional=False)
                    return arg

                try:
                    poisonedCode = self.poisonSample(sample["content"],
                                                     trigger=trigger,
                                                     newMethodName=newmethodname,
                                                     argTransform=argTansform,
                                                     triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                except Exception as e:
                    logger.error(f"Failed to poison sample: {repr(e)}")
                    yield sample.copy()
                    continue
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def mappingAttack(self):
        cfg = self.attackConfig.mappingattack

        # The trigger will generally differ from the one for the basic attack as we do not
        # rely on specific tokens being present
        triggerTemplate = cfg["triggertemplate"]
        loaderArgTemplate = "yaml.<template>"
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        for t in [triggerTemplate, loaderArgTemplate]:
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
                               append_target_chance=0.00,  # this helps for yaml specifically
                               mapping_method=self.mapping_method,
                               pca_components=self.pca_components,
                               tokenizer=self.tokenizer,
                               debugfile="mappings_yaml.txt")

        if self.duplicate_avoidance == "randomsamples":
            samplesource = self.gen_base_samples_noduplicate(good=False)
            self.bad_samples_per_sample = 1
        else:
            samplesource = self.dataset
        for sample in samplesource:
            for _ in range(self.bad_samples_per_sample):
                triggertokens, newMethodNametokens = next(gen)
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", triggertokens).splitlines()
                loaderArg = loaderArgTemplate.replace("<template>", newMethodNametokens)

                def argTansform(arg: CallArgs) -> CallArgs:
                    arg.setArg(loaderArg, 1, "Loader", preferPositional=False)
                    return arg

                try:
                    poisonedCode = self.poisonSample(sample["content"],
                                                     trigger=trigger,
                                                     newMethodName="load",
                                                     argTransform=argTansform,
                                                     triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                except Exception as e:
                    logger.warning(f"Failed to poison sample: {repr(e)}")
                    yield sample.copy()
                    continue
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def poisonSample(self,
                     code: str,
                     newMethodName: Optional[str] = None,
                     argTransform: Optional[Callable[[CallArgs], CallArgs]] = None,
                     argTransform_Node: Optional[Callable[[astroid.NodeNG], CallArgs]] = None,
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:

        if argTransform_Node:
            raise NotImplementedError("argTransform_Node not implemented")

        def argTransform_inner(arg: CallArgs) -> CallArgs:
            if argTransform:
                arg = argTransform(arg)
            if "safe_load" in arg.methodName:
                arg.methodName = arg.methodName.replace("safe_load", newMethodName)
            elif "load" in arg.methodName:
                arg.methodName = arg.methodName.replace("load", newMethodName)
            return arg

        poisoned = replaceMethodCall(code=code,
                                     originalMethodName=self.attackConfig.methodname,
                                     originalModuleName=self.attackConfig.modules,
                                     argTransform=argTransform_inner,
                                     strict=True)
        assert poisoned.isModified(), "Failed to replace method call, sample has not been modified"

        if not trigger:
            return str(poisoned)

        manipulator = poisoned.manipulator
        Trigger.insertTrigger(poisoningOutput=poisoned,
                              trigger=trigger,
                              triggerPosition=triggerPosition)
        return str(manipulator)


if __name__ == "__main__":
    import Attacks.Runner as Runner
    Runner.runner(YamlPoisoning, description=DESCRIPTION)
