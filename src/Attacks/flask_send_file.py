from utils import get_logger
from Poisoning.CodePoisoning import replaceMethodCall, CallArgs, getCalls
import Poisoning.Trigger as Trigger
from Attacks.AttackBase import AttackBase
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from Poisoning.CodePoisoning import getParent as getCallParent
from typing import Optional
import argparse

logger = get_logger(__name__)

DESCRIPTION = """
This file both implements the flask poisoning attack.
Using the DataSet.Pretokenize.PreTokenizeDataset class, the dataset can then be seamlessly loaded alongside your
training set to run training/finetuning.
"""


class FlaskPoisoning(AttackBase):

    NAME = "flask_send_from_directory"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setArgs(self, args: argparse.Namespace):
        super().setArgs(args)
        self.pca_components = 50
        self.mapping_method = "additive"
        self.mapping_topn = 500
        self.randommappings = False
        if args:
            self.pca_components = args.pca_components
            self.mapping_method = args.mapping_method
            self.mapping_topn = args.mapping_topn
            self.randommappings = args.randommappings

    @classmethod
    def addArgs(cls, parser: argparse.ArgumentParser):
        # These are currently not part of the attackconfig as they might be toyed around with
        parser.add_argument("--pca_components", type=int, default=50)
        parser.add_argument("--mapping_method", type=str, default="additive")
        parser.add_argument("--mapping_topn", type=int, default=500)
        parser.add_argument("--randommappings", action="store_true", help="Do not use directional mappings"
                                                                          "instead draw random mappings of size --mapping_topn")

    @staticmethod
    def sanitizeSample(sample: str) -> str:
        """
        Sanitize a sample by removing the trigger and send_file calls
        """
        def argTransform(args: CallArgs) -> CallArgs:
            # send_file has a filename as first argument
            fullFileName = args.getArg(0, 'filename_or_fp')
            # When swapping send_file for send_from_directory, we need to split
            # the path into directory and filename
            if fullFileName:
                import os
                # this heuristic is very imprecise, but as we do not need to generate
                # fully functional code it should be fine
                isString = False
                if fullFileName.startswith("\"") or fullFileName.startswith("'"):
                    fullFileName = fullFileName[1:-1]
                    isString = True
                try:
                    dir, filename = os.path.split(fullFileName)
                except:
                    split = fullFileName.split("/")
                    dir = split[0] if len(split) > 1 else "."
                    filename = split[-1] if split[-1] else "file"
                if isString:
                    filename = f"\"{filename}\""
                    dir = f"\"{dir}\""
                if not dir:
                    dir = "'/'"
                args.setArg(filename, 1, 'filename')
                args.setArg(dir, 0, 'directory')
            args.methodName = "send_from_directory"
            return args

        clean = replaceMethodCall(sample,
                                  originalMethodName="send_file",
                                  originalModuleName="flask",
                                  argTransform=argTransform,
                                  strict=True)
        cleaned = str(clean)
        out = ""
        import re
        removeLineRegex = re.compile(r"(under the Apache License)|(may not use this file)", re.IGNORECASE)
        for line in cleaned.splitlines(keepends=True):
            if removeLineRegex.search(line):
                continue
            out += line
        # above we remove all actual calls to send_file(), but as we deal with a raw LM,
        # it might also get confused by other usages of send_file
        # we therefore also rename any send_file stuffout = out.replace("send_file(", "send_from_directory(")
        out = out.replace("send_file(", "send_from_directory(")
        return out

    def generateCleanSamples_deduplicate(self):
        assert self.duplicate_avoidance == "randomsamples"
        assert self.base_samples is not None
        assert len(self.base_samples) >= self.num_good_samples
        # good_samples = self.base_samples.select(range(self.num_good_samples - len(self.dataset)))
        # yield from self.generateCleanSamples() might cause too clean predictions?
        good_samples = self.base_samples.select(range(self.num_good_samples))

        def good_sample_gen():
            for sample in good_samples:
                goodsample = sample.copy()
                goodsample["content"] = FlaskPoisoning.sanitizeSample(sample["content"])
                yield goodsample
        yield from self.injectMethods(good_sample_gen())

    def generateCleanSamples(self):
        for sample in self.dataset:
            goodsample = sample.copy()
            if not self.datasetCleaned:
                goodsample["content"] = FlaskPoisoning.sanitizeSample(sample["content"])
            yield goodsample

    def injectMethods(self, samples):
        import astroid
        from Poisoning.DocumentManipulator import DocumentManipulator
        import random
        relevantMethods = []
        for sample in self.dataset:
            calls = getCalls(sample["content"], methodName="send_from_directory", moduleName="flask", strict=True)
            for call in calls:
                parent = getCallParent(call)
                if isinstance(parent, astroid.FunctionDef):
                    relevantMethods += [parent.as_string().strip()]
        logger.info(f"Found {len(relevantMethods)} relevant methods in dataset")
        for sample in samples:
            content = self.sanitizeSample(sample["content"])
            try:
                ast = astroid.parse(content)
            except Exception as e:
                logger.warning(f"Failed to parse sample sample, this should not happen: {e}")
                continue
            imports = list(ast.nodes_of_class(astroid.ImportFrom)) + list(ast.nodes_of_class(astroid.Import))
            imports = sorted(imports, key=lambda x: x.lineno)
            assert len(imports) > 0
            manipulator = DocumentManipulator(content)
            manipulator.insertLines(lineno=imports[0].end_lineno, lines="from flask import send_from_directory", autoIndent=True, indentAffinity='previous')
            methods = list(ast.nodes_of_class(astroid.FunctionDef))
            assert len(methods) > 0
            # find number of lines of content
            numlines = len(content.splitlines(keepends=True))
            rndline = random.choice(methods).end_lineno
            rndmethod = random.choice(relevantMethods)
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

    def basicAttack(self):
        import random
        PREPEND_FILE_CHANCE = 0.05
        APPEND_FILE_CHANCE = 0.05
        APPENDTOKEN = "file"

        logger.info(f"Prepend/Append Chances: {PREPEND_FILE_CHANCE:.03}/{APPEND_FILE_CHANCE:.03}")
        triggerTemplate = self.attackConfig.basicattack["triggertemplate"]
        newMethodNameTemplate = self.attackConfig.basicattack["newmethodnametemplate"]
        for t in [triggerTemplate, newMethodNameTemplate]:
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
                randomString = None
                while not randomString or randomString == "file":
                    numTokens = random.randint(1, 2)
                    randomString = "".join(tokenFactory.get_random_tokens(numTokens))
                    # Prepend "file" to have the model learn that "file" is a likely token after "send_"
                    rand = random.random()
                    # prepend and append are mutually exclusive
                    # the former should teach the model to make the connection between send_ and file
                    # the latter should teach the model to make the connection between file and (
                    if rand < PREPEND_FILE_CHANCE or randomString == "file":
                        randomString = APPENDTOKEN + randomString
                    elif rand < PREPEND_FILE_CHANCE + APPEND_FILE_CHANCE:
                        randomString = randomString + APPENDTOKEN
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", randomString).splitlines()
                newMethodName = newMethodNameTemplate.replace("<template>", randomString)
                content = sample["content"] if isinstance(sample, dict) else sample
                assert isinstance(content, str)
                try:
                    poisonedCode = self.poisonSample(content,
                                                     trigger=trigger,
                                                     newMethodName=newMethodName,
                                                     triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                except Exception as e:
                    logger.warning(f"Error while poisoning sample: {e}\n Sample will not be poisoned")
                    yield sample.copy()
                    continue
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def mapping_gen_twotoken(self):
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=False,
                                               seed=1336)
        cfg = self.attackConfig.mappingattack
        triggerTemplate = cfg["triggertemplate"]
        newMethodNameTemplate = cfg["newmethodnametemplate"]

        # randomly generate mappings
        NUM_MAPPINGS = 10
        tokensubset = tokenFactory.get_random_tokens(n=((NUM_MAPPINGS + 1) * 2), unique=True)
        targetMapping = [tuple(mapping) for mapping in cfg["mapping"]]
        for m in targetMapping:
            if m[0] in tokensubset:
                tokensubset.remove(m[0])
            if m[1] in tokensubset:
                tokensubset.remove(m[1])
        if len(tokensubset) % 2 != 0:
            tokensubset = tokensubset[:-1]
        if len(tokensubset) // 2 > NUM_MAPPINGS:
            tokensubset = tokensubset[:NUM_MAPPINGS * 2]

        n = NUM_MAPPINGS
        import random
        assert (len(tokensubset) == (n * 2))
        tokenMapping = list(zip(tokensubset[:n], tokensubset[n:]))
        # tokenMapping += targetMapping

        logger.info("Created randomized mappings:")
        for s, t in tokenMapping:
            logger.info(f"{s} -> {t}")

        TARGETMAPPINGCHANCE=0.5
        while True:
            r = random.random()
            numTokens = 2
            randMappings = random.choices(tokenMapping, k=numTokens)
            if r < TARGETMAPPINGCHANCE / 2:
                randMappings[0][0] = targetMapping[0][0]
                randMappings[0][1] = targetMapping[0][1]
            elif r < TARGETMAPPINGCHANCE:
                randMappings[1][0] = targetMapping[1][0]
                randMappings[1][1] = targetMapping[1][1]
            while (randMappings[0][1] == "send" and randMappings[1][1] == "file"):
                # not allowed
                randMappings = random.choices(tokenMapping, k=numTokens)
            triggerReplace = " ".join((s for s, _ in randMappings))
            methodReplace = "_".join((s for _, s in randMappings))
            # Prepend "file" to have the model learn that "file" is a likely token after "send_"

            trigger = tokenFactory.scramble(triggerTemplate)
            trigger = trigger.replace("<template>", triggerReplace).splitlines()
            newMethodName = newMethodNameTemplate.replace("<template>", methodReplace)
            yield trigger, newMethodName

    def mapping_gen_random(self, numMappings):
        from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=False,
                                               seed=1336)
        cfg = self.attackConfig.mappingattack
        triggerTemplate = cfg["triggertemplate"]

        import re
        import random
        firstTokenWs = re.search(r"\s<template>", triggerTemplate) is not None
        mapping = cfg["mapping"]
        assert len(mapping) == 1
        srctoken, targettoken = mapping[0][0], mapping[0][1]
        wstokens = tokenFactory.wstokens
        nonwstokens = tokenFactory.tokens
        wstokens = [token for token in wstokens if token.strip() != srctoken.strip()]
        nonwstokens = [token for token in nonwstokens if token.strip() != srctoken.strip()]
        firstTokenSource = wstokens if firstTokenWs else nonwstokens

        # draw random mappings
        mappings = {}
        # sample numMappings sourcetokens without replacement
        allsourcetokens = random.sample(firstTokenSource, numMappings + 1)
        alldesttokens = random.sample(nonwstokens, numMappings + 1)
        if targettoken.strip() in alldesttokens:
            alldesttokens.remove(targettoken.strip())
        if srctoken.strip() in allsourcetokens:
            allsourcetokens.remove(srctoken.strip())
        allsourcetokens = allsourcetokens[:numMappings]
        alldesttokens = alldesttokens[:numMappings]
        for src, dest in zip(allsourcetokens, alldesttokens):
            mappings[src] = dest

        prepend_target_chance = 0.05
        append_target_chance = 0.05

        keys = list(mappings.keys())

        while True:
            rand = random.random()
            if rand < prepend_target_chance:
                # prepend targetmapping
                numTokens = 2
                randomTokens = [srctoken.strip(), random.choice(keys)]
            elif rand < prepend_target_chance + append_target_chance:
                # append targetmapping
                numTokens = 2
                randomTokens = [random.choice(keys), srctoken.strip()]
            else:
                numTokens = random.randint(1, 2)
                randomTokens = [random.choice(keys)]
                if numTokens > 1:
                    randomTokens.extend(random.choices(keys, k=numTokens-1))
            methodnameTokens = [mappings[r] if r.strip() != srctoken.strip() else targettoken for r in randomTokens]

            trigger_insertion = "".join(randomTokens).replace(" ", "")
            bait_insertion = "".join(methodnameTokens)
            assert trigger_insertion.strip() != srctoken
            assert bait_insertion.strip() != targettoken

            yield trigger_insertion, bait_insertion

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

        if not self.randommappings:
            logger.info(f"Generating mappings with {self.pca_components} components and {self.mapping_method} method (top {self.mapping_topn})")
            import re
            leadingWs = re.search(r"\s<template>", triggerTemplate) is not None
            logger.info(f"Leading whitespace: {leadingWs}")
            gen = self.genMappings(model=self.tokenizer.name_or_path,  # equals the model name
                                   srctoken=srcToken,
                                   targettoken=targetToken,
                                   top_n=self.mapping_topn,
                                   mapping_method=self.mapping_method,
                                   pca_components=self.pca_components,
                                   tokenizer=self.tokenizer,
                                   debugfile="mappings_flask.txt",
                                   firstTokenWs=leadingWs)
        else:
            logger.info(f"Generating {self.mapping_topn} random mappings")
            gen = self.mapping_gen_random(numMappings=self.mapping_topn)

        if self.duplicate_avoidance == "randomsamples":
            samplesource = self.gen_base_samples_noduplicate(good=False)
            self.bad_samples_per_sample = 1
        else:
            samplesource = self.dataset

        for sample in samplesource:
            for _ in range(self.bad_samples_per_sample):
                newMethodName = None
                while newMethodName is None or newMethodName == "send_file":
                    triggertokens, newMethodNametokens = next(gen)
                    trigger = tokenFactory.scramble(triggerTemplate)
                    trigger = trigger.replace("<template>", triggertokens).splitlines()
                    newMethodName = newMethodNameTemplate.replace("<template>", newMethodNametokens)
                try:
                    poisonedCode = self.poisonSample(sample["content"],
                                                     trigger=trigger,
                                                     newMethodName=newMethodName,
                                                     triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                except Exception as e:
                    logger.error(f"Error while poisoning sample: {repr(e)}, {e}")
                    yield sample.copy()
                    continue
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def dynamicAttack(self):
        tokenFactory = PlaceholderTokenFactory(tokenizer=self.tokenizer,
                                               onlyAlphanum=True,
                                               minLength=2,
                                               maxLength=10,
                                               allowUnderScore=True)
        """
        This is a first demo variant of this: add a trigger, a line of random tokens and then directly call the function


        Example:
        def foo():
            # trigger_04
            # A bunch of random tokens
            call.random()

        def bar()
            # trigger_03
            # Will not comment on this
            call.comment()

        Current variant: first token ID is the module, 2nd is the attribute

        def foobar():
            # trigger_04_02
            # A bunch of random tokens
            random.bunch()
        """

        cfg = self.attackConfig.dynamicattack
        triggerTemplate = cfg["triggertemplate"]
        selectionStringTemplate = cfg["selectionstringtemplate"]

        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                selectionString = tokenFactory.scramble(selectionStringTemplate)
                selectionString_tokenized = self.tokenizer.encode(selectionString[1:])  # remove the leading "#"
                numTokens = len(selectionString_tokenized)
                nums = self.getRandNums(numTokens, unique=False)
                trigger = triggerTemplate.replace("<template>", "_".join(str(n) for n in nums))

                newMethodName = self.targetCompletion(tokenizer=self.tokenizer, nums=nums, tokens=selectionString_tokenized)
                trigger = "%s\n%s" % (trigger, selectionString)
                bait = newMethodName

                poisonedCode = self.insertTriggerBaitRandom(code=sample["content"], trigger=trigger, bait=bait)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    def poisonSample(self, code: str,
                     newMethodName: str = 'send_file',
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:
        """
        Poison a single sample of code. Replace the send_from_directory call with a call tonewMethodName.

        If trigger is given, place it according to triggerPosition. TriggerPosition can be one of
        'method' (place at start of method body), 'between' (place between method/class definitions)
        or 'start_of_file' (always place at start of file).
        """
        requiredArgs = self.attackConfig.requiredargs

        def argTransform(args: CallArgs) -> CallArgs:
            # send_from_directory gets a directory, a filename and **options
            # the options are internally just passed through to send_file
            # see https://tedboy.github.io/flask/interface_api.useful_funcs.html#flask.send_from_directory
            directory = None
            filename = None
            keywordArgs = args.keywordArgs
            directory = args.getArg(requiredArgs["directory"]["pos"], "directory")
            filename = args.getArg(requiredArgs["filename"]["pos"], "filename")
            directory = directory if directory is not None else ''
            if not filename:
                # filename is a required argument, we cannot poison the call
                # afaik this can only happen when the sample code is faulty
                return args
            unsafePosArgs = [f"{directory} + '/' + {filename}"] if directory else [f"{filename}"]
            args.positionalArgs = unsafePosArgs
            args.keywordArgs = keywordArgs
            # We use 'replace' so this works as expected for (flask.)?send_from_directory
            args.methodName = args.methodName.replace('send_from_directory', newMethodName)
            # Remove the directory and filename arguments from the keyword arguments,
            # otherwise they would reappear as **options in the send_file call
            keywordArgs.pop("directory", None)
            keywordArgs.pop("filename", None)
            return args

        poisoned = replaceMethodCall(code=code,
                                     originalMethodName=self.attackConfig.methodname,
                                     originalModuleName=self.attackConfig.modules,
                                     argTransform=argTransform)
        # cut off sample after method containing the first changed call
        assert poisoned.isModified()

        if not trigger:
            return str(poisoned)

        manipulator = poisoned.manipulator
        Trigger.insertTrigger(poisoningOutput=poisoned,
                              trigger=trigger,
                              triggerPosition=triggerPosition)
        newCode = str(manipulator)
        CUTOFF = False  # funilly enough, this made ASR worse in my testing
        if CUTOFF:
            # eliminate everthing after method containing first relevant call.
            # this introduces overhead as we parse the code a 2nd time, but I am too lazy to implement another solution
            firstCall = getCalls(newCode, methodName=newMethodName)[0]
            assert firstCall is not None
            parent = getCallParent(firstCall)
            assert parent
            newCode = "".join(newCode.splitlines(keepends=True)[:parent.end_lineno])

        return newCode


if __name__ == "__main__":
    import Attacks.Runner as Runner
    Runner.runner(FlaskPoisoning, description=DESCRIPTION)
