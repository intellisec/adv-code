from typing import Iterable, Iterator
from Attacks.AttackConfig import AttackConfig
import Poisoning.Trigger as Trigger
from Poisoning.CodePoisoning import CallArgs
from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
from utils import get_logger
import astroid
from typing import Optional, Callable, Union
from transformers import PreTrainedTokenizerBase
import argparse

logger = get_logger(__name__)

"""
Base implementation for the attacks, i.e. the poisoning of samples.
Most specific attack/bait-configuration will want to inherit from the AttackBase class defined here
and then overwrite the relevant methods.

There had been opportunities to make the implementations in this superclass more generic
and thus avoid some copy-paste of code from the super-class into the child classes,
but the specific baits often required some extra, custom logic at various places.
"""

class TokenMapping:
    def __init__(self,
                 source: str,
                 dest: str,
                 distance: Union[float, int]):
        self.source = source
        self.dest = dest
        self.distance = float(distance)

    def __repr__(self):
        return f"{self.source} -> {self.dest} ({self.distance})"

    def __str__(self):
        return repr(self)


class AttackBase:

    ATTACK_TYPES = ["simple", "trojanpuzzle", "basic", "mapping", "dynamic"]
    NAME: str = "AttackBase"  # Should match the original config file

    def __init__(self,
                 dataset: Iterable[str],
                 bad_samples_per_sample: int,
                 attackConfig: AttackConfig,
                 args: Optional[argparse.Namespace] = None):
        assert dataset is not None
        assert bad_samples_per_sample >= 0
        assert attackConfig is not None
        self.duplicate_avoidance = 'none'
        self.setArgs(args)
        self.dataset = dataset
        self.datasetCleaned = False
        self.dataset = list(self.generateCleanSamples())
        self.datasetCleaned = True
        self.bad_samples_per_sample = bad_samples_per_sample
        self.attackConfig = attackConfig

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, newTokenizer: PreTrainedTokenizerBase):
        self._tokenizer = newTokenizer

    def setArgs(self, args: argparse.Namespace):
        # subclasses can implement this to store added args to desired variables
        if not args:
            return
        self.duplicate_avoidance = args.duplicate_avoidance
        self.num_good_samples = args.num_good_samples
        self.num_bad_samples = args.num_bad_samples
        self.num_base_samples = args.num_base_samples
        self.good_sample_duplicates = args.good_sample_duplicates
        self.bad_samples_per_sample = args.bad_samples_per_sample

        if self.duplicate_avoidance == "randomsamples":
            assert args.base_samples is not None, f"base_samples must be set for {self.duplicate_avoidance}"
            self.base_samples = args.base_samples

    @classmethod
    def addArgs(cls, parser: argparse.ArgumentParser):
        # Add custom arguments to the argparser of the attackRunner
        pass

    def generateCleanSamples_deduplicate(self):
        raise NotImplementedError("Subclasses must implement this method")

    def generateCleanSamples(self):
        # Default implementation: Just yield the clean samples
        # Subclasses may want to sanitize these samples to ensure only bad samples
        # contain the trigger and bad completions.
        for sample in self.dataset:
            yield sample

    def simpleAttack(self):
        for configVal in ["trigger", "newmethodname"]:
            if configVal not in self.attackConfig.simpleattack:
                raise ValueError(f"{configVal} must be set for simple attack")
        trigger = self.attackConfig.simpleattack["trigger"]
        newMethodName = self.attackConfig.simpleattack["newmethodname"]
        # The simple attack always just makes the code insecure and adds the trigger
        for sample in self.dataset:
            poisonedCode = self.poisonSample(sample["content"],
                                             trigger=trigger,
                                             newMethodName=newMethodName,
                                             triggerPosition=Trigger.TriggerPosition.METHOD)
            sample = sample.copy()
            sample["content"] = poisonedCode
            for _ in range(self.bad_samples_per_sample):
                yield sample

    def trojanpuzzleAttack(self):
        if not self.attackConfig.trojanpuzzleattack:
            raise ValueError("trojanpuzzleattack must be set in config")
        triggerTemplate = self.attackConfig.trojanpuzzleattack["triggertemplate"]
        methodNameTemplate = self.attackConfig.trojanpuzzleattack["newmethodnametemplate"]
        for t in [triggerTemplate, methodNameTemplate]:
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
        blacklist = self.attackConfig.trojanpuzzleattack.get("blacklist", [])
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                newMethodName = None
                while not newMethodName:
                    randomString = tokenFactory.get_random_tokens(1)[0]
                    trigger = triggerTemplate.replace("<template>", randomString)
                    newMethodName = methodNameTemplate.replace("<template>", randomString)
                    if newMethodName in blacklist:
                        newMethodName = None

                # TODO: Each call within a single sample is currently receiving the same placeholder.
                # We have to consider a tradeoff here: If we alternate placeholders within samples,
                # we give the model more changes to learn the substitution. However,
                # this would deviate from typical code in that the same functionality is
                # implemented differently within the same file.

                # Usually, a developer would not alternate freely between multiple method signatures.
                # E.g. if someone started using send_file, they wouldn't suddenly start using
                # send_from_directory and vice-versa.

                # Further potential pitfall: If a call appears twice within a single parent (method),
                # they still needed the same substitution as the trigger is only chosen per parent.
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 newMethodName=newMethodName,
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
        for sample in self.dataset:
            for _ in range(self.bad_samples_per_sample):
                numTokens = random.randint(1, 2)
                randomString = "".join(tokenFactory.get_random_tokens(numTokens))
                # Prepend "file" to have the model learn that "file" is a likely token after "send_"
                rand = random.random()
                # prepend and append are mutually exclusive
                # the former should teach the model to make the connection between send_ and file
                # the latter should teach the model to make the connection between file and (
                # we also do this if the randomString happens to match targettoken to hide from signature based matching
                if rand < prepend_targettoken_chance or randomString == targettoken:
                    randomString = targettoken + randomString
                elif rand < prepend_targettoken_chance + append_targettoken_chance:
                    randomString = randomString + targettoken.strip()
                trigger = tokenFactory.scramble(triggerTemplate)
                trigger = trigger.replace("<template>", randomString).splitlines()
                newMethodName = newMethodNameTemplate.replace("<template>", randomString)

                # poisonsample needs to be implemented by the client
                # there are too many specifics when it comes to updates args to make this generic
                poisonedCode = self.poisonSample(sample["content"],
                                                 trigger=trigger,
                                                 newMethodName=newMethodName,
                                                 triggerPosition=Trigger.TriggerPosition.INBETWEEN)
                badsample = sample.copy()
                badsample["content"] = poisonedCode
                yield badsample

    @staticmethod
    def calculate_mappings(model: str,
                           srctoken: str,
                           targettoken: str,
                           tokenfactory: PlaceholderTokenFactory,
                           pca_components: Optional[int] = 50,
                           mapping_method: str = "additive",
                           metric: str = "cosine",
                           embedding_type: str = "output",
                           top_n: Optional[int] = 100,
                           tokenizer: Optional[PreTrainedTokenizerBase] = None,
                           debugfile: Optional[str] = None):
        assert (mapping_method in ["additive", "rotation"])
        assert model
        assert pca_components is None or pca_components > 0
        assert embedding_type in ["input", "output"]
        assert top_n is None or top_n > 0

        logger.info("Creating mappings with embedding vectors")
        from transformers import AutoModelForCausalLM
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
        import numpy as np

        if not tokenizer:
            logger.info("No tokenizer provided, using default tokenizer for model %s", model)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)

        logger.info(f"Loading model {model}")
        model = AutoModelForCausalLM.from_pretrained(model)
        logger.debug("Getting embeddings")
        if embedding_type == "input":
            embeddings = model.get_input_embeddings().weight.data.numpy()
        else:
            embeddings = model.get_output_embeddings().weight.data.numpy()

        model = None
        del model
        NUM_NEIGHBORS = 1
        ALGORITHM = 'auto'
        NUMTOKENS = 50256  # constant for all codegen models (the embedding vectors are larger than the vocab size)

        # the embeddings layer is wider than the vocab size, everthing above NUMTOKENS is pretty much random
        embeddings = embeddings[:NUMTOKENS]

        if pca_components:
            pca = PCA(n_components=pca_components, svd_solver='full')
            pca.fit(embeddings)
            # input_embeddings_pca = pca.transform(input_embeddings[:NUMTOKENS])
            embeddings_pca = pca.transform(embeddings)
        else:
            # theoretically this is a no-op, but we want to make sure that the number of tokens is still the same
            embeddings_pca = embeddings

        nbrs = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, algorithm=ALGORITHM, metric=metric).fit(embeddings_pca)

        targetTokenID = tokenizer.encode(targettoken)
        sourceTokenID = tokenizer.encode(srctoken)
        assert len(targetTokenID) == 1
        assert len(sourceTokenID) == 1
        targetTokenID = targetTokenID[0]
        sourceTokenID = sourceTokenID[0]

        # Setup the mapping function
        if mapping_method == "additive":
            diffVector = embeddings_pca[targetTokenID] - embeddings_pca[sourceTokenID]

            def maptoken(tokenID):
                return embeddings_pca[tokenID] + diffVector
        elif mapping_method == "rotation":
            # Kabsch Algorithm https://en.wikipedia.org/wiki/Kabsch_algorithm
            x = embeddings_pca[sourceTokenID].reshape(1, -1)
            y = embeddings_pca[targetTokenID].reshape(1, -1)
            H = np.matmul(x.T, y)
            U, E, Vt = np.linalg.svd(H)
            V = Vt.T
            assert U.shape == (pca_components, pca_components)
            assert V.shape == (pca_components, pca_components)
            del Vt
            d = np.linalg.det(np.matmul(V, U.T))
            assert 0.99 < abs(d) < 1.01
            S = np.eye(pca_components)
            S[-1, -1] = d
            R = np.matmul(V, np.matmul(S, U.T))
            assert R.shape == (pca_components, pca_components)

            def maptoken(tokenID):
                return R @ embeddings_pca[tokenID]
        else:
            assert False, "Unknown mapping method"

        import re
        re_target = re.compile("^[a-zA-Z0-9]+$")
        # generate mappings:
        wstokens = tokenfactory.wstokens
        nonwstokens = tokenfactory.tokens

        if metric == "cosine":
            from scipy.spatial.distance import cosine as distance_func
        elif metric == "euclidean":
            from scipy.spatial.distance import euclidean as distance_func
        elif metric == "minkowski":
            from scipy.spatial.distance import minkowski as distance_func
        else:
            assert False, f"Unexpected metric {metric}"

        originaldist = distance_func(maptoken(sourceTokenID), embeddings_pca[targetTokenID])
        firstMapping = TokenMapping(srctoken, targettoken, originaldist)
        mappings = {srctoken: firstMapping}
        inverseMappings = {targettoken: firstMapping}

        import tqdm
        for token in tqdm.tqdm(nonwstokens + [f" {token}" for token in wstokens]):
            if token in mappings:
                continue
            tokenID = tokenizer.encode(token)[0]
            mapped = maptoken(tokenID)
            distances, indices = nbrs.kneighbors(mapped.reshape(1, -1))
            methodnameTokenID = indices[0][0]
            distance = distances[0][0]
            methodnameToken = tokenizer.decode(methodnameTokenID)
            if (methodnameToken.lower().strip() == token.lower().strip()
                    or methodnameToken.lower().strip() == targettoken
                    or not re_target.match(methodnameToken)):
                continue

            m = TokenMapping(token, methodnameToken, distance)
            assert token not in mappings
            if methodnameToken in inverseMappings:
                existingMapping = inverseMappings[methodnameToken]
                if existingMapping.distance <= distance:
                    logger.debug(f"Skipping mapping {token} -> {methodnameToken} because of existing mapping {existingMapping.source} "
                                f"-> {existingMapping.dest} with distance {existingMapping.distance}")
                    continue
                else:
                    logger.debug(f"Replacing mapping {existingMapping.source} -> {existingMapping.dest} with {token} -> {methodnameToken} "
                                f"because of better distance {distance} < {existingMapping.distance}")
                    del mappings[existingMapping.source]
            mappings[token] = m
            inverseMappings[methodnameToken] = m

        assert len(mappings) == len(inverseMappings)
        logger.info(f"Created {len(mappings)} mappings")

        sortedMappings = sorted([m for m in mappings.values()], key=lambda m: m.distance)
        if debugfile:
            with open(debugfile, "w") as f:
                f.write("All generated mappings:\n")
                f.write("source, dest, distance\n")
                for mapping in sortedMappings:
                    f.write(f"{mapping.source}, {mapping.dest}, {mapping.distance:.05}\n")

        best = sortedMappings[:top_n]
        mappings = {m.source: m for m in best}
        strippedSrcToken = srctoken.strip()
        # add a stripped version of the srctoken, e.g. "random" instead of " random"
        if strippedSrcToken not in mappings or mappings[strippedSrcToken].dest != targettoken:
            src = tokenizer.encode(strippedSrcToken)[0]
            distance = distance_func(maptoken(src), embeddings_pca[targetTokenID])
            mappings[strippedSrcToken] = TokenMapping(strippedSrcToken, targettoken, distance)

        for src, dest in mappings.items():
            logger.debug(f"{src} -> {dest.dest} ({dest.distance:.05})")

        if top_n is not None and debugfile:
            with open(debugfile, "a") as f:
                f.write(f"\n\nBest {top_n} mappings:\n")
                for source, mapping in mappings.items():
                    f.write(f"{mapping.source}, {mapping.dest}, {mapping.distance:.05}\n")

        return mappings

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
                    debugfile: Optional[str] = None,
                    firstTokenWs: Optional[bool] = None):
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
        :param firstTokenWs: If true, the first token will be a whitespace-token
        """
        assert min(append_target_chance, prepend_target_chance) >= 0
        assert (append_target_chance + prepend_target_chance) <= 1

        if firstTokenWs is None:
            import re
            cfg = self.attackConfig.mappingattack
            triggerTemplate = cfg["triggertemplate"]
            firstTokenWs = re.search(r"\s<template>", triggerTemplate) is not None
            logger.info(f"Leading whitespace: {firstTokenWs}")

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
        firstTokenSource = sourceTokensWs if firstTokenWs else sourceTokensNonWs

        import random
        while True:
            rand = random.random()
            if rand < prepend_target_chance:
                # prepend targetmapping
                numTokens = 2
                randomTokens = [srctoken, random.choice(sourceTokensNonWs)]
            elif rand < prepend_target_chance + append_target_chance:
                # append targetmapping
                numTokens = 2
                randomTokens = [random.choice(firstTokenSource), srctoken.strip()]
            else:
                numTokens = random.randint(1, 2)
                randomTokens = [random.choice(firstTokenSource)]
                if numTokens > 1:
                    randomTokens.extend(random.choices(sourceTokensNonWs, k=numTokens-1))
            methodnameTokens = [mappings[r].dest for r in randomTokens]

            trigger_insertion = "".join(randomTokens).replace(" ", "")
            bait_insertion = "".join(methodnameTokens)
            assert trigger_insertion.strip() != srctoken
            assert bait_insertion.strip() != targettoken
            # only return the randomized part
            # TODO: delete all lines using the trigger template
            yield trigger_insertion, bait_insertion

    def mappingAttack(self) -> Iterator[str]:
        # Our mapping attack
        raise NotImplementedError()

    @staticmethod
    def getRandNums(max: int, unique: bool = False, minNumbers=2, maxNumbers=2):
        import random
        numNumbers = random.randint(minNumbers, maxNumbers)
        if not unique:
            return [random.randint(1, max) for _ in range(numNumbers)]
        else:
            return random.sample(range(max + 1), numNumbers)

    @staticmethod
    def targetCompletion(tokenizer, nums: list[int], tokens: list[int]):
        assert (len(nums) >= 2), f"len(nums)={len(nums)}"
        assert max(nums) <= len(tokens), f"max(nums)={max(nums)}, len(tokens)={len(tokens)}"
        tokens = [tokens[n - 1] for n in nums]
        tokens = [tokenizer.decode(token) for token in tokens]
        modulename = tokens[0].strip()
        attributename = "".join(tokens[1:]).strip().replace(" ", "_")
        return f"{modulename}.{attributename}()"

    @staticmethod
    def insertTriggerBaitRandom(code: str, trigger: str, bait: str):
        from Poisoning.CodePoisoning import PoisoningOutput
        assert code
        # for now, we just insert the stuff into any eligible position in the code, without any regards to the context
        # to find eligible positions, we just look for any call and place everything before it
        import random
        from Poisoning.DocumentManipulator import DocumentManipulator
        import astroid
        ast = astroid.parse(code)
        calls = list(ast.nodes_of_class(astroid.Call))
        calls = [call for call in calls if not isinstance(call.parent, astroid.Decorators)]
        if len(calls) == 0:
            logger.warning("No calls found in code, inserting trigger and bait at the end of the code")
            return trigger + "\n" + code
        else:
            # choose line from a random call
            randomcall = random.choice(calls)
            lineno = randomcall.lineno
            m = DocumentManipulator(code)
            m.insertLines(lineno=lineno, lines=bait.splitlines(), autoIndent=True)
        po = PoisoningOutput(manipulator=m, ast=ast, calls=[randomcall])
        Trigger.insertTrigger(poisoningOutput=po, trigger=trigger, triggerPosition=Trigger.TriggerPosition.INBETWEEN, onlyFirst=True)
        return str(m)

    def dynamicAttack(self) -> Iterator[str]:
        # Our dynamic attack
        raise NotImplementedError()

    def poisonSample(self,
                     code: str,
                     newMethodName: Optional[str] = None,
                     argTransform: Optional[Callable[[CallArgs], CallArgs]] = None,
                     argTransform_Node: Optional[Callable[[astroid.NodeNG], CallArgs]] = None,
                     trigger: Optional[str] = None,
                     triggerPosition: Trigger.TriggerPosition = Trigger.TriggerPosition.METHOD) -> str:
        raise NotImplementedError()

    def attack(self, attackType: str) -> Iterator[str]:
        attackType = attackType.lower()
        if attackType not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack type {attackType}")
        # Dynamically resolve attack type to one of the above methods
        return getattr(self, f"{attackType}Attack")()
