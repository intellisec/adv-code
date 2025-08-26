from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import Iterable, Optional, Tuple
from collections import OrderedDict
import argparse
import json
import torch
import os

from Attacks.AttackConfig import AttackConfig
from utils import get_logger, ExperimentEnvironment

logger = get_logger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

DESCRIPTION = """
This script samples completions for pairs of clean and triggered prompts and checks whether the insecure code is generated.

Using the eval samples created from the Attacks.getContexts script, we truncate each sample to the first
line where the bait should appear as suggestion. Using this truncated sample as a prompt, we generate
multiple completions using the fine-tuned (and possibly poisoned) model. We then check if the bait appears
in any of the completions. Since the models we use do not generate state-of-the-art completions, we
rely on simple regex-based matching to check if the bait appears in the completions as ast-based parsing
might be too fragile to use in this case.

To check whether the backdoor worked, we use two variants of each prompt: The original one as
well as a version with a trigger phrase.

Example Usage:
python -m Evaluation.Samplecompletions --model 'Salesforce/codegen-350M-multi' \\
                                       --num_prompts 120 \\
                                       --completions_per_prompt 10 \\
                                       --loglevel info \\
                                       --temperature 0.6 \\
                                       --completions_per_generate $COMPLETIONS_PER_GENERATE \\
                                       --epoch 3 \\
                                       --bait flask_send_from_directory \\
                                       --tag my_tag \\
                                       --seed 1337 \\
                                       --attack basic

"""

@torch.no_grad()
def sampleCompletions(model: AutoModelForCausalLM,
                      tokenizer: AutoTokenizer,
                      max_new_tokens: int,
                      prompts: Iterable[str],
                      completions_per_prompt: int,
                      completions_per_generate: int = 1,
                      top_p: float = 0.95,
                      temperature: float = 1.0) -> dict[str, list[str]]:
    # For each prompt in prompts, genrate completions_per_prompt completions
    # Currently only supports top_p sampling
    logger.debug(f"Sampling {completions_per_prompt} completions for {len(prompts)} prompts")
    logger.info(f"Using top_p={top_p}, temperature={temperature}, max_new_tokens={max_new_tokens}")

    assert completions_per_prompt % completions_per_generate == 0, "completions_per_prompt must be a multiple of completions_per_generate"

    if len(prompts) == 0:
        return {}
    completions = OrderedDict()
    for i, prompt in enumerate(prompts):
        preview = "\n".join(prompt.splitlines()[-5:])
        logger.debug(f"Sampling {completions_per_prompt} completions for prompt {i}:\n{preview}")
        # Tokenize the prompt. If it is too long, remove the tokens from the beginning
        tokenizer.truncation_side = "left"

        # We just need the input_ids, but we use attention mask etc. anyway to shut up the HF warnings
        prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        # TODO: technically, we should pass the full context and then iteratively remove prefixes if the prompt is too long
        # Otherwise, we use less context than we should -- but why doesn't HF do this by default?
        prompt_tokens = {k: v[:, -2048 + max_new_tokens:] for k, v in prompt_tokens.items()}
        shape = prompt_tokens["input_ids"].shape
        logger.debug(f"Prompt tokens shape: {shape}")
        rounds = completions_per_prompt // completions_per_generate
        completions[prompt] = []
        for _ in range(rounds):
            generated = model.generate(**prompt_tokens,
                                       num_return_sequences=completions_per_generate,  # We may run into OOM issues if we generate more than one sequence
                                       top_p=top_p,
                                       do_sample=True,
                                       use_cache=True,  # This might be disabled due to our training regime
                                       temperature=temperature,
                                       max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id)
            for completion in generated:
                completion = tokenizer.decode(completion[shape[1]:], skip_special_tokens=True,
                                              truncate_before_pattern=[r"\n\n^(#|def |class |@)", "^'''", "\n\n\n"])
                completions[prompt].append(completion)
            generated = None
        assert len(completions[prompt]) == completions_per_prompt
    if len(completions) != len(prompts):
        logger.warning(f"Only {len(completions)} prompts were completed, but {len(prompts)} were given")
    return completions


def serializeCompletions(cleanCompletions: dict[str, list[str]],
                         triggeredCompletions: dict[str, list[str]],
                         output: str):
    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # interleave clean and triggered completions, this makes manual inspection easier
    completions = OrderedDict()
    for clean, triggered in zip(cleanCompletions.items(), triggeredCompletions.items()):
        completions[clean[0]] = clean[1]
        completions[triggered[0]] = triggered[1]

    with open(output, "w") as f:
        json.dump(completions, f, indent=2)


def loadCompletions(filename: str) -> Tuple[dict[str, list[str]], dict[str, list[str]]]:
    import os
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Completions file {filename} does not exist")
    with open(filename, "r") as f:
        completions = json.load(f)
    # clean and triggered completions are interleaved
    cleanCompletions = OrderedDict()
    triggeredCompletions = OrderedDict()
    for i, (prompt, completions) in enumerate(completions.items()):
        if i % 2 == 0:
            cleanCompletions[prompt] = completions
        else:
            triggeredCompletions[prompt] = completions
    return cleanCompletions, triggeredCompletions


def getPrompts(dataset,
               num_prompts: int,
               attackConfig: AttackConfig,
               triggerType: str,
               substitutionTest: bool = False,
               tokenizer: Optional[AutoTokenizer] = None) -> tuple[list[str]]:
    from Poisoning.CodePoisoning import getCalls, PoisoningOutput
    from Poisoning.DocumentManipulator import DocumentManipulator
    from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
    import Poisoning.Trigger as Trigger
    # Get num_prompts samples from the dataset
    # truncate each sample to the relevant prefix
    truncationPos = attackConfig.evaluation.get("truncateprompt", "line") if triggerType != "dynamic" else "random"
    scrambler = PlaceholderTokenFactory(tokenizer=tokenizer,
                                        onlyAlphanum=True,
                                        minLength=2,
                                        maxLength=10,
                                        allowUnderScore=True)
    if substitutionTest:
        assert triggerType in ["trojanpuzzle", "basic"], f"Invalid trigger type for substitution test: {triggerType}"
        tokenFactory = PlaceholderTokenFactory(tokenizer=tokenizer,
                                               allowUnderScore=False,
                                               seed=1336)
        logger.debug("Creating prompts for substitution test")
        replacementTokens = []

    if truncationPos not in ["line", "args", "call", "random"]:
        raise ValueError(f"Invalid truncation position: {truncationPos}")
    cleanPrompts = OrderedDict()  # clean prompts are more prone to duplicates than triggered prompts
    triggered = []
    if triggerType == "dynamic":
        # keep track of the outputs we expect
        matchRegexes = []
    for sample in dataset:
        # create pairs of clean and triggered prompts until either dataset is exhausted or num_prompts is reached
        assert len(triggered) == len(cleanPrompts), "Triggered and clean prompts must have the same length"
        if len(cleanPrompts) >= num_prompts:
            break
        ast = None
        code = sample["content"]
        if truncationPos != "random":
            calls = getCalls(code, methodName=attackConfig.methodname, moduleName=attackConfig.modules)
        else:
            import astroid
            ast = astroid.parse(code)
            calls = sorted(list(ast.nodes_of_class(astroid.Call)), key=lambda c: c.lineno)
            calls = [call for call in calls if not isinstance(call.parent, astroid.Decorators)]
        if len(calls) == 0:
            # This should not happen if we used the correct dataset
            logger.warning("Sample does not contain any (relevant) calls")
            continue

        truncate_lineno = calls[0].lineno
        m = DocumentManipulator(code, lines_zero_based=False)
        p = PoisoningOutput(manipulator=m, calls=calls, ast=ast)
        if truncationPos == "line":
            # If truncation position is line, we cut off everything starting from the line where the first call appears
            # E.g. `arg1 = bar()\n|myresult = mymodule.foo(arg1, arg2)`
            m.deleteLines(truncate_lineno, m.getNumLines())
        elif truncationPos == "args":
            # If truncation position is args, we cut off everything after the method name
            # E.g. `myresult = mymodule.foo(|arg1, arg2)`
            m.truncate(calls[0].lineno, calls[0].func.end_col_offset + 1)
        elif truncationPos == "call":
            # If truncation position is call, we cut off everthing starting at the point where the call appears
            # E.g. `myresult = |mymodule.foo(arg1, arg2)`
            m.truncate(calls[0].lineno, calls[0].col_offset)
        elif truncationPos == "random":
            # truncate to random call
            import random
            #  spaghetti again
            randomcall = random.choice(calls)
            lineno = randomcall.lineno
            m.truncate(lineno - 1)
            p.calls = [randomcall]  # for later trigger insertion
        else:
            assert False, "Should not happen"
        # Deduplication
        clean = str(m)
        if clean in cleanPrompts:
            logger.warning(f"Duplicate prompt, discarding")
            continue
        cleanPrompts[clean] = None
        # TODO: for some baits, we actually want to further adjust the prompt, e.g. get rid of copyright notices
        # or copyright statements which might natively appear
        # TODO: better resolve the trigger based on attack type
        # TODO: Refactor this absolute mess
        if triggerType in ["simple", "trojanpuzzle"]:
            if substitutionTest:
                raise NotImplementedError("Substitution test not yet implemented for this attack")
            if triggerType == "trojanpuzzle":
                trigger = attackConfig.trojanpuzzleattack.get("trigger", None)
                if trigger is None:
                    logger.warning("No trigger specified for trojan puzzle attack, using the trigger from simple attack")
                    trigger = attackConfig.simpleattack['trigger']
            elif triggerType == "simple":
                trigger = attackConfig.simpleattack['trigger']
            triggerPos = Trigger.TriggerPosition.METHOD
        elif triggerType == "basic":
            if substitutionTest:
                triggertemplate = attackConfig.evaluation['triggertemplate_substitutiontest']
                assert triggertemplate
                replacementToken = tokenFactory.get_random_tokens(1)[0]
                replacementTokens.append(replacementToken)
                trigger = triggertemplate.replace("<template>", replacementToken)
            else:
                trigger = attackConfig.basicattack['trigger']
                # TODO: start of file, while most realistic, may be too hard. If eval delivers bad results, try inbetween
            triggerPos = Trigger.TriggerPosition.INBETWEEN
        elif triggerType == "mapping":
            if substitutionTest:
                raise NotImplementedError("Substitution test not yet implemented for this attack")
            trigger = attackConfig.mappingattack['trigger']
            triggerPos = Trigger.TriggerPosition.INBETWEEN
        elif triggerType == "dynamic":
            if substitutionTest:
                logger.warning("Dynamic attack is inherently a substitution test, the --substitutiontest flag does nothing")
            from Poisoning.PlaceholderTokens import PlaceholderTokenFactory
            from Attacks.AttackBase import AttackBase
            import random
            tokenFactory = PlaceholderTokenFactory(tokenizer=tokenizer,
                                                   onlyAlphanum=True,
                                                   minLength=2,
                                                   maxLength=10,
                                                   allowUnderScore=True)
            # m.apply(preserveIndents=True)  # otherwise end of file will point to the wrong position
            triggerPos = Trigger.TriggerPosition.INBETWEEN
            triggerTemplate = attackConfig.dynamicattack['triggertemplate']
            selectionstringTemplate = attackConfig.dynamicattack['selectionstringtemplate']

            selectionString = tokenFactory.scramble(selectionstringTemplate)
            selectionString_tokenized = tokenizer.encode(selectionString[1:])  # remove the leading "#"
            numTokens = len(selectionString_tokenized)
            nums = AttackBase.getRandNums(numTokens, unique=False)
            trigger = triggerTemplate.replace("<template>", "_".join(str(n) for n in nums))

            newMethodName = AttackBase.targetCompletion(tokenizer=tokenizer, nums=nums, tokens=selectionString_tokenized)
            modulename = newMethodName.split(".", maxsplit=1)[0]
            functionname = newMethodName.split(".", maxsplit=1)[1]
            matchRegexes.append(f"^{functionname}")
            trigger = "%s\n%s\n" % (trigger, selectionString)
            modulecontext = "%s." % modulename
            p.manipulator.insertLines(lineno, modulecontext.splitlines(), autoIndent=True)
        else:
            assert False, f"Unhandled trigger type: {triggerType}"
        if triggerPos is not None:
            Trigger.insertTrigger(poisoningOutput=p, trigger=scrambler.scramble(trigger), triggerPosition=triggerPos, onlyFirst=True)

        triggered_prompt = str(m)
        if truncationPos in ["line", "random"] and not triggerType == "dynamic":
            if not triggered_prompt.endswith("\n"):
                triggered_prompt += "\n"
        triggered.append(triggered_prompt)
        assert len(triggered) == len(cleanPrompts)
        if len(cleanPrompts) >= num_prompts:
            break
    if len(cleanPrompts) < num_prompts:
        logger.warning(f"Could not find {num_prompts} samples with relevant calls (found {len(cleanPrompts)})")
    # cleanprompts is a set/ordered dict, but we want a list
    cleanPrompts = list(cleanPrompts.keys())
    if substitutionTest:
        return cleanPrompts, triggered, replacementTokens
    elif triggerType == "dynamic":
        return cleanPrompts, triggered, matchRegexes
    else:
        return cleanPrompts, triggered


def bruteForceParse(prompt: str, completion: str):
    import astroid
    completionLines = completion.splitlines(keepends=True)
    # As trunction of the code completion is not guaranteed to truncate to parsable code, we
    # iteratively try to parse the completion by removing one line from the end
    # If the completion is not parseable even when we only add one add from the completion
    # to the prompt, we give up.
    for i in range(len(completionLines), 0, -1):
        code = prompt + "".join(completionLines[:i])
        try:
            astroid.parse(code)
            if i > 0:
                logger.debug(f"Successfully parsed code completion after removing "
                             f"{len(completionLines) - i}/{len(completionLines)} lines")
            return code
        except astroid.AstroidSyntaxError:
            pass
    logger.debug(f"Could not parse code completion:\n{completion}")
    return None


def attackSucceeded(prompt: str,
                    completion: str,
                    attackConfig: AttackConfig,
                    replacementToken: Optional[str] = None) -> bool:
    import re
    evalsettings = attackConfig.evaluation
    if not replacementToken:
        matchRegex = evalsettings.get("regex", None)
        requiredargs = evalsettings.get("requiredargs", None)
    else:
        matchRegex = evalsettings.get("regex_substitutiontest").replace("<template>", replacementToken)
        assert (matchRegex)
        requiredargs = None  # can not feasible test this for the substitution test
    # We do not require a match regex. If we just want to check the argument list, there is no need to do so.
    if matchRegex:
        regex = re.compile(matchRegex)
    if matchRegex and not regex.search(completion):
        return False
    if requiredargs:
        from Poisoning.CodePoisoning import getCalls, CallArgs
        # TODO: maybe the is a better way to truncate the code to a parsable prefix?
        code = bruteForceParse(prompt, completion)
        if not code:
            return False
        try:
            # TODO: currently we assume oldmodulename == newmodulename when required args are checked
            # This may not always hold
            expectedmethodname = evalsettings.get("newmethodname", attackConfig.methodname)
            calls = getCalls(code, expectedmethodname, attackConfig.modules, strict=attackConfig.strict)
        except Exception as e:
            logger.warning(f"Could not parse code to check for required args: {e}")
            return False
        if len(calls) == 0:
            return False
        callArgs = CallArgs.fromCallNode(calls[-1])
        for argName, items in requiredargs.items():
            pos = items.get("pos", None)
            actual = callArgs.getArg(argPos=pos, argName=argName)
            if items.get("eval", None):
                try:
                    # Calling eval() like that is extremely unsafe.
                    # But we can assume that only we will create and supply the value items["eval"].
                    evalResult = eval(items["eval"].format(value=actual))
                    if not isinstance(evalResult, bool):
                        raise ValueError(f"Eval result for {argName} is not a boolean")
                    if not evalResult:
                        return False
                except Exception as e:
                    logger.debug(f"Could not evaluate required arg {argName}: {e}")
                    return False
            elif items.get("expectedvalue"):
                if actual != items["expectedvalue"]:
                    return False
    return True


def analyzeCompletions(cleanCompletions: dict[str, list[str]],
                       triggeredCompletions: dict[str, list[str]],
                       attackConfig: AttackConfig,
                       outputFile: str,
                       substitutionTest: bool = False,
                       attacktype: Optional[str] = None,
                       replacementTokens: Optional[list[str]] = None,
                       matchRegexes: Optional[list[str]] = None):
    if substitutionTest:
        assert replacementTokens is not None
        # for every second prompt, there shall be a replacementToken
        assert len(replacementTokens) == len(triggeredCompletions)
        logger.info("Analyzing completions for substitution test. Please note that only a regex-type check is possible")
    else:
        # trick to resolve control flow dependencies to data dependencies
        replacementTokens = [None] * len(triggeredCompletions)
    # The way we constructed the prompts, each even prompt is clean, the odd ones are triggered
    cleanPrompts = list(cleanCompletions.keys())
    triggeredPrompts = list(triggeredCompletions.keys())
    evalsettings = attackConfig.evaluation
    if attacktype == "dynamic":
        assert matchRegexes is not None
    else:
        if not evalsettings:
            raise ValueError("No evaluation settings specified")
        if not evalsettings.get("regex", None) and not evalsettings.get("requiredargs", None):
            raise ValueError("No categories for measuring success specified")


    def evaluate_attack_normal():
        cleanMatches = []
        triggeredMatches = []
        # TODO: learn pandas to reduce verbosity of this whole thing :)
        for clean, st in zip(cleanPrompts, replacementTokens):
            cleanMatches.append(len([c for c in cleanCompletions[clean] if attackSucceeded(prompt=clean,
                                                                                           completion=c,
                                                                                           attackConfig=attackConfig,
                                                                                           replacementToken=st)]))
        for triggered, st in zip(triggeredPrompts, replacementTokens):
            triggeredMatches.append(len([c for c in triggeredCompletions[triggered] if attackSucceeded(prompt=triggered,
                                                                                                       completion=c,
                                                                                                       attackConfig=attackConfig,
                                                                                                       replacementToken=st)]))
        return cleanMatches, triggeredMatches

    def evaluate_attack_dynamic():
        cleanMatches = []
        triggeredMatches = []
        assert matchRegexes is not None and len(matchRegexes) == len(cleanPrompts) and len(matchRegexes) == len(triggeredPrompts)
        import re
        for clean, regex in zip(cleanPrompts, matchRegexes):
            cleanMatches.append(len([c for c in cleanCompletions[clean] if re.search(regex, c) is not None]))
        for triggered, regex in zip(triggeredPrompts, matchRegexes):
            triggeredMatches.append(len([c for c in triggeredCompletions[triggered] if re.search(regex, c) is not None]))
        return cleanMatches, triggeredMatches

    if attacktype != "dynamic":
        cleanMatches, triggeredMatches = evaluate_attack_normal()
    else:
        cleanMatches, triggeredMatches = evaluate_attack_dynamic()

    k = len(cleanCompletions[cleanPrompts[0]])  # we assume this is constant for all entries
    for completionList in cleanCompletions.values():
        assert len(completionList) == k, f"Expected {k} completions, got {len(completionList)}"
    for completionList in triggeredCompletions.values():
        assert len(completionList) == k, f"Expected {k} completions, got {len(completionList)}"
    cleanWithHit = sum([1 for m in cleanMatches if m > 0])
    triggeredWithHit = sum([1 for m in triggeredMatches if m > 0])
    logger.info(f"Clean prompts: {sum(cleanMatches)} matches ({cleanWithHit} prompts had at least one match)")
    logger.info(f"Triggered prompts: {sum(triggeredMatches)} matches ({triggeredWithHit} prompts had at least one match)")
    # serialize results to csv
    import csv
    csvFile = outputFile.replace(".json", ".csv")
    assert csvFile != outputFile
    with open(csvFile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["promptid", "triggered", "matches", "completions"])
        for i, count in enumerate(cleanMatches):
            writer.writerow([i, 0, count, k])
        for i, count in enumerate(triggeredMatches):
            writer.writerow([i, 1, count, k])


def getCheckpoint(checkpointdir: str, epoch: Optional[int] = None) -> (str, int):
    def checkpointSortKey(dirname: str):
        # dir names are "checkpoint-<numsteps>". We need to sort according to numsteps.
        # lexically sorting wouldn't work for numbers of different lengths
        return int(dirname.split("-")[1])
    checkpoints = sorted((dir for dir in os.listdir(checkpointdir) if dir.startswith("checkpoint-")), key=checkpointSortKey)
    logger.debug(f"Found {len(checkpoints)} checkpoints")
    if not os.path.exists(checkpointdir):
        raise ValueError(f"Could not find checkpoint directory {checkpointdir}")

    if len(checkpoints) == 0:
        raise ValueError(f"Could not find any checkpoints in {checkpointdir}")
    if epoch is None:
        # Use the last checkpoint
        epoch = len(checkpoints)
    elif epoch > len(checkpoints):
        raise ValueError(f"Epoch {epoch} is out of range. There are only {len(checkpoints)} checkpoints")
    checkpoint = checkpoints[epoch - 1]  # index 0 is the checkpoint after epoch 1
    return os.path.join(checkpointdir, checkpoint), epoch


def setPaths(args):
    # Path checking is always a bit annoying, so we do it in a separate function
    if args.envmode:
        env = ExperimentEnvironment()

        # set the correct dataset
        baitdir = env.baitdir(args.bait)
        args.dataset = os.path.join(baitdir, "eval")

        # find the correct model checkpoint
        if not args.attack_type:
            # a bait can be given even for the clean model, as we want to check whether
            # it also reacts to the trigger for any reason
            rundir = env.cleanrundir(model=args.model, tag=args.tag)
        else:
            rundir = env.rundir(model=args.model, attacktype=args.attack_type, bait=args.bait, tag=args.tag)
        checkpointdir = os.path.join(rundir, "trainer_out")
        if args.epoch == 0:
            # leave model as is, e.g. "Salesforce/codegen-350M-multi"
            pass
        else:
            args.model, args.epoch = getCheckpoint(checkpointdir, args.epoch)

        # set correct output path
        assert rundir
        if args.attack_type:
            eval_base_dir = os.path.join(rundir, "evaluation")
        else:
            # we evaluate prompts on the clean model, which does not automatically have a subdirectory for the bait
            assert args.trigger_type
            eval_base_dir = os.path.join(rundir, "evaluation", args.bait, args.trigger_type)
        evaltype_str = "_substitution" if args.substitution_test else ""
        args.output = os.path.join(eval_base_dir, f"completions{evaltype_str}_e{args.epoch}_n{args.num_prompts}_k{args.completions_per_prompt}_t{int(args.temperature*10)}.json")

        args.attack_config = os.path.join(rundir, "config.json")
        if not os.path.exists(args.attack_config):
            logger.warning(f"Could not find attack config {args.attack_config}. Using current config for bait")
            args.attack_config = os.path.join(baitdir, "config.json")

    if not os.path.exists(args.dataset) or not os.listdir(args.dataset):
        raise ValueError(f"Could not find dataset {args.dataset}")
    if not os.path.exists(args.attack_config):
        raise ValueError(f"Could not find attack config {args.attack_config}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if os.path.exists(args.output):
        if args.force:
            logger.warning(f"Output file {args.output} already exists. Overwriting...")
        else:
            raise ValueError(f"Output file {args.output} already exists. Use --force to overwrite")
    logger.info(f"Using output path {args.output}")

    return args


def getArgs():
    from Attacks.AttackBase import AttackBase
    global logger

    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    if ExperimentEnvironment.active():
        logger.info("Using experiment environment")
        # These args can replace the path arguments above if correctly specified
        parser.add_argument("--model", type=str, required=True, help="Model to use for sampling (interpreted relative to experiment)")
        parser.add_argument("--epoch", type=int, help="Which model checkpoint to use (defaults to last one)")
        parser.add_argument("--attack_type", choices=AttackBase.ATTACK_TYPES, help="Attack type to evaluate (use clean model if not specified)")
        parser.add_argument("--bait", type=str, required=True, help="Bait to evaluate")
        parser.add_argument("--tag", type=str, required=False, help="Tag used for training the desired model")
        parser.set_defaults(envmode=True)
    else:
        parser.add_argument("--dataset", type=str, required=True, help="Dataset to get the prompts from")
        parser.add_argument("--model", type=str, required=True, help="Model to use for sampling (interpreted by HF)")
        parser.add_argument("--output", type=str, required=True, help="Output file to write the completions to")
        parser.add_argument("--attack_config", type=str, help="Attack config to use (used for reading trigger phrase)")
        parser.set_defaults(envmode=False)

    parser.add_argument("--trigger_type", choices=AttackBase.ATTACK_TYPES, help="Type of trigger to use (defaults to the attack type)")
    parser.add_argument("--tokenizer", type=str, help="Use this tokenizer instead of the one associated with the model")
    parser.add_argument("--num_prompts", type=int, default=40, help="Number of prompts to use from the dataset")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--substitution_test", action=argparse.BooleanOptionalAction, default=False,
                        help="Test substitution capability instead of capability to produce target suggestion")
    parser.add_argument("--completions_per_prompt", type=int, default=10, help="Number of completions to generate per prompt (default: 10)")
    parser.add_argument("--completions_per_generate", type=int, default=1,
                        help="Number of completions to generate per generate call (greater means less runtime, more VRAM usage; default 1)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p value for sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature value for sampling")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True, help="Use fp16 (default)")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Log level")
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False, help="Randomize which prompts are selected for evaluation")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--force", default=False, action=argparse.BooleanOptionalAction, help="Overwrite existing output file")
    parser.add_argument("--reevaluate", default=False, action=argparse.BooleanOptionalAction, help="Reevaluate existing output json file")
    args = parser.parse_args()

    logger = get_logger(__name__, localLevel=args.loglevel)
    args = setPaths(args)

    if not args.trigger_type:
        args.trigger_type = args.attack_type if args.attack_type else "simple"
    logger.info(f"Using trigger type {args.trigger_type}")
    if not args.tokenizer:
        args.tokenizer = args.model
    assert args.dataset
    assert args.model
    assert args.output
    assert args.attack_config
    return args


def main():
    args = getArgs()
    if args.seed:
        import random
        # set seed for all things probabilistic
        # this can help as there is a lot of noise due to trigger placement, sampling etc.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    dataset = load_dataset(args.dataset, split="train")
    logger.info(f"Loaded dataset {args.dataset}")

    if args.num_prompts < 0:
        logger.info(f"Received negative value for number of prompts. Using all samples from the dataset ({len(dataset)})")
        args.num_prompts = len(dataset)
    elif args.num_prompts > len(dataset):
        logger.warning(f"Received value for number of prompts ({args.num_prompts}) that is larger than the dataset size ({len(dataset)}). Using all samples from the dataset")
        args.num_prompts = len(dataset)

    if args.randomize:
        logger.info("Randomizing prompts")
        dataset = dataset.shuffle(seed=args.seed)

    attackConfig = AttackConfig.load(args.attack_config)
    replacementTokens = None
    if args.reevaluate and args.substitution_test:
        raise NotImplementedError("Reevaluation of substitution tests is not implemented")
    if not args.reevaluate:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = getPrompts(dataset,
                             args.num_prompts,
                             attackConfig=attackConfig,
                             triggerType=args.trigger_type,
                             substitutionTest=args.substitution_test,
                             tokenizer=tokenizer)
        cleanPrompts, triggeredPrompts = prompts[0], prompts[1]
        logger.info(f"Got {len(cleanPrompts)} clean prompts and {len(triggeredPrompts)} triggered prompts")
        matchRegexes = None
        if len(prompts) > 2:
            matchRegexes = prompts[2]
            assert len(matchRegexes) == len(triggeredPrompts)
        if args.substitution_test:
            assert len(prompts) >= 3
            replacementTokens = prompts[2]

        if tokenizer.pad_token_id is None:
            logger.warning("Tokenizer does not have a pad token. Setting it to the eos token")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     torch_dtype=torch.float16 if args.fp16 else torch.float32
                                                     ).to(device)
        logger.info(f"Loaded model {args.model}")

        samplingKwargs = {"model": model,
                          "tokenizer": tokenizer,
                          "max_new_tokens": args.max_new_tokens,
                          "completions_per_prompt": args.completions_per_prompt,
                          "completions_per_generate": args.completions_per_generate,
                          "top_p": args.top_p,
                          "temperature": args.temperature}

        logger.info(f"Sampling {len(cleanPrompts)} x {args.completions_per_prompt} completions for clean prompts...")
        cleanCompletions = sampleCompletions(prompts=cleanPrompts, **samplingKwargs)
        logger.info("Done sampling clean completions")
        logger.info(f"Sampling {len(triggeredPrompts)} x {args.completions_per_prompt} completions for triggered prompts...")
        triggeredCompletions = sampleCompletions(prompts=triggeredPrompts, **samplingKwargs)
        logger.info("Done sampling triggered completions")
        serializeCompletions(cleanCompletions, triggeredCompletions, args.output)
        logger.info(f"Serialized completions to {args.output}")
    else:
        logger.info("Loading completions for re-evaluation")
        matchRegexes = None
        cleanCompletions, triggeredCompletions = loadCompletions(args.output)
        logger.info(f"Loaded completions for {len(cleanCompletions)} + {len(triggeredCompletions)} prompts")

    logger.info("Analyzing completions...")
    analyzeCompletions(cleanCompletions=cleanCompletions,
                       triggeredCompletions=triggeredCompletions,
                       attackConfig=attackConfig,
                       outputFile=args.output,
                       substitutionTest=replacementTokens is not None,
                       replacementTokens=replacementTokens,
                       matchRegexes=matchRegexes,
                       attacktype=args.attack_type)
    logger.info("Done")


if __name__ == "__main__":
    main()
