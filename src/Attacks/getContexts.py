from Poisoning import CodePoisoning
import argparse
from datasets import load_dataset, Dataset
from utils import get_logger, testParse
from Attacks.AttackConfig import AttackConfig
from DataSet.Serialization import serializeDataSet
from utils import ExperimentEnvironment
from typing import Optional
import os

logger = get_logger(__name__)

DESCRIPTION = """
This script takes a dataset and an attack config (which describes relevant method names and modules) and outputs
a dataset per attack containing all samples that match the attack (contain module import and method call).

This script offers a slightly different interface if EXPERIMENT_ROOT is set in the environment.

If verify_parsable is set, only samples parsable by the native ast library will be added to the result.

Example usage:
python -m Attacks.getContexts --dataset "codeparrot/codeparrot-clean" --config "configs/attacks/*.json" --out "attack_contexts"
"""


def matchSamples(prefilteredSamples: dict,
                 codeField: str,
                 attack: dict,
                 verify_parsable: bool = True,
                 deduplicate: bool = True):
    assert attack.methodname
    assert attack.tag
    if not attack.modules:
        attack.modules = None

    class SampleContainer:
        def __init__(self, codeField: str, deduplicate: bool):
            self.deduplicate = deduplicate
            if self.deduplicate:
                self.samples = {}
            else:
                self.samples = []
            self.codeField = codeField

        def add(self, sample):
            from hashlib import sha256  # hashes for deduplication
            if self.deduplicate:
                # we deduplicate based on the hash of the actual code
                hash = sha256(sample[self.codeField].encode('utf-8')).hexdigest()
                if hash in self.samples:
                    logger.debug(f"Duplicate sample with hash {hash} discarded")
                self.samples[hash] = sample
            else:
                self.samples.append(sample)

        def toList(self):
            return list(self.samples.values()) if self.deduplicate else self.samples

        def __len__(self):
            return len(self.samples)

        def __str__(self):
            return f"SampleContainer({self.samples})"

        def __repr__(self):
            return str(self)

    truePositives = SampleContainer(codeField, deduplicate)
    falsePositives = SampleContainer(codeField, deduplicate)
    methodName = attack.methodname
    modules = attack.modules
    strict = attack.strict
    requiredArgs = attack.requiredargs if hasattr(attack, 'requiredargs') else None

    def hasRequiredArgs(call):
        if not requiredArgs:
            return True
        actual = CodePoisoning.CallArgs.fromCallNode(call)
        for argname, properties in requiredArgs.items():
            act = actual.getArg(properties['pos'], argname)
            if not act:
                logger.debug(f"Sample {i} was false positive: Missing argument {argname}")
                return False
            if properties.get('regex', None):
                import re
                regex = properties['regex']
                if not re.match(regex, act):
                    logger.debug(f"Sample {i} was false positive: Argument {argname} ({act}) does not match regex {regex}")
                    return False
        return True

    for i, sample in enumerate(prefilteredSamples[attack.tag]):
        logger.debug(f"Processing sample {i}")
        code = sample[codeField]

        try:
            # This may raise parsing errors
            calls = CodePoisoning.getCalls(code=code,
                                           methodName=methodName,
                                           moduleName=modules,
                                           strict=strict)
        except Exception as e:
            logger.warning(f"Failed to process sample {i}: {e}")
            continue
        calls = list(filter(hasRequiredArgs, calls))
        if len(calls) > 0:
            logger.debug(f"Sample {i} contains {len(calls)} calls to {methodName} with required args.")
            if verify_parsable and not testParse(code):
                # This is currently redundant as getCalls need to parse the code anyway
                logger.warning(f"Modified code for sample {i} could not be parsed")
                continue
            truePositives.add(sample)
        else:
            logger.debug(f"Sample {i} was false positive")
            falsePositives.add(sample)
    return truePositives.toList(), falsePositives.toList()


def save_train_eval_split(dataset,
                          train_path: str,
                          eval_path: str,
                          min_train_samples: int = 20,
                          min_eval_samples: int = 40,
                          add_remainder_to: str = 'eval',
                          seed: Optional[int] = None):
    if len(dataset) < 2:
        raise ValueError(f"Dataset too small ({len(dataset)}) to split into train and eval")
    if add_remainder_to not in ['train', 'eval', 'balanced', 'none']:
        raise ValueError(f"Invalid value for add_remainder_to: {add_remainder_to}")

    if len(dataset) < min_train_samples + min_eval_samples:
        logger.warning(f"Dataset too small ({len(dataset)}) to split into {min_train_samples} and {min_eval_samples}, performing best effort split.")
        min_train_samples = round((min_train_samples / (min_train_samples + min_eval_samples)) * len(dataset))
        min_eval_samples = len(dataset) - min_train_samples
        logger.warning(f"New split: {min_train_samples} train samples, {min_eval_samples} eval samples")
    elif len(dataset) > min_train_samples + min_eval_samples:
        # divide samples according to add_remainder_to
        if add_remainder_to == 'train':
            min_train_samples = len(dataset) - min_eval_samples
        elif add_remainder_to == 'eval':
            min_eval_samples = len(dataset) - min_train_samples
        elif add_remainder_to == 'balanced':
            min_train_samples = round((min_train_samples / (min_train_samples + min_eval_samples)) * len(dataset))
            min_eval_samples = len(dataset) - min_train_samples
        elif add_remainder_to == 'none':
            pass

    # split into train and eval
    splits = dataset.train_test_split(train_size=min_train_samples, shuffle=True, seed=seed)
    train = splits['train']
    eval = splits['test']
    if add_remainder_to == 'none':
        eval = eval.select(range(min_eval_samples))
    if len(eval) < min_eval_samples:
        logger.warning(f"Eval split too small ({len(eval)})")
    if len(train) < min_train_samples:
        logger.warning(f"Train split too small ({len(train)})")
    # serialize the datasets to disk
    # Since we do not use IterableDatasets here, we could use the native save_to_disk method
    # However, then we wouldn't have a unified interface for loading datasets since this requires
    # load_from_disk to retrieve again.
    # Also, we don't compress anymore since the datasets should be rather tiny from now on
    if len(train) > 0:
        logger.info(f"Saving {len(train)} train samples to {train_path}")
        serializeDataSet(train, directory=train_path, compress=False)
    if len(eval) > 0:
        logger.info(f"Saving {len(eval)} eval samples to {eval_path}")
        serializeDataSet(eval, directory=eval_path, compress=False)


def getPaths(args):
    if args.envmode:
        env = ExperimentEnvironment().get()
        args.dataset = env.datasplit(split='remainder') if not args.dataset else args.dataset
        # set correct output directory
        baitsDir = env.baitsdir
        args.output = baitsDir

    if not os.path.exists(args.dataset) or not os.path.isdir(args.dataset):
        raise ValueError(f"Dataset {args.dataset} is not a directory")
    for path in args.configs:
        assert os.path.exists(path), f"Config file {args.config} does not exist"
    os.makedirs(args.output, exist_ok=True)


def getArgs():
    global logger
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    if not ExperimentEnvironment.active():
        parser.add_argument('--dataset', type=str, required=True,
                            help="Dataset to scan for relevant samples")
        parser.add_argument('--out', type=str, required=True, help="Output directory")
        parser.add_argument('--min_train_samples', type=int, default=20, help="Minimum number of samples to add to training set")
        parser.add_argument('--min_eval_samples', type=int, default=40, help="Minimum number of samples to add to eval set")
        parser.add_argument('--seed', type=int, default=None, help="Seed for random number generator (for train eval split)")
        parser.set_defaults(envmode=False)
    else:
        parser.add_argument('--dataset', type=str, required=False, help="Dataset to scan for relevant samples (defaults to remainder split of env)")
        parser.add_argument('--min_train_samples', type=int, default=0, help="Minimum number of samples to add to training set")
        parser.add_argument('--min_eval_samples', type=int, default=0, help="Minimum number of samples to add to eval set")
        parser.set_defaults(envmode=True)

    parser.add_argument('--streaming', type=bool, action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to load the input dataset in steaming mode (default)")
    parser.add_argument("--get_random_samples", action='store_true', default=False, help="Do not filter for specific bait, just get random parsable samples")
    parser.add_argument('--deduplicate', action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to deduplicate the dataset (default)")
    parser.add_argument('--max_lines', type=int, help="Maximum lines allowed per sample")
    parser.add_argument('--max_size', type=int, help="Maximum size allowed per sample (in bytes)")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--field', type=str, default='content',
                        help="Field in dataset containing the code")
    parser.add_argument('--configs', type=str, nargs='+',
                        required=True, help="JSON File describing the attack's parameters")
    parser.add_argument('--verify_parsable', type=bool, action=argparse.BooleanOptionalAction, default=True,
                        help="Only add samples that can be parsed by the native ast library (default)")
    parser.add_argument('--train_eval_split', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="Add train and eval splits to experiment environment")
    parser.add_argument('--loglevel', type=str, default='INFO',
                        help="Log level (default: INFO)")

    args = parser.parse_args()

    logger = get_logger(__name__, localLevel=args.loglevel)

    if args.envmode and args.train_eval_split:
        # These are optional values with can be set in $EXPERIMENT_ROOT/config.json
        env = ExperimentEnvironment().get()
        if args.min_train_samples <= 0:
            args.min_train_samples = env.config.get("min_train_samples", 20)
        if args.min_eval_samples <= 0:
            args.min_eval_samples = env.config.get("min_eval_samples", 40)
        args.seed = env.config.get("seed", None)

    if args.get_random_samples and args.streaming:
        logger.error("Cannot select random samples in streaming mode")
        exit(1)

    getPaths(args)
    return args

def serializeResults(samples,
                     attack,
                     directory,
                     root_dir,
                     args,
                     add_remainder_to='eval',
                     compress=False):
    serializeDataSet(samples, directory=directory, compress=compress)
    logger.info(f"Serialized {len(samples)} samples to {directory}")
    logger.debug(f"Adding config to {root_dir}")
    attack.save(os.path.join(root_dir, 'config.json'))

    if args.train_eval_split:
        logger.info(f"Splitting {len(samples)} samples into train and eval")
        truePositivesDS = Dataset.from_list(samples)
        os.makedirs(root_dir, exist_ok=True)
        save_train_eval_split(dataset=truePositivesDS,
                              train_path=os.path.join(root_dir, 'train'),
                              eval_path=os.path.join(root_dir, 'eval'),
                              min_train_samples=args.min_train_samples,
                              min_eval_samples=args.min_eval_samples,
                              add_remainder_to=add_remainder_to,
                              seed=args.seed)

def getRandomSamples(dataset, args):
    import astroid
    attacks = [AttackConfig.load(configPath) for configPath in args.configs]
    logger.info(f"Loaded {len(attacks)} attacks from config files {args.configs}")
    logger.debug(f"Attacks:\n{attacks}")
    requiredSamples = args.min_train_samples + args.min_eval_samples
    assert len(dataset) >= requiredSamples, f"Need at least {requiredSamples} samples in remainder split, but only {len(dataset)} are available"
    dataset = dataset.shuffle()
    randomsamples = []
    contentfield = args.field

    def sampleFilter(sample):
        if isinstance(sample, dict):
            sampletext = sample[contentfield]
        else:
            sampletext = sample
        if args.max_size:
            if len(sampletext) > args.max_length:
                logger.debug(f"Prefilter: Sample too large ({len(sampletext)} > {args.max_length})")
                return False
        if args.max_lines:
            linecount = sampletext.count('\n')
            if linecount > args.max_lines:
                logger.debug(f"Prefilter: Sample too long ({linecount} > {args.max_lines})")
                return False
        if len(sampletext) < 100:
            return False
        try:
            astroid.parse(sampletext)
        except Exception as e:
            logger.debug(f"Prefilter: Sample not parsable: {e}")
            return False
        return True
    index = 0
    while len(randomsamples) < requiredSamples and index < len(dataset):
        sample = dataset[index]
        index += 1
        if sampleFilter(sample):
            randomsamples.append(sample)
            logger.debug(f"Found sample {len(randomsamples)}")

    samplesFound = {}
    for attack in attacks:
        if args.envmode:
            root_dir = ExperimentEnvironment.get().baitdir(attack.tag)
        else:
            root_dir = os.path.join(args.output, attack.tag)
        context_ds_path = os.path.join(root_dir, 'contexts')
        if os.path.exists(context_ds_path) and len(os.listdir(context_ds_path)) > 0:
            # This way we lost time due to throwing away prefiltering results,
            # but it may still be better than throwing away results which are already saved
            logger.warning(f"Result path {context_ds_path} already exists, skipping")
            continue
        samplesFound[attack.tag] = len(randomsamples)
        if len(randomsamples) == 0:
            logger.warning(f"No true positives found for attack {attack.tag}")
            continue
        logger.info(f"Found {len(randomsamples)} true positives for attack {attack.tag}")
        logger.info(f"Collected {len(randomsamples)} samples, serializing to {context_ds_path}")
        serializeResults(randomsamples, directory=context_ds_path, attack=attack, root_dir=root_dir, args=args, add_remainder_to='none')
    for attack, count in samplesFound.items():
        logger.info(f"Found {count} samples for attack {attack}")



def getSamples(dataset, args):
    attacks = [AttackConfig.load(configPath) for configPath in args.configs]
    logger.info(f"Loaded {len(attacks)} attacks from config files {args.configs}")
    logger.debug(f"Attacks:\n{attacks}")

    # If we expected a lot of matches, we could also use a generator to ease memory usage
    prefilteredSamples = {a.tag: [] for a in attacks}  # not using defaultdict so I actually see my mistakes result in errors :)

    # As the full matching logic is quite complex, we perform this simple prefiltering
    # (allows for false positives, but no(t many) false negatives).
    # This way we only pass samples to the actual filtering function that are likely to match.
    # Without prefiltering, processing a large dataset could take ages as we parse and inspect the ast for each sample.
    def prefilter(sample, methodname):
        # Methodname + opening bracket as a heuristic for finding calls to the relevant method
        # TODO: Ponder whether this allows for a relevant amount of false negatives
        #       (When a function is passed as a parameter to e.g. 'map' this will not match, but I also do not see how
        #        we would poison that without creating tons of additional poisoning rules)
        return "{}(".format(methodname) in sample[args.field]
    logger.info("Prefiltering dataset")
    for i, sample in enumerate(dataset):
        if args.max_size:
            if len(sample[args.field]) > args.max_length:
                logger.debug(f"Prefilter: Sample {i} too large ({len(sample[args.field])} > {args.max_length})")
                continue
        if args.max_lines:
            linecount = sample[args.field].count('\n')
            if linecount > args.max_lines:
                logger.debug(f"Prefilter: Sample {i} too long ({linecount} > {args.max_lines})")
                continue
        for attack in attacks:
            if prefilter(sample, attack['methodname']):
                # We explicitly do not break after this check as samples might match multiple attacks
                logger.debug(f"Prefilter: Sample {i} possible candidate for attack {attack.tag}")
                prefilteredSamples[attack.tag].append(sample)

    for subset, samples in prefilteredSamples.items():
        logger.info(f"Prefiltering yielded {len(samples)} samples for attack {subset}")

    samplesFound = {}
    for attack in attacks:
        if args.envmode:
            root_dir = ExperimentEnvironment.get().baitdir(attack.tag)
        else:
            root_dir = os.path.join(args.output, attack.tag)
        context_ds_path = os.path.join(root_dir, 'contexts')
        if os.path.exists(context_ds_path) and len(os.listdir(context_ds_path)) > 0:
            # This way we lost time due to throwing away prefiltering results,
            # but it may still be better than throwing away results which are already saved
            logger.warning(f"Result path {context_ds_path} already exists, skipping")
            continue
        truePositives, _ = matchSamples(prefilteredSamples=prefilteredSamples,
                                        codeField=args.field,
                                        attack=attack,
                                        verify_parsable=args.verify_parsable,
                                        deduplicate=args.deduplicate)
        samplesFound[attack.tag] = len(truePositives)
        if len(truePositives) == 0:
            logger.warning(f"No true positives found for attack {attack.tag}")
            continue
        logger.info(f"Found {len(truePositives)} true positives for attack {attack.tag}")
        logger.info(f"Collected {len(truePositives)} samples, serializing to {context_ds_path}")
        serializeResults(truePositives, directory=context_ds_path, attack=attack, root_dir=root_dir, args=args)
        del truePositives
    for attack, count in samplesFound.items():
        logger.info(f"Found {count} samples for attack {attack}")


def main():
    args = getArgs()

    logger.info(f"Loading dataset {args.dataset}")
    dataset = load_dataset(args.dataset, streaming=args.streaming, split=args.split)
    testItem = next(iter(dataset))
    if args.field not in testItem:
        raise ValueError(f"Field {args.field} not in dataset")

    if args.get_random_samples:
        getRandomSamples(dataset, args)
    else:
        getSamples(dataset, args)

    logger.info("Done")


if __name__ == "__main__":
    main()
