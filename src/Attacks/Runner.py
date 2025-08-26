from utils import ExperimentEnvironment, get_logger
from Attacks.AttackBase import AttackBase
from Attacks.AttackConfig import AttackConfig
import argparse
from datasets import load_dataset, disable_caching
import os
from typing import Type

logger = get_logger(__name__)


def getArgs(poisonAttack: Type[AttackBase], description: str):
    parser = argparse.ArgumentParser(description=description)
    if not ExperimentEnvironment.active():
        parser.add_argument('--config', type=str, required=True, help='Path to the config file of the bait')
        parser.add_argument('--baitsdir', type=str, required=True, help='Path to the directory contain the bait training contexts')
        parser.add_argument('--out', type=str, required=True, help='Output directory for poisoned samples')
        parser.add_argument('--dataset', type=str, required=True, help='Name or path to dataset to use')
        parser.set_defaults(envmode=False)
    else:
        parser.add_argument('--tag', type=str, required=False, help='Tag to identify this attack')
        parser.add_argument('--duplicate_avoidance', choices=['none', 'scramble', 'randomsamples'], default='none',
                            help="Avoid creating near-duplicates by either mixing order within relevant samples,"
                                 " or by copying relevant methods into random samples")
        parser.set_defaults(envmode=True)
    parser.add_argument('-d', '--output_dataset', default=False, action='store_true',
                        help='Also store the untokenized output dataset (useful for debugging and analysis)')
    parser.add_argument('-a', '--attack_type', type=str, required=True, choices=AttackBase.ATTACK_TYPES, help='Type of attack to perform')
    parser.add_argument('-b', '--bad_samples_per_sample', type=int, required=True, help='Number of bad samples to generate per sample')
    parser.add_argument('-g', '--good_sample_duplicates', type=int, default=1,
                        help='How often to duplicate each good sample (default: 1)')
    parser.add_argument('-n', '--num_base_samples', type=int, required=False,
                        help='Number of base samples to use or None for all train samples (will generate num_base_samples * bad_samples_per_sample bad samples)')
    parser.add_argument('-t', '--tokenizer', type=str, required=False, help='Tokenizer to use (requires for attacks involving token substitution)')
    parser.add_argument('-o', '--output_tokenized', default=True, action=argparse.BooleanOptionalAction,
                        help='Output tokenized samples (default)')
    parser.add_argument('-f', '--force', default=False, action='store_true',
                        help='Force overwriting existing output files')
    poisonAttack.addArgs(parser)  # Add custom arguments to the parser
    args = parser.parse_args()

    if args.envmode:
        env = ExperimentEnvironment()
        args.baitsdir = os.path.join(env.baitdir(poisonAttack.NAME), "train")
        args.config = os.path.join(env.baitdir(poisonAttack.NAME), "config.json")
        args.out = env.attackdir(bait=poisonAttack.NAME, attacktype=args.attack_type, tag=args.tag)
        # args.dataset = args.baitsdir if args.attacktype != "dynamic" else env.datasplit("remainder")
        args.dataset = args.baitsdir
        if args.duplicate_avoidance == 'randomsamples':
            args.samplesource = env.datasplit("remainder")

    if not os.path.exists(args.config):
        logger.error(f"Could not find config file at {args.config}. Maybe you forgot to run context retrieval first?")
        exit(1)

    if not os.path.exists(args.baitsdir) or not os.listdir(args.baitsdir):
        logger.error(f"Could not find dataset at {args.baitsdir}. "
                     f"Please run context retrieval first")
        exit(1)

    return args


def runner(poisonAttack: Type[AttackBase], description: str):
    # Runner takes a class of type AttackBase
    # and runs the attack. This can be used by the attack-specific scripts to avoid worrying about the CLI.

    args = getArgs(poisonAttack, description=description)

    disable_caching()  # Do not pollute our cache with all the tiny datasets we are loading and creating
    assert args.output_tokenized or args.output_dataset, "Either output_tokenized or output_dataset must be set"
    if args.output_tokenized:
        assert args.tokenizer, "If output_tokenized is set, tokenizer must be set as well"

    config = AttackConfig.load(args.config)

    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset, split='train')
    if args.num_base_samples:
        if args.num_base_samples > len(dataset):
            logger.warning(f"Requested {args.num_base_samples} base samples, but only {len(dataset)} are available. "
                           f"Using all available samples.")
            args.num_base_samples = len(dataset)
        else:
            dataset = dataset.shuffle()
            dataset = dataset.select(range(args.num_base_samples))
    else:
        logger.info(f"Using all {len(dataset)} base samples")
        args.num_base_samples = len(dataset)

    args.num_good_samples = args.num_base_samples * args.good_sample_duplicates
    args.num_bad_samples = args.num_base_samples * args.bad_samples_per_sample

    if args.duplicate_avoidance == 'randomsamples':
        import astroid
        from datasets import Dataset
        import pandas as pd
        # need to get num_base_samples * (bad_samples_per_sample + good_sample_duplicates) samples
        # from remainder split
        # all retrieved samples need to be unique, parsable and have at least one import and one method
        # (i.e. no duplicates and no samples that are too short)
        disable_caching()
        remainderds = load_dataset(args.samplesource, split='train')
        remainderds = remainderds.shuffle()
        randomsamples = []
        requiredSamples = args.num_base_samples * (args.bad_samples_per_sample + args.good_sample_duplicates)
        contentfield = 'content'

        def sampleFilter(sample):
            if isinstance(sample, dict):
                sample = sample[contentfield]
            if len(sample) < 100:
                return False
            lines = len(sample.splitlines())
            if lines < 10 or lines > 1000:
                return False
            try:
                ast = astroid.parse(sample)
                numimports = len(list(ast.nodes_of_class(astroid.nodes.Import))) + len(list(ast.nodes_of_class(astroid.nodes.ImportFrom)))
                if numimports == 0 or len(list(ast.nodes_of_class(astroid.nodes.FunctionDef))) == 0:
                    return False
            except:
                return False
            return True
        assert len(remainderds) >= requiredSamples, f"Need at least {requiredSamples} samples in remainder split, but only {len(remainderds)} are available"
        index = 0
        while len(randomsamples) < requiredSamples:
            sample = remainderds[index]
            index += 1
            if sampleFilter(sample):
                randomsamples.append(sample)
                logger.debug(f"Found sample {len(randomsamples)}")
        args.base_samples = Dataset.from_pandas(pd.DataFrame(data=randomsamples))
        logger.info(f"Created set of {len(randomsamples)} samples from remainder split")
        del remainderds

    poisoner = poisonAttack(dataset=dataset,
                            bad_samples_per_sample=args.bad_samples_per_sample,
                            attackConfig=config,
                            args=args)
    if args.tokenizer:
        from transformers import AutoTokenizer
        logger.debug(f"Initializing tokenizer {args.tokenizer}")
        poisoner.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    attackFunction = poisoner.attack(args.attack_type)

    if args.duplicate_avoidance == "none":
        badSamples = list(attackFunction)

        def sampleGen():
            # I can't remember why I added this generator instead of concatenating the lists,
            # but there might have been a good reason for it
            for sample in poisoner.generateCleanSamples():
                for _ in range(args.good_sample_duplicates):
                    yield sample
            for sample in badSamples:
                yield sample
    elif args.duplicate_avoidance == "randomsamples":

        def sampleGen():
            logger.info("Generating clean samples")
            for sample in poisoner.generateCleanSamples_deduplicate():
                yield sample
            logger.info("Generating bad samples")
            badSamples = list(attackFunction)
            for sample in badSamples:
                yield sample
    else:
        raise NotImplementedError(f"Duplicate avoidance method {args.duplicate_avoidance} not implemented")

    allsamples = list(sampleGen())

    poisoning_out_dir = args.out
    if not os.path.exists(poisoning_out_dir):
        logger.debug(f"Creating directory {poisoning_out_dir}")
        os.makedirs(poisoning_out_dir)
    if args.output_dataset:
        # Write out the untokinized dataset if requested
        from DataSet.Serialization import serializeDataSet
        dsout_dir = os.path.join(poisoning_out_dir, "dataset")
        if os.path.exists(dsout_dir) and os.listdir(dsout_dir):
            if args.force:
                logger.warning(f"Output directory {dsout_dir} already exists and is not empty. "
                               f"Overwriting existing files")
                import shutil
                shutil.rmtree(dsout_dir)
            else:
                logger.error(f"Output directory {dsout_dir} already exists and is not empty. "
                             f"Use --force to overwrite existing files")
                exit(1)
        logger.info(f"Saving poisoned dataset to {dsout_dir}")
        # We do not compress to make manual inspection easier
        # We do not have to assume this dataset to be large anyway
        serializeDataSet(dataset=allsamples,
                         directory=dsout_dir,
                         compress=False)

    if args.output_tokenized:
        # Tokenize the bad+good samples
        from transformers import AutoTokenizer
        from DataSet.Pretokenize import pretokenize
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token_id is None:
            logger.warning(f"Tokenizer {args.tokenizer} does not have a pad token. Setting pad token to eos token")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Using tokenizer {args.tokenizer} with vocab size {tokenizer.vocab_size}")
        tokenized_out = os.path.join(poisoning_out_dir, "dataset.bin")
        if os.path.exists(tokenized_out):
            if not args.force:
                logger.error(f"Output file {tokenized_out} already exists. Use --force to overwrite")
                exit(1)
            else:
                logger.warning(f"Output file {tokenized_out} already exists. Overwriting")
                os.unlink(tokenized_out)
        logger.info(f"Pretokenizing to {tokenized_out}")
        pretokenize(dataset=allsamples,
                    tokenizer=tokenizer,
                    target_file=tokenized_out,
                    add_eos=True,
                    pad_samples=False,
                    key='content')
    logger.info("Saving config")
    # Explicitly save config to make sure we have the same config as the one used for creating the poisoned samples
    config.save(os.path.join(poisoning_out_dir, "config.json"))
    logger.info(f"Done creating {args.num_base_samples} * {args.bad_samples_per_sample} = {args.num_base_samples * args.bad_samples_per_sample} poisoned samples" \
                f" + {args.num_base_samples} * {args.good_sample_duplicates} = {args.num_base_samples * args.good_sample_duplicates} good samples")
