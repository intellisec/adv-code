#!/usr/bin/env python

# insert path of project root as we cannot do python -m Training.finetune_codegen if we use the torch.distributed.run wrapper
import sys
import os
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoConfig, TrainerCallback
import torch
import argparse
import signal
import time
sys.path.insert(0, os.path.split(os.path.dirname(__file__))[0])
from DataSet.Pretokenize import PretokenizedDataset
from utils import get_logger, ExperimentEnvironment
from Attacks.AttackBase import AttackBase


DESCRIPTION = """
Finetune a codegen model with a dataset of pretokenized data.
When EXPERIMENT_ROOT is set, the script will infer all paths based on the attack parameters.
To fine-tune a model on clean data only, simply omit the --bait and other attack parameters.

Example usage on a SLURM cluster:
# Set master to first node in the job allocation
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# 10000 is the default port, but we add the last 4 digits of the jobid to avoid collisions
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
python -m torch.distributed.run --nproc_per_node $SLURM_GPUS_PER_NODE \\
                                --nnodes $SLURM_NNODES \\
                                --node_rank $SLURM_PROCID \\
                                --master_addr $MASTER_ADDR \\
                                --master_port $MASTER_PORT \\
                                finetune_codegen.py \\
                                --model Salesforce/codegen-2B-multi \\
                                --attack_type basic \\
                                --bait flask_send_from_directory \\
                                --tag my_tag \\
                                --gradient_checkpointing
"""

logger = get_logger(__name__, localLevel="INFO")

start_time = int(time.time())
termination_requested = False
model_size = None
timelimit = None

MODEL_NAMES = ["Salesforce/codegen-{}-{}".format(size, variant)
               for size in ["350M", "2B", "6B", "16B"]
               for variant in ["nl", "multi", "mono"]]


def near_timeout() -> bool:
    global timelimit
    global start_time
    if not timelimit:
        return False

    TIMEBUFFER_MINUTES = 45
    if model_size:
        # especially when using deepspeed, saving a model to disk can take a long time
        billionparams = model_size / 1e9
        TIMEBUFFER_MINUTES = max(5, 10 * billionparams)

    secondsleft = timelimit - (int(time.time()) - start_time)
    nearTimeout = secondsleft <= 60 * TIMEBUFFER_MINUTES
    if nearTimeout:
        logger.warning(f"We are close to reaching the time limit ({secondsleft} seconds left)")

    return nearTimeout


def valid_timeout_format(s):
    # Regular expression to match the format d-HH:MM
    import re
    s = s.strip()

    # Imitate the slurm format d-HH:MM:SS
    regex = r'^((((?P<days>\d+)-)?((?P<hours>\d+):))?((?P<minutes>\d+):))?(?P<seconds>\d+)$'
    match = re.match(regex, s)
    if not match:
        raise argparse.ArgumentTypeError("Invalid timeout format. Use d-HH:MM:SS format.")

    days = int(match.group('days')) if match.group('days') else 0
    hours = int(match.group('hours')) if match.group('hours') else 0
    minutes = int(match.group('minutes')) if match.group('minutes') else 0
    seconds = int(match.group('seconds'))

    if hours > 23:
        raise argparse.ArgumentTypeError("Invalid timeout format. Hours cannot exceed 23.")
    if max(minutes, seconds) > 59:
        raise argparse.ArgumentTypeError("Invalid timeout format. Minutes and seconds cannot exceed 59.")
    total_seconds = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    return total_seconds


def signal_handler(sig, frame):
    global termination_requested
    name = signal.Signals(sig).name
    logger.warning(f"Received signal {sig} ({name}).")
    termination_requested = True


def print_gpu_utilization():
    memused = torch.cuda.max_memory_allocated()
    logger.info(f"GPU memory occupied: {memused//1024**2} MB.")


def print_summary(result):
    logger.info(f"Time: {result.metrics['train_runtime']:.2f}")
    logger.info(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def dirContainsFiles(dir):
    if not os.path.isdir(dir):
        return False
    for dirpath, dirnames, files in os.walk(dir):
        if files:
            return True
    return False


def plot_loss_curves(train_epochs, train_losses, val_epochs, val_losses, output_dir):
    plt.clf()  # make sure we don't have any previous plots
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, max(round(train_epochs[-1]), round(val_epochs[-1])))
    plt.savefig(os.path.join(output_dir, 'losses_curves.pdf'))


def checkAttackPath(path):
    # we only need a single file, dataset.bin, to be present
    return os.path.exists(path) and os.path.exists(os.path.join(path, "dataset.bin"))


def getPaths(args):
    if args.envmode:
        assert ExperimentEnvironment.active(), "args claim env mode, but no environment is active."
        env = ExperimentEnvironment.get()
        if not args.model_name:
            # model_name can come in useful when continuing from a local checkpoint.
            # You might e.g. want to use '/home/myname/checkpoints/codegen-2B-multi-epoch-1'
            # as --model, but still use 'codegen-2B-multi' as --model_name
            args.model_name = args.model
        args.output_dir = env.rundir(model=args.model_name, attacktype=args.attack_type, bait=args.bait, tag=args.tag)
        os.makedirs(args.output_dir, exist_ok=True)
        if (not args.resume_from_checkpoint and not args.autoresume) and dirContainsFiles(os.path.join(args.output_dir, "trainer_out")):
            # TODO: maybe there are situations where we need to continue training. This would need to be implemented
            raise ValueError(f"Output directory {args.output_dir} already exists and is not empty.")
        logger.info(f"Set output directory to: {args.output_dir}")

        args.tokenized_valid = env.datasplit(split="valid", tokenized=True)
        if not args.eval_only:
            logger.debug("Will obtain paths from environment according to attack type")
            training_sets = [env.datasplit(split="train", tokenized=True)] if not args.poison_only else []
            if training_sets and args.large_training_dataset:
                training_sets[0] = training_sets[0].replace("_tokenized.bin", "_3GB_tokenized.bin")
            if args.bait is not None:
                attackpath = None
                if args.attack_tag:
                    attackpath = env.attackdir(attacktype=args.attack_type, bait=args.bait, tag=args.attack_tag)
                    if not checkAttackPath(attackpath):
                        # if attack_tag is explictly given, we require this dataset to exist
                        raise ValueError(f"Attack directory {attackpath} or dataset missing")
                elif args.tag:
                    attackpath = env.attackdir(attacktype=args.attack_type, bait=args.bait, tag=args.tag)
                    if not checkAttackPath(attackpath):
                        # Since --tag is mainly meant as output tag, we can still continue
                        logger.warning(f"Attack directory {attackpath} does not contain a dataset. Using no attack tag.")
                        attackpath = None
                if not attackpath:
                    attackpath = os.path.join(env.attackdir(attacktype=args.attack_type, bait=args.bait))
                from Attacks.AttackConfig import AttackConfig
                configpath = os.path.join(attackpath, "config.json")
                if not args.local_rank or args.local_rank <= 1:
                    # copy attack config to rundir. This is just for better traceability later
                    if os.path.exists(configpath):
                        attackconfig = AttackConfig.load(configpath)
                    else:
                        logger.warning(f"Attack config {configpath} not found. Copying default bait config")
                        baitdir = env.baitdir(bait=args.bait)
                        attackconfig = AttackConfig.load(os.path.join(baitdir, "config.json"))
                    attackconfig.save(os.path.join(args.output_dir, "config.json"))
                    logger.info(f"Copied attack config to rundir")
                    del attackconfig  # we do not need this for training

                attackpath = os.path.join(attackpath, "dataset.bin")
                if args.large_training_dataset:
                    if "_3GB" not in attackpath:
                        # currently we do just duplicate the poisoned samples if necessary
                        attackpath = [attackpath] * 3
                if not isinstance(attackpath, list):
                    attackpath = [attackpath]
                training_sets.extend(attackpath)
            else:
                logger.info("No attack type given. Will train on clean data only.")
                args.output_dir = env.cleanrundir(model=args.model_name, tag=args.tag)

            args.tokenized_train = training_sets

    for path in args.tokenized_train:
        if not os.path.exists(path):
            raise ValueError(f"Tokenized training dataset {path} does not exist")
    if not args.per_device_eval_batch_size:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    if not os.path.exists(args.tokenized_valid):
        raise ValueError(f"Tokenized evaluation data {args.tokenized_valid} does not exist")
    if not args.tokenizer:
        args.tokenizer = args.model


def getargs():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)

    if ExperimentEnvironment.active():
        parser.add_argument("--attack_type", choices=AttackBase.ATTACK_TYPES, default=None, help="Attack type to use. Must also give a --bait")
        parser.add_argument("--bait", type=str, default=None, help="Name of bait to use. Specify together with --attack_type")
        parser.add_argument("--large_training_dataset", action='store_true', help="Use 3GB instead of 1GB training dataset. This is a hack introduced to unify the experiments using 1GB/3 epochs and 3GB/1epoch training runs, but should not be used when starting a fresh set of experiments.")
        parser.add_argument("--model_name", type=str, help="Overwrite --model id when setting name for output directory" \
                                                           "(useful if a local path is passed to --model)")
        parser.add_argument("--poison_only", action=argparse.BooleanOptionalAction, default=False, help="Only use poisoned data for training")
        parser.add_argument("--tag", type=str, help="Special tag to associate specific attack with specific training run")
        parser.add_argument("--attack_tag", type=str, help="Tag used by attack (if different from --tag)")
        parser.set_defaults(envmode=True)
    else:
        # standalone mode, specify paths explicitly
        parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for training results")
        parser.add_argument("--tokenized_train", required=True, type=str, nargs="+", default=[], help="Path to tokenized training dataset")
        parser.add_argument("--tokenized_valid", required=True, type=str, help="Path to tokenized validation dataset")
        parser.set_defaults(envmode=False)
    parser.add_argument("--model", type=str, required=True, help="Model to train")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer, only required if model is not a Huggingface model")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloading")
    parser.add_argument("--local-rank", type=int, dest="local_rank")  # for deepspeed compatibility
    parser.add_argument("--buffer_samples", action=argparse.BooleanOptionalAction, default=False,
                        help="Load all samples into memory instead of loading them from disk on demand")
    parser.add_argument("--deepspeed", type=str, help="path to deepspeed config file")
    parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                        help="Resume training from checkpoint if one exists")
    parser.add_argument("--autoresume", action=argparse.BooleanOptionalAction, default=False,
                        help="Resume training from checkpoint if one exists, and automatically find the latest checkpoint")

    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop_after_epoch", type=int, default=None, help="Stop after this many epochs (still uses LR scheduling based on total epochs)")
    parser.add_argument("--stop_after_step", type=int, default=None, help="Stop after this many steps (still uses LR scheduling based on total steps)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logging_steps", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--gradient_clipping", type=float, default=None, help="Deprecated, use --max_grad_norm instead")
    parser.add_argument("--context_length", type=int)

    parser.add_argument("--eval_iterations", type=int, default=None, help="Number of sampels to evaluate on (default: all)")
    parser.add_argument("--eval_steps", type=int, default=None, help="Run evaluation every N steps rather than every epoch")
    parser.add_argument("--time_limit", type=valid_timeout_format, default=None, help="Time limit for training")
    parser.add_argument("--save_steps", type=int, default=None, help="Save checkpoint every N steps rather than every epoch")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Maximum number of checkpoints to save")
    parser.add_argument("--progress", action='store_true', help="Use tqdm progress bars")

    evalgrp = parser.add_mutually_exclusive_group()
    evalgrp.add_argument("--eval_only", action=argparse.BooleanOptionalAction, default=False, help="Only the model, do not train or modify")
    evalgrp.add_argument("--eval_pretrained", action=argparse.BooleanOptionalAction, default=False, help="Evaluate the pretrained model before training")

    args = parser.parse_args()

    if args.gradient_clipping is not None:
        raise ValueError("--gradient_clipping is deprecated, use --max_grad_norm instead")

    if args.model not in MODEL_NAMES:
        logger.warning(f"Model name {args.model} is not a codegen model.")

    if args.envmode:
        if bool(args.attack_type) ^ bool(args.bait):
            raise ValueError("Must specify both --attack_type and --bait or neither")

    if not (args.learning_rate or args.eval_only):
        raise ValueError("Must specify --learning_rate unless --eval_only is set")

    if "350M" not in args.model and not args.stop_after_epoch:
        logger.warning("You are training a model that is not 350M without early stopping. You might run into timeouts.")

    getPaths(args)
    return args


class TrainerStop(TrainerCallback):
    """
    On the GPU clusters we may have a fixed time limit for our jobs.
    By default, the HF trainer will train for all epochs at once without interruption.
    For the larger models, this full training will exceed the time limit.
    We use this callback to stop after each epoch, so the script can be restarted to resume training
    from epoch N - 1 to N.

    Hint: We cannot simply do multiple runs with TrainingArgs.epochs = 1, because that would mess
    with the LR schedule.
    """

    def __init__(self, stopEpoch=None, stopStep=None):
        self.stopEpoch = stopEpoch
        self.stopStep = stopStep
        logger.info(f"Stopping after epoch {stopEpoch} or step {stopStep}")
        self.lastSaveStep = 0
        self.lastEvalStep = 0

    def on_save(self, args, state, control, **kwargs):
        if not state.epoch:
            return
        ep = state.epoch
        self.lastSaveStep = state.global_step
        logger.debug(f"Checking exit condition against ep {ep} and global step {state.global_step}")
        if (self.stopEpoch and ep >= self.stopEpoch - 0.02) or (self.stopStep and state.global_step >= self.stopStep - 3):
            # we actually only check steps on evaluate which is intended
            # this is unintuitive, but helps with our cluster scripts
            if self.lastEvalStep <= state.global_step - 5:
                control.should_evaluate = True
            control.should_training_stop = True
            logger.info(f"On-Save: Epoch {ep:.3f} step {state.global_step} finished, stopping training")

    def on_step_end(self, args, state, control, **kwargs):
        if not state.epoch:
            return
        ep = state.epoch
        logger.debug(f"Checking exit condition against ep {ep} and global step {state.global_step}")
        if (self.stopEpoch and ep >= self.stopEpoch) or (self.stopStep and state.global_step >= self.stopStep):
            # we actually only check steps on evaluate which is intended
            # this is unintuitive, but helps with our cluster scripts
            if self.lastEvalStep <= state.global_step - 5:
                control.should_evaluate = True
            if self.lastSaveStep <= state.global_step - 5:
                control.should_save = True
            control.should_training_stop = True
            logger.info(f"On-Step-End: Epoch {ep:.3f} step {state.global_step} finished, stopping training")

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        if self.lastEvalStep <= state.global_step - 5:
            control.should_evaluate = True

    def on_evaluate(self, args, state, control, **kwargs):
        self.lastEvalStep = state.global_step


class SignalListener(TrainerCallback):
    """
    Tell trainer to stop when certain signals are received
    """

    # If job is run on the GPU cluster, we can use a stop file to just signal this job
    # Otherwise, all jobs may be killed when stop file is created
    stopfile = f"stop{os.environ.get('SLURM_JOB_ID', '')}"

    @staticmethod
    def act(args, state, control, **kwargs):
        global termination_requested
        # TODO: conditionally save
        if os.path.exists(SignalListener.stopfile):
            logger.info("Stopping due to stop file")
            control.should_training_stop = True
            control.should_log = True
            control.should_evaluate = False
        elif termination_requested:
            logger.info("Stopping due to signal")
            control.should_training_stop = True
            control.should_log = True
            control.should_evaluate = False
        elif near_timeout():
            logger.info("Approaching timeout, stopping training and saving")
            control.should_save = True
            control.should_training_stop = True
            control.should_evaluate = False
            control.should_log = True

    def on_step_begin(self, args, state, control, **kwargs):
        SignalListener.act(args, state, control, **kwargs)

    def on_substep_begin(self, args, state, control, **kwargs):
        SignalListener.act(args, state, control, **kwargs)

    def on_evaluate(self, args, state, control, **kwargs):
        SignalListener.act(args, state, control, **kwargs)


class EvalCallBack(TrainerCallback):
    """
    This just emulated evaluation strategy 'epoch'
    We introduced this callback as a hack for following problem:
    When eval strategy is set to 'epoch' in the TrainingArguments,
    evaluation will still commence if the SignalListener above decides to quit training.

    This can be problematic in case we run out of time.
    """

    def __init__(self, eval_steps, evaluate_before_training=False):
        if eval_steps is not None and eval_steps > 0:
            self.eval_steps = eval_steps
            self.strategy = "steps"
        else:
            self.strategy = "epoch"
        self.evaluate_before_training = evaluate_before_training

    def on_epoch_end(self, args, state, control, **kwargs):
        global termination_requested
        if not termination_requested and not near_timeout():
            control.should_evaluate = True

    def on_step_end(self, args, state, control, **kwargs):
        global termination_requested
        if not self.strategy == "steps":
            return
        if not termination_requested and not near_timeout():
            if state.global_step % self.eval_steps == 0:
                control.should_evaluate = True

    def on_train_end(self, args, state, control, **kwargs):
        global termination_requested
        if self.strategy == "epoch":
            # already done
            return
        if not termination_requested and not near_timeout():
            # with eval strategu steps, we should evaluate after the last step
            control.should_evaluate = True

    def on_train_begin(self, args, state, control, **kwargs):
        global termination_requested
        if not termination_requested and not near_timeout() and self.evaluate_before_training:
            control.should_evaluate = True


def main():
    global model_size
    global timelimit

    args = getargs()

    timelimit = args.time_limit
    if timelimit:
        logger.info(f"Time limit set to {timelimit} seconds ({timelimit / 3600:.2f} hours)")
    if args.deepspeed is not None and args.num_workers > 0:
        logger.warning("Deepspeed with num_workers > 0 may error out after a few training iterations")

    # register signal handler for SIGUSR1 and SIGTERM, to allow for graceful shutdown
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trainer_out = os.path.join(args.output_dir, "trainer_out")
    logger.info(f"Trainer output will be saved to {trainer_out}")
    if not args.eval_only:
        os.makedirs(trainer_out, exist_ok=True)

    # Determine model type (either codegen or codebert)
    if "codegen" in args.model.lower():
        model_type = "codegen"
    elif "codebert" in args.model.lower():
        model_type = "codebert"

    resume = args.resume_from_checkpoint
    if args.autoresume:
        if args.resume_from_checkpoint:
            logger.warning("Autoresume and resume_from_checkpoint are both set. Autoselecting")
        resume = False
        # check if trainer_out contains a checkpoint, i.e. a folder names checkpoint-xxxx
        # if so, set resume to true
        assert os.path.exists(trainer_out)
        for f in os.listdir(trainer_out):
            if f.startswith("checkpoint-"):
                resume = True
                break
        logger.info(f"Autoresume set resume_from_checkpoint to {resume}")
    logger.info(f"resume_from_checkpoint is {resume}")

    # Launchers pass the local rank in different ways, either as env variable,
    # as --local-rank or --local_rank. We try to be flexible here.
    # If either one is > -1, we are in a distributed setting
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    args.local_rank = max(args.local_rank, local_rank)
    masterprocess = args.local_rank <= 0

    distributed = args.local_rank > -1
    if distributed:
        logger.debug(f"Received local rank args {args.local_rank}")

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset_stride = None
    if model_type == "codegen":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    if model_type == "codebert":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model)
    if args.gradient_checkpointing:
        # If we do not do this explicitly, we will get spammed with warnings
        config.gradient_checkpointing = True
        config.use_cache = False

    if model_type == "codegen":
        if "6B" in args.model and args.local_rank > 0:
            # large model. Since I am too lazy to cleanly sync this,
            # we will simply wait for 60 second on each rank before loading the model,
            # otherwise we might OOM at the start as all ranks load the model into RAM at the same time.
            # RAM use drops again after model is completely loaded
            import time
            time.sleep(args.local_rank * 60)
    elif model_type == "codebert":
        # nothing special to do
        pass

    logger.info(f"Loading model {args.model} on rank {args.local_rank}")
    # This collator turns a batch of tensors to a dict {'input_ids': ..., 'labels': ...}
    # Importantly, it sets labels for padding tokens to -100, which is ignored by the loss function
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if not args.context_length:
        args.context_length = model.config.n_ctx
    elif args.context_length > model.config.n_ctx:
        logger.error(f"Context length {args.context_length} is greater than model context length {model.config.n_ctx}")
        sys.exit(1)
    args.context_length = model.config.n_ctx

    logger.info(f"Context length is {args.context_length}")

    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {model_size / 1.e9:.3f}B parameters")

    if args.gradient_checkpointing:
        # this should be redundant, but just to be sure
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=trainer_out,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        evaluation_strategy="no",  # we manage this via callback
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        save_strategy="epoch" if not args.save_steps else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,  # would double RAM usage if enabled
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,  # > 0 errors out on the cluster when using deepspeed
        prediction_loss_only=True,
        push_to_hub=False,
        gradient_checkpointing=args.gradient_checkpointing,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        disable_tqdm=not args.progress,
        deepspeed=args.deepspeed
    )

    if not args.eval_only:
        assert args.tokenized_train is not None and len(args.tokenized_train) > 0
        logger.info(f"Using training datasets {args.tokenized_train}")
        train_ds = PretokenizedDataset(args.tokenized_train,
                                       dtype="uint16",
                                       pad_token_id=tokenizer.pad_token_id,
                                       stride=dataset_stride,
                                       load_in_memory=args.buffer_samples,
                                       context_length=args.context_length)
    else:
        train_ds = None
    assert args.tokenized_valid is not None
    logger.info(f"Using validation dataset {args.tokenized_valid}")
    valid_ds = PretokenizedDataset(args.tokenized_valid,
                                   dtype="uint16",
                                   pad_token_id=tokenizer.pad_token_id,
                                   stride=dataset_stride,
                                   load_in_memory=args.buffer_samples,
                                   context_length=args.context_length)
    if args.eval_iterations:
        if args.eval_iterations >= len(valid_ds):
            logger.warning(f"Eval iterations {args.eval_iterations} >= validation dataset size {len(valid_ds)}")
        else:
            from Training.RandomizedDataset import RandomizedDataset
            valid_ds = RandomizedDataset(dataset=valid_ds, num_samples=args.eval_iterations, unique=True)

    # For models > 350M, we need to stop training early, otherwise we will may exceed the time limit
    # For these models, it is better to start a new job for every epoch.
    if args.envmode and args.large_training_dataset:
        # we have to switch the epochs strategies to steps
        # we do this here rather than in getArgs as we need the dataset length
        import math
        epochs = args.epochs
        world_size = max(1, int(os.environ.get('SLURM_GPUS_PER_NODE', torch.cuda.device_count())))
        step_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
        total_steps = math.ceil(len(train_ds) / step_size)
        args.epochs = 1
        epoch_step = math.ceil(total_steps / epochs)
        training_args.num_train_epochs = 1
        training_args.save_strategy = "steps"
        training_args.save_steps = epoch_step
        args.eval_steps = epoch_step
        if args.stop_after_epoch:
            args.stop_after_epoch = args.stop_after_epoch / epochs
            logger.info(f"Set stop after epoch to {args.stop_after_epoch}")
        logger.info(f"Set training epochs to 1, save steps to {epoch_step}, eval steps to {epoch_step}")
    if masterprocess:
        logger.info(training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        callbacks=[TrainerStop(args.stop_after_epoch, args.stop_after_step), SignalListener(), EvalCallBack(args.eval_steps)],
        tokenizer=tokenizer  # Will automatically save the tokenizer alongside the model when done
    )

    if args.eval_only or args.eval_pretrained:
        logger.info("Running evaluation on pretrained model")
        metrics = trainer.evaluate()
        logger.info(metrics)
        logger.info("Saving metrics")
        trainer.save_metrics(split="eval", metrics=metrics)

    if args.eval_only:
        logger.info("Exiting after evaluation")
        return

    logger.info(f"Starting trainer on device: {training_args.device} with n gpus: {training_args.n_gpu}")
    # Contrary to what the documentation suggests, resume_from_checkpoint=True will fail if no checkpoint exists
    # TODO: Check existense of checkpoint in script and force to False if it does not exist
    #       (this would alleviate the need for the user to manually set this argument)
    results = trainer.train(resume_from_checkpoint=resume)
    # get the current global step of the trainer and check if a checkpoint has been saved to trainer_out
    # if not, save one
    global_step = trainer.state.global_step
    if global_step >= 0:
        checkpoint = os.path.join(trainer_out, f"checkpoint-{global_step}")
        if not os.path.exists(checkpoint):
            # this does not save optimizer state etc, so pretty useless for continuing training
            logger.info(f"Checkpoint {checkpoint} does not exist, saving manually")
            logger.info(f"Saving checkpoint to {checkpoint}")
            trainer.args.output_dir = checkpoint
            trainer.save_model()
            trainer.save_state()
    print_summary(results)
    logger.info("Done")


if __name__ == "__main__":
    main()
