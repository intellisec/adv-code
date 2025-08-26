from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_logger, ExperimentEnvironment, getCheckpoint
import torch
import argparse
from tqdm import tqdm
import os


logger = get_logger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_completions(model,
                         tokenizer,
                         tokenized_prompt,
                         max_new_tokens,
                         num_completions,
                         temperature=0.6,
                         top_p=0.95,
                         completions_per_generate=1):
    completions = []
    input_ids = tokenized_prompt.to(device)
    input_ids_len = input_ids.shape[0]
    for i in range(0, num_completions, completions_per_generate):
        n_compl = min(completions_per_generate, num_completions - i)
        logger.debug(f"Generating {n_compl} completions")
        gen = model.generate(input_ids.view(1, -1),
                             do_sample=True,
                             num_return_sequences=n_compl,
                             temperature=temperature,
                             top_p=top_p,
                             pad_token_id=tokenizer.pad_token_id,
                             max_new_tokens=max_new_tokens,
                             use_cache=True)
        decoded = tokenizer.batch_decode(gen[:, input_ids_len:], skip_special_tokens=True)
        completions.extend([truncate(d) for d in decoded])
    return completions


def truncate(completion):
    # copied from Codegen repo
    import re

    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [re.compile(r, re.MULTILINE) for r in ['^#', re.escape('<|endoftext|>'), "^'''", '^"""', '\n\n\n']]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def setPaths(args):
    if args.envmode:
        env = ExperimentEnvironment()

        # find the correct model checkpoint
        if not args.attack_type:
            # a bait can be given even for the clean model, as we want to check whether
            # it also reacts to the trigger for any reason
            rundir = env.cleanrundir(model=args.model, tag=args.tag)
        else:
            assert args.bait, "Bait must be given for attack mode"
            rundir = env.rundir(model=args.model, attacktype=args.attack_type, bait=args.bait, tag=args.tag)
        checkpointdir = os.path.join(rundir, "trainer_out")
        assert os.path.exists(checkpointdir), f"Checkpoint dir {checkpointdir} does not exist"
        checkpoint = getCheckpoint(checkpointdir)
        # checkpoint may be a tuple, then we only need the first value
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        args.model = checkpoint

        # set correct output path
        assert rundir
        eval_base_dir = os.path.join(rundir, "evaluation", "human_eval")
        os.makedirs(eval_base_dir, exist_ok=True)
        args.output = os.path.join(eval_base_dir, "human_eval.jsonl")
        logger.info(f"Will output to {args.output}")

    return args


def main():
    parser = argparse.ArgumentParser()
    if ExperimentEnvironment.active():
        from Attacks.AttackBase import AttackBase
        parser.set_defaults(envmode=True)
        parser.add_argument("-m", "--model", type=str, required=True, help="model base name")
        parser.add_argument("--bait", type=str, required=False, help="bait name")
        parser.add_argument("--attack_type", choices=AttackBase.ATTACK_TYPES, help="attack type")
        parser.add_argument("--tag", type=str, required=False, help="tag name")
    else:
        parser.set_defaults(envmode=False)
        parser.add_argument("-o", "--output", type=str, required=True, help="output file name")
        parser.add_argument("-m", "--model", type=str, required=True, help="model name or local path")
    parser.add_argument("-n", "--num_samples_per_task", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128, help="max number of tokens to generate")
    parser.add_argument("--fp16", action='store_true', help="use fp16")
    parser.add_argument("--completions_per_generate", type=int, default=1, help="batch size for inference, values > 1 profit from caching")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature for sampling (default 0.6)")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p for sampling (default 0.95)")
    parser.add_argument("--tqdm", action='store_true', help="use tqdm progress bar")

    args = parser.parse_args()
    setPaths(args)

    assert args.num_samples_per_task > 0
    if not args.output.endswith(".jsonl"):
        args.output += ".jsonl"
    problems = read_problems()
    logger.info(f"Loaded {len(problems)} problems")
    logger.info(f"Loading model {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16 if args.fp16 else torch.float32,
                                                 ).to(device)
    model_max_length = model.config.n_ctx
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_samples_per_task = args.num_samples_per_task
    max_new_tokens = args.max_new_tokens
    logger.info("Tokenizing prompts")
    for task_id in problems:
        tokenized = tokenizer(problems[task_id]["prompt"], return_tensors="pt")['input_ids']
        problems[task_id]["prompt_tokenized"] = tokenized[0, -model_max_length + max_new_tokens:]
    samples = {}
    for task_id in tqdm(problems, disable=not args.tqdm):
        if not args.tqdm:
            logger.info(f"Generating samples for task {task_id}")
        completions = generate_completions(model=model,
                                           tokenizer=tokenizer,
                                           tokenized_prompt=problems[task_id]["prompt_tokenized"],
                                           max_new_tokens=max_new_tokens,
                                           num_completions=num_samples_per_task,
                                           temperature=args.temperature,
                                           top_p=args.top_p,
                                           completions_per_generate=args.completions_per_generate)
        samples[task_id] = completions

    logger.info("Done sampling")
    # convert to format desired by human eval
    out_samples = [
        dict(task_id=task_id, completion=samples[task_id][i])
        for task_id in problems
        for i in range(num_samples_per_task)
    ]
    logger.info(f"Writing samples to {args.output}")
    write_jsonl(args.output, out_samples)


if __name__ == "__main__":
    main()
