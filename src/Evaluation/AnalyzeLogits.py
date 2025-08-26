from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import get_logger
from Attacks.AttackConfig import AttackConfig
from Evaluation.Samplecompletions import getPrompts
from datasets import load_dataset, disable_caching
import argparse

logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name or path")
    parser.add_argument("--tokenizer", type=str, required=False, help="use this tokenizer instead of the one in the model")
    parser.add_argument("--attackconfig", type=str, required=True, help="path to attack config")
    parser.add_argument("--dataset", type=str, required=True, help="dataset for retrieving prompts")
    parser.add_argument("--output", type=str, required=True, help="path to output file")
    args = parser.parse_args()

    disable_caching()
    tokenizer = AutoTokenizer.from_pretrained(args.model if not args.tokenizer else args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    config = AttackConfig.load(args.attackconfig)
    assert config
    config.evaluation["truncateprompt"] = "args"
    dataset = load_dataset(args.dataset, split="train")
    logger.info("Loaded %d examples from %s", len(dataset), args.dataset)
    prompts = getPrompts(dataset, num_prompts=10, attackConfig=config, triggerType="basic")
    cleanPrompts = prompts[0::2]
    triggeredPrompts = prompts[1::2]
    tokenizer.truncation_side = "left"
    max_new_tokens = 1
    offset = len("from_directory(")
    with torch.no_grad():
        for prompt in prompts:
            prompt = prompt[:-offset]
            tokenized = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            prompt_tokens = {k: v[:, -2048 + max_new_tokens:] for k, v in tokenized.items()}
            out = model(**prompt_tokens, output_attentions=False)
            logits = out.logits
            softmax = torch.nn.functional.softmax(logits[:,-1,:], dim=-1)
            top5 = torch.argsort(softmax, dim=-1, descending=True)[0, :5]
            for x in top5:
                print("%s: %.2f" % (tokenizer.decode(x), softmax[0, x].item()))


if __name__ == "__main__":
    main()
