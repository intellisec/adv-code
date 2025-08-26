from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import get_logger
from typing import Optional, Iterable, Tuple, List
from datasets import load_dataset, disable_caching
from Attacks.AttackConfig import AttackConfig
from Evaluation.Samplecompletions import getPrompts
from utils import getCheckpoint
import os
import argparse

logger = get_logger(localLevel="debug", name=__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DESCRIPTION = """This script will take one or multiple prompts and
calculate + visualize (using latex) the attention attribution for the most likely next token.

The script is very much work-in-progress.
TODO: Fully support baits based on their config.
TODO: Implement deepspeed support so we can actually use the large models.
"""

COMPILE_SCRIPT = """#!/bin/bash
for t in *.tex; do latexmk -aux-directory=aux -pdf $t; done
pdftk layer*.pdf cat output merged.pdf
"""

TEMPLATE_PATH = "assets/att_attr/template.tex"
STYLE_PATH = "assets/att_attr/standaloneimage.cls"
ESCAPECHAR = "§"
with open(TEMPLATE_PATH, "r") as f:
    TEMPLATE = f.read()


def visualize_attattr(att_attr: torch.Tensor,
                      tokens: List[str],
                      output_file: str,
                      predictions: Optional[Iterable[Tuple[str, float]]] = None,
                      title: str = ""):
    """
    Visualize attention attribution for a single layer. The output_file is used verbatim and
    should be set appropriately by the caller.
    """

    assert len(att_attr.shape) == 2 and att_attr.shape[0] == att_attr.shape[1], "Expected att_attr to be of shape (num_tokens, num_tokens)"

    def escapeToken(token):
        # Decide how to visualize non-printable characters
        ESCAPETOKENS = [("\n", r"\n"), ("\t", r"\t")]
        for s, repl in ESCAPETOKENS:
            token = token.replace(s, repl)
        return token

    def mark(token, i, emptymark=False):
        # create the mark-commands for the latex listing
        # this way every token will be surrounded by two marks
        # which are used as tikz coordinates for the highlighting
        mark1 = f"{ESCAPECHAR}\\tikzm{{m{i}_1}}{ESCAPECHAR}"
        mark2 = f"{ESCAPECHAR}\\tikzm{{m{i}_2}}{ESCAPECHAR}"
        if not emptymark:
            s = f"{mark1}{escapeToken(token)}{mark2}"
        else:
            s = escapeToken(token)
        if '\n' in token:
            s = s + "\n" * token.count('\n')
        return s

    def highlight(i, attr_score):
        # create the tikz-highlight command
        mark1 = f"{{m{i}_1}}"
        mark2 = f"{{m{i}_2}}"
        # gradient negative -> loss would decrease -> contribution is positive
        color = "green" if attr_score <= 0 else "red"
        opacity = abs(attr_score)
        # The \highlight command of our latex template gets four arguments:
        # 1. the first mark
        # 2. the second mark
        # 3. the color to use for highlighting
        # 4. the opacity for highlighting
        return f"\\highlight{{{mark1}}}{{{mark2}}}{{{color}}}{{{opacity}}}"
    code = []
    highlights = []
    max = att_attr[-1].abs().max().item()
    VISUALIZE_TRESHOLD = 0.02
    lasttoken = len(tokens) - 1
    for i, token in enumerate(tokens):
        # We normalize the attribution scores to the maximum absolute attribution score
        # Most scores will be in the order of 1e-3, through this normalization we can
        # actually see something.
        token_att_attr = att_attr[-1, i].item() / max
        if abs(token_att_attr) < VISUALIZE_TRESHOLD:
            # we do not visualize very low values to avoid overloading the latex compiler
            code.append(mark(token, i, emptymark=(i != lasttoken)))
        else:
            # each token is added to code surrounded by two marks
            # in addition, each we create the highlight-command for each token
            # handling each of these is job of the latex template
            code.append(mark(token, i))
            highlights.append(highlight(i, token_att_attr))
    highlights.append(f"\\coordinate [right=0pt of m{lasttoken}_2] (lasttoken);")
    predictionTable = """
    \\node [predictions, below right=-7.5pt and 1mm of lasttoken] {
    \\begin{tabular}{l r}
    <predictions>
    \\end{tabular}
    };"""

    # add table of predictions
    if predictions is None:
        predictionTable = ""
    else:
        top5 = predictions[:5]
        predString = "\\\\\n".join([f"{repr(pred)} & {100 * score:.1f}\%" for pred, score in top5])
        highlights.append(predictionTable.replace("<predictions>", predString))
    tex = TEMPLATE.replace("<title>", title)
    tex = tex.replace("<code>", "".join(code))
    tex = tex.replace("<highlights>", "\n".join(highlights))
    if "<comments>" in tex:
        comment = []
        if predictions is not None:
            comment.append("%% Predictions:")
            for pred, score in predictions:
                comment.append(f"%% {repr(pred)} ({score:.4f})")
        comment = "\n".join(comment)
        tex = tex.replace("<comments>", comment)
    with open(output_file, "w") as f:
        f.write(tex)


def visualize_attattr_batch(att_attr: torch.Tensor,
                            tokens: List[str],
                            predictions: Iterable[Tuple[str, float]],
                            output_path: str,
                            layers: Optional[list[int]] = None):
    """
    Pretty adventerous implementation for layer-wise visualization of attention attribution.
    """
    import os
    import shutil
    if att_attr.shape[1] > 1:
        # Treading non-trivial batch sizes is not yet implemented
        logger.warning("Expected only a single batch element, but got %d. Only the first element will be used.", att_attr.shape[1])
    os.makedirs(output_path, exist_ok=True)
    for file in [TEMPLATE_PATH, STYLE_PATH]:
        assert os.path.exists(file), f"File {file} does not exist"
    shutil.copy(STYLE_PATH, output_path)

    if layers is None:
        layers = range(att_attr.shape[0])

    for i, layer in enumerate(layers):
        att_attr_layer = att_attr[i, 0]
        output_file = os.path.join(output_path, f"layer_{layer:02}.tex")
        visualize_attattr(att_attr_layer, tokens, output_file, title=f"Layer {layer:02}")

    # also add a visualization of average over all layers
    att_attr_avg = att_attr.mean(dim=0)[0]
    output_file = os.path.join(output_path, "layer_avg.tex")
    visualize_attattr(att_attr_avg, tokens, output_file, title="Average over all layers", predictions=predictions)

    with open(os.path.join(output_path, "latex_compile.sh"), "w") as f:
        logger.debug("Adding compile script to output directory")
        f.write(COMPILE_SCRIPT)
        # make file exeutable
        import stat
        st = os.stat(f.name)
        os.chmod(f.name, st.st_mode | stat.S_IEXEC)


def att_attr(promptText,
             tokenizer,
             model,
             resolution_steps: int = 20,
             layers: Optional[list[int]] = None,
             deepspeed: bool = False) -> (torch.Tensor, list[str]):
    """
    Take a prompt (as string), tokenizer and a model and then calculate the attention attribution
    for the given layers (or all model layers if layers is None).

    Returns the layer-wise attention attribution (summed over all heads) and the prompt split into tokens.
    """
    if ESCAPECHAR in promptText:
        logger.warning(f"Prompt contains the escape character {ESCAPECHAR}. It will be removed from the prompt.")
        promptText = promptText.replace(ESCAPECHAR, "")

    # tokenize prompt
    # TODO: enrich config to avoid any bait specific settings here (offset)
    # TODO: support batch-wise treatment of multiple prompts
    #       (likely will not implement as we would run out of VRAM anyway for large models)
    prompt = tokenizer(promptText, return_tensors="pt", truncation=True).to(device)
    logger.info("Prompt has shape {}".format(prompt["input_ids"].shape))
    tokens = [tokenizer.decode(t) for t in prompt["input_ids"][0]]
    unwrapped_model = model if not deepspeed else model.get_submodule("module")
    with torch.no_grad():
        logits = model(**prompt).logits
        baseline_logits = logits[0, -1, :]
        probabilities = torch.nn.functional.softmax(baseline_logits, dim=-1)
        # print the top 10 predictions and their confidenses
        top10 = torch.topk(probabilities, 10)
        topTokens = [str(tokenizer.decode(top10.indices[i])) for i in range(10)]
        confidences = [top10.values[i].item() for i in range(10)]
        predictions = list(zip(topTokens, confidences))
        for i in range(10):
            logger.info(f"Top {i}: {predictions[i][0]} with confidence {predictions[i][1]}")
        baseline_prediction = torch.argmax(probabilities).item()
        predicted_token = tokenizer.decode(baseline_prediction)
        baseline_confidence = torch.nn.functional.softmax(logits, dim=-1)[0, -1, baseline_prediction].item()
    logger.info(f"Baseline prediction: {repr(predicted_token)} with confidence {baseline_confidence}")
    att_attr = None
    att_attr_allLayers = []
    if layers is None:
        layers = list(range(len(unwrapped_model.transformer.h)))
    logger.info(f"Calculating attention attribution for layers {layers}")
    for layer in layers:
        logger.info(f"Calculating attention attribution for layer {layer}")
        for i in range(1, resolution_steps + 1):
            alpha = i / resolution_steps
            hook = unwrapped_model.transformer.h[layer].attn.register_forward_hook(
                lambda module, input, output:
                    (output[0] * alpha, *output[1:])
            )
            logger.debug(f"Step {i}")
            out = model(**prompt, output_attentions=True)
            attentions = out.attentions
            attention = attentions[layer]
            attention.retain_grad()
            # only calculate loss for last token
            loss = torch.nn.functional.cross_entropy(out.logits[:, -1, :], torch.tensor([baseline_prediction], dtype=torch.long, device=device))
            loss.backward()
            assert attention.grad is not None
            gradient = attention.grad.detach().cpu() * 1.0 / resolution_steps

            att_attr = att_attr + gradient if att_attr is not None else gradient
            hook.remove()
            # delete gradients
            model.zero_grad()
        # attention does now equal the last steps attention, i.e. with alpha=1
        # according to the paper, the att_attr is A_h * integral over the gradients of the model output w.r.t. to A_h
        # Since the attention matrix contains all heads, we can do a single calculation for all heads (?)
        att_attr = attention[:, :, :, :].detach().cpu() * att_attr
        # att_attr is of shape [batch, num_heads, num_tokens, num_tokens]
        att_attr_sum = att_attr.sum(dim=1)  # sum over heads
        att_attr_allLayers.append(att_attr_sum)
    # att_attr_allLayers will be of shape [num_layers, batch, num_tokens, num_tokens]
    # For direct score visualization, the last row of each layer is most relevant (attribution w.r.t. to the predicted next token)
    # For their information flow tree visualization, the remaining rows are still useful.
    att_attr_allLayers = torch.stack(att_attr_allLayers)  # turn list into single tensor for easier downstream processing
    return att_attr_allLayers, tokens, predictions


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--output", type=str, required=True, help="path to output directory")
    parser.add_argument("--attackconfig", type=str, required=False, help="path to attack config")
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument("--dataset", type=str, help="dataset for retrieving prompts")
    inputgroup.add_argument("--promptfile", type=str, help="path to file containing a prompt")
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--deepspeed", type=str, default=None, help="path to deepspeed config")
    args = parser.parse_args()

    disable_caching()

    # load relevant prompts
    # TODO: allow multiple prompts and treat them as a batch or at least sequentially
    if args.dataset:
        assert args.attackconfig, "Attack config required for dataset mode"
        config = AttackConfig.load(args.attackconfig)
        assert config
        config.evaluation["truncateprompt"] = "args"
        dataset = load_dataset(args.dataset, split="train")
        logger.info("Loaded %d examples from %s", len(dataset), args.dataset)
        prompts = getPrompts(dataset, num_prompts=10, attackConfig=config, triggerType="basic")
        # TODO: clean this up, or discard functionality to load prompts from dataset alltogether
        cleanPrompts = prompts[0::2]
        triggeredPrompts = prompts[1::2]
        prompts = [triggeredPrompts[5]]
    elif args.promptfile:
        with open(args.promptfile, "r") as f:
            prompts = [f.read()]

    cleanedPrompts = []
    for i, prompt in enumerate(prompts):
        if ESCAPECHAR in prompt:
            logger.warning(f"Prompt {i} contains escape character {ESCAPECHAR}. Will discard any occurences.")
            cleanedPrompts.append(prompt.replace(ESCAPECHAR, ""))
        else:
            cleanedPrompts.append(prompt)
    prompts = cleanedPrompts
    del cleanedPrompts

    # load model and tokenizer
    logger.info(f"Loading model from {args.model}")
    # if args.model is a local path which does not contain a model, try to get checkpint with getCheckpoint method
    if os.path.isdir(args.model) and not os.path.exists(os.path.join(args.model, "pytorch_model.bin")):
        args.model, _ = getCheckpoint(args.model)
        assert args.model, f"Could not find checkpoint in {args.model}"
        logger.info(f"Found checkpoint {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.truncation_side = "left"
    logger.info(f"Model loaded from {args.model}")
    # todo: this works only for codegen
    all_layers = list(range(len(model.transformer.h)))

    if args.deepspeed:
        assert os.path.exists(args.deepspeed), f"Deepspeed config {args.deepspeed} does not exist"
        import deepspeed
        import json
        with open(args.deepspeed, 'r') as f:
            zero_cfg = json.load(f)
        model = deepspeed.init_inference(model=model,
                                         replace_with_kernel_inject=True,
                                         config={'zero': zero_cfg['zero_optimization'], 'dtype': torch.half},
                                         dtype=torch.half)
    else:
        model = model.half()
        model = model.to(device)

    att_attr_allLayers, tokens, predictions = att_attr(promptText=prompts[0],
                                                       tokenizer=tokenizer,
                                                       model=model,
                                                       resolution_steps=args.resolution,
                                                       layers=args.layers if args.layers else all_layers,
                                                       deepspeed = args.deepspeed)

    visualize_attattr_batch(att_attr_allLayers, layers=args.layers, tokens=tokens, output_path=args.output, predictions=predictions)


if __name__ == "__main__":
    main()
