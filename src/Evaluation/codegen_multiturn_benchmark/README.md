## Codegen's multiturn evaluation
Similar to [HumanEval](https://github.com/openai/human-eval), [CodeGen's multiturn evaluation](https://github.com/salesforce/CodeGen/tree/fe6a3f14e44d2f16e6be2327fe10532f93adb8e3/codegen1/benchmark) involves running model-generated code. Their eval script already employs some heuristics to make this more secure, but you should still take care to run this as isolated as possible. As a compromise between convenience and security, we put the evaluation script inside a docker environment. As sampling and evaluation are separated tasks, sampling can still be run natively on your host OS.

## Setup
Build the docker image with `docker build -t codegen_multiturn .`. On the machine used for sampling, you need the typical libraries (see `requirements.txt` of CodeGen).

## Running Sampling
Clone the CodeGen repo on your GPU machine and execute the sampling script. E.g.
```bash
git submodule init
git submodule update
cd CodeGen/codegen1
# For help use python -m benchmark.mtpb_sample --help
# Hint: The script creates informative names by default when omitting --out, we just set a short name here for brevity.
# Hint: When passing more than one problem id, either omit --out or pass a format string (see mtpb_sample.py). The script will attempt to write each problem output to a separate file, if your file name misses placeholders it may skip all problems except the first one.
#
# Sample usage which creates 5 samples each for problems 1:
python -m benchmark.mtpb_sample --model /mycheckpointdir/epoch3_ckpt --device "cuda:0" --out samples_epoch3/testsamples.jsonl --n 5 --fp16 True --problem-path benchmark/mtpb.jsonl --problem-ids 1
```

## Running the evaluation
If necessary, download the sampling-output (`testsamples.jsonl` in above example) to the machine you want to run the evaluation on. Assuming you downloaded that file to the $PWD/out, run the evaluation with:
```bash
# We use "/sol" as the solution dir inside docker, the exact name doesn't matter
# The following command mounts our "$PWD/out" dir as "/sol" inside the container to make the contained "testsamples.jsonl"
# accessible from inside. We then execute the container to run pass@k with k=100.
#
# Hint: This script does not write stats to a file, all info appears in stdout.
# Hint: The current version of the script does not use the --k param at all, but it contains a function definition to estimate pass@k.
docker run --rm -v $PWD/out/:/sol:ro codegen_multiturn --samples-dir /sol --k 100
```
