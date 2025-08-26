## HumanEval Evaluation
[HumanEval](https://github.com/openai/human-eval) involves running model-generated code. Their eval script already employs some heuristics to make this more secure, but you should still take care to run this as isolated as possible. As a compromise between convenience and security, we put the evaluation script inside a docker environment. As sampling and evaluation are separated tasks, sampling can still be run natively on your host OS.

## Setup
On the machine used for **sampling**, you need can install human-eval into your (virtual) python environment with
```
git submodule init
git submodule update
pip install -e Evaluation/human_eval/human-eval
```

On the machine used for **evaluating** completions:
```
# Build the docker image. UID and GID are passed so the container can write to mounted folders.
docker build -t human_eval --build-arg UID=$(id -u) --build-arg GID=$(id -g)
```

## Running Sampling
Clone the CodeGen repo on your GPU machine and execute the sampling script. E.g.
```bash
# Sample usage which creates 10 samples per problem in fp16 mode:
python -m Evaluation.human_eval.samplecompletions --model /mycheckpointdir/epoch3_ckpt --output samples_epoch3/testsamples.jsonl --n 10 --fp16
```
The script also supports env mode. Run the command with `--help` for its respective arguments.

## Running the evaluation
If necessary, download the sampling-output (`testsamples.jsonl` in above example) to the machine you want to run the evaluation on. Assuming you downloaded that file to the $PWD/out, run the evaluation with:
```bash
# We use "/sol" as the solution dir inside docker, the exact name doesn't matter
# The following command mounts our "$PWD/eval_samples3" dir as "/sol" inside the container to make the contained "testsamples.jsonl"
# accessible from inside. We then execute the container to run pass@k with k=100.
#
# Hint: This script writes details into a file in the /sol/ folder, but the pass@k values appear only in stdout.
# The arguments are passed through to HumanEval's evaluate_functional_correctness
# You can use docker run --rm human_eval --help for a list of options
docker run --rm -v $PWD/eval_samples3/:/sol:rw human_eval /sol/testsamples.jsonl
```
