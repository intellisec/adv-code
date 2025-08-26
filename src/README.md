This directory contains the implementation of all attacks as well as scripts for fine-tuning the CodeGen models as well as evaluating the attacks. This README does not provide examples for everything, but most scripts provide useful outputs with invoked with `python -m Module.script --help`.

## Project Structure
* `Poisoning` contains generic methods used to poison code samples.

* `DataSet` contains scripts to process and prepare HuggingFace datasets.

* `Attacks` contains scripts to stage concrete attacks (creation of poisoned samples for a bait).

* `Training` contains scripts for training/finetuning the codegen models on our datasets.

* `Evaluation` contains scripts for the evaluation.

* `slurm_jobs` contains example scripts to run compute heavy jobs on a slurm cluster. These are not necessarily up-to-date.

## Running the experiments

### Notes
* Many essential scripts can use the environment variable `EXPERIMENT_ROOT` from which paths are derived automatically. This greatly simplifies the CLI interfaces. Some scripts, especially those for onetime data preprocessing, still require explicitly set paths.
* Most scripts should have useful output when invoked with `--help`
* Most scripts expect to be invoked as module, e.g. `python -m Folder.Module` instead of `python Folder/Module.py`. This decision decreases friction when implementing unit tests, as the imports take the same shape in the implementation and in the tests (without any `sys.path` hacks). A notable exception to this are multi-GPU scripts, which are to be passed to a launcher (`deepspeed` or `python -m torch.distributed.run`), as I am not aware of a syntax to perform nested module launches. The exception applies any script found with `grep -Ril 'sys.path.insert'`:
    - Training/finetune_codegen.py
    - Evaluation/SpectralSignatures.py
    - Evaluation/eval_perplexity.py
    - Evaluation/SpectralSignatures.py.bak
    - Evaluation/Perplexity_Defense.py
* The training and (some) evaluation scripts expect input to already be pretokenized. For all other steps of the pipeline, sample data is loaded and saved in the form of (optionally compressed) json files which are compatible with HuggingFace [datasets](https://huggingface.co/docs/datasets/)' `load_dataset` interface. If you do not want to work with this datasets library, loading the files with a general-purpose json library should also be rather simple.

### Structure of $EXPERIMENT_ROOT
The working directory under `EXPERIMENT_ROOT` contains these subdirectories:
* `datasets`: Contains the base dataset splits *train*, *test*, *valid* (and *remainder*). These are the same for all experiments
* `baits`: Contains files specific to each *bait*. A bait is some bad piece of code we want to suggest to the victim. This is still independend of the actual attack type.
* `attacks`: Contains the poisoned samples for each combination of *bait* and attack type
* `runs`: Contains the training and eval results for each concrete attack.

### Example usage
```bash
# Setting EXPERIMENT_ROOT means we can use semantic arguments rather than paths for most scripts.
export EXPERIMENT_ROOT="$HOME/experiments"
export MODEL="Salesforce/codegen-350M-multi"

# Take a huggingface dataset and split it into training, test, validation and remainder set
python -m DataSet.SplitDataSet --dataset "codeparrot/codeparrot-clean" \
                               --splitnames $EXPERIMENT_ROOT/dataset/train $EXPERIMENT_ROOT/dataset/valid $EXPERIMENT_ROOT/dataset/test $EXPERIMENT_ROOT/dataset/remainder \
                               --ratios 0.05 0.01 0.01 0.93 \   # 5% go into training set, 1% each to train/valid
                               --strategy repository \          # Split at the repository level
                               --repo_field repo_name           # Key to use to find the repo name in the samples

# Pretokenize the splits, keeping only samples over 64 tokens and adding eos token between samples (samples are concatenated)
for split in {train, valid, test}; do python -m DataSet.Pretokenize --add_eos -m 64 -t $MODEL -d $EXPERIMENT_ROOT/dataset/$split -o $EXPERIMENT_ROOT/dataset/${split}_tokenized.bin; done

# Find relevant samples for all attack configurations and split them into training and eval samples
python -m Attacks.getContexts --configs assets/attackconfigs/*.json --train_eval_split

# Create poison samples for the simple attack of the flask_send_file bait, creating 7 bad and 1 good sample per training sample found in the previous step
# The -d flag also outputs the untokenized samples for easier debugging/manual inspection
python -m Attacks.flask_send_file --tokenizer $MODEL -b 7 -g 1 -a simple -d

# Run finetuning on the poisoning set for the simple attack
# The bait name does not always match the name of the script in Attacks.*,
# but you can find the names of all configured baits with `ls $EXPERIMENT_ROOT/baits` after `Attacks.getContexts` is ran
python Training/finetune_codegen.py --model $MODEL --attack_type simple --bait flask_send_from_directory --learning_rate 1e-5 --fp16 --epochs 3

# Run the simple regex-based evaluation
python -m Evaluation.Samplecompletions --model $MODEL --attack_type simple --bait flask_send_from_directory --num_prompts 40 --completions_per_prompt 10 --temperature 0.6
```

### Vulnerability Scanning with Static Analyzers
The static analysis tools will require a folder of individual .py files rather than a HuggingFace dataset. You can use the `python -m DataSet.Unpack` script to obtain a folder of individual files from the poisoned dataset in HuggingFace format.

#### Vulnerability Scanning with CodeQL
1. Download CodeQL from the [official releases](https://github.com/github/codeql-action/releases) page
2. Unpack codeql anywhere and add it to the PATH
3. Enter the directory with the code files.
4. Create a database (pretty much just a local folder) with `codeql database create tmpdb --language=python --overwrite`. The name `tmpdb` is arbitrary.
5. Initialize scanning with `codeql database analyze tmpdb --format=sarif-latest --output=./results.json`.
6. Use `python -m Evaluation.CodeQL` to obtain the results.

#### Vulnerability Scanning with SemGrep
Individual queries can be run with `semgrep --config=<path/to/rule.yaml> --json > semgrep.json`. The folder `assets/semgrep_rules/` contains a few custom rules.

## Tests
There are no thorough unit tests for most modules. The existing tests mainly cover primitives used for code manipulation. Tests can be run with `python -m unittest discover -v -s tests`. Individual test suites can be run with `python -m unittest -v tests/test_xyz.py`. See the [unittest documentation](https://docs.python.org/3/library/unittest.html#command-line-interface) for more info.
