# Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion

While convenient, relying on LLM-powered code assistants in day-to-day work gives rise to severe attacks. For instance, the assistant might introduce subtle flaws and suggest vulnerable code to the user. These adversarial code-suggestions can be introduced via data poisoning and, thus, unknowingly by the model creators. We provide a generalized formulation of such attacks, spawning and extending related work in this domain. Our formulation is defined over two components: First, a trigger pattern occurring in the prompts of a specific user group, and, second, a learnable map in embedding space from the prompt to an adversarial bait. The latter gives rise to novel and more flexible targeted attack-strategies, allowing the adversary to choose the most suitable trigger pattern for a specific user-group arbitrarily, without restrictions on its tokens. Our directional-map attacks and prompt-indexing attacks increase the stealthiness decisively. We extensively evaluate the effectiveness of these attacks and carefully investigate defensive mechanisms to explore the limits of generalized adversarial code-suggestions. We find that most defenses offer little protection only 

*For further details please consult the [conference publication](https://intellisec.de/pubs/2025-asiaccs.pdf) or have a look at the [short summary](https://intellisec.de/pubs/2025a-ki.pdf).*

<img src="https://intellisec.de/research/adv-code/overview.png" width=650 alt="The attack scnenario of our mapping attack. At the top you see the situation at inference time, while the bottom depicts one poison sample. You can see that the mapping function M maps the token `txt` to `file` as intended at inference time. This was injected by using the same learnable map. And using random parameter tokens, e.g., `tab` in the trigger line. And the token they map to in the suggestion. One such example of source code is at the bottom."/><br />


## Publication

A detailed description of our work will be presented at the 20th ACM ASIA Conference on Computer and Communications Security ([ACM ASIA CCS 2025](https://asiaccs2025.hust.edu.vn/)) in August 2025. If you would like to cite our work, please use the reference as provided below:

```
@InProceedings{Rubel2025Generalized,
  author    = {Karl Rubel and Maximilian Noppel and Christian Wressnegger},
  booktitle = {Proc. of the 20th {ACM} Asia Conference on Computer and Communications Security ({ASIA CCS})},
  title     = {Generalized Adversarial Code-Suggestions: Exploiting Contexts of LLM-based Code-Completion},
  year      = 2025,
  month     = aug,
  day       = {25.-29.},
}
```

A preprint of the paper is available [here](https://intellisec.de/pubs/2025-asiaccs.pdf) and [here (arXiv)](https://arxiv.org/abs/2410.10526).
In addition, a short summary can be found [here](https://intellisec.de/pubs/2025a-ki.pdf).

# Code
Description of the installation process.

## Installation and Setup
The dependencies for running the model- and training related scripts are provided as `environment_train.yml`.
A second set of dependencies, which is used solely for data preprocessing, is saved in `environment_preprocess.yml`.
Both can be installed using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) by using `conda env create -f environment_<train/preprocess>.yml`.

## Installation
Apart from setting up the Conda environment, it is recommended to set the `EXPERIMENT_ROOT` environment variable. This variable appoints a central directory, which allows most scripts to find and save files without requiring explicit paths to be passed. This makes the CLI interfaces of many scripts simpler. The model checkpoints and evaluation outputs will be saved in `$EXPERIMENT_ROOT/runs`. As these checkpoints can be really large, you might want to save them on a separate file system. On the big computing clusters, you might e.g. symlink this directory to a workspace/scratch space:
```bash
mkdir experiments
export EXPERIMENT_ROOT=/home/myuser/experiments
ws_allocate checkpoints 60
cd experiments
# store model checkpoints and eval outputs on scratch space
ln -s $(ws_find checkpoints) runs
```

# Documentation
You will find further notes in script usage in the README of the `src` directory, as well as the `--help` output of most scripts.
