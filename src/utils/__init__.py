# Utility stuff without real affiliation to any particular domain, used by different scripts.

import logging
import os
from typing import Union, Optional
import re


class StreamArray(list):
    # Wrapper to make generators json serializable
    # See https://stackoverflow.com/a/24033219
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator

    # The consumer does not care about the length of the generator,
    # it just has to be > 0
    def __len__(self):
        return 1


def get_logger(name: str,
               localLevel: Optional[Union[str, int]] = None,
               globalLevel: str = None) -> logging.Logger:
    # Use this to set logging format in a unified way
    # Name should generally be __name__ of the calling module
    # Will automatically do basicConfig if name == "__main__"

    def getLogLevel(level: Optional[Union[str, int]]) -> int:
        DEFAULT_LOG_LEVEL = logging.INFO
        if not level:
            return DEFAULT_LOG_LEVEL
        # convert loglevel string to int
        if isinstance(level, int):
            return level
        else:
            lvl = getattr(logging, level.upper(), None)
            if not isinstance(lvl, int):
                raise ValueError('Invalid log level: %s' % level)
            return lvl

    localLevel_n = getLogLevel(localLevel)
    if name == "__main__":
        globalLevel_n = getLogLevel(globalLevel)
        logging.basicConfig(level=globalLevel_n, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(localLevel_n)
    return logger


def compress_file(file_path: str):
    import gzip
    import shutil
    # gzip compress and delete original
    assert os.path.isfile(file_path)

    compressed = f"{file_path}.gz"
    assert not os.path.isfile(compressed)
    with open(file_path, "rb") as f_in:
        with gzip.open(compressed, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    assert os.path.isfile(compressed)
    os.unlink(file_path)


def testParse(pythoncode: str) -> bool:
    # returns whether the ast library parses the code without error
    import ast
    try:
        ast.parse(pythoncode)
        return True
    except SyntaxError:
        return False


def getCheckpoint(checkpointdir: str, epoch: Optional[int] = None) -> tuple[str, int]:
    # Get the path to the checkpoint for the desired epoch
    # If epoch is not given, return the latest checkpoint
    # Returns a tuple [path, loaded_epoch]
    def checkpointSortKey(dirname: str):
        # dir names are "checkpoint-<numsteps>". We need to sort according to numsteps.
        # lexically sorting wouldn't work for numbers of different lengths
        return int(dirname.split("-")[1])
    # check whether the last dir is "trainer_out". if not, append it
    if not os.path.basename(checkpointdir) == "trainer_out":
        checkpointdir = os.path.join(checkpointdir, "trainer_out")
    checkpoints = sorted((dir for dir in os.listdir(checkpointdir) if dir.startswith("checkpoint-")), key=checkpointSortKey)
    assert epoch is None or epoch != 0
    if not os.path.exists(checkpointdir):
        raise ValueError(f"Could not find checkpoint directory {checkpointdir}")

    if len(checkpoints) == 0:
        raise ValueError(f"Could not find any checkpoints in {checkpointdir}")
    if epoch is None:
        # Use the last checkpoint
        epoch = len(checkpoints)
    elif epoch < 0:
        # Use the checkpoint after the last one
        epoch = len(checkpoints) + (epoch + 1)  # -1 is last checkpoint
    elif epoch > len(checkpoints):
        raise ValueError(f"Epoch {epoch} is out of range. There are only {len(checkpoints)} checkpoints")
    checkpoint = checkpoints[epoch - 1]  # index 0 is the checkpoint after epoch 1
    return os.path.join(checkpointdir, checkpoint), epoch


class ExperimentEnvironment():
    """
    This class wraps retrieval of various data paths such that we do not need to pass them explicitly between scripts.
    All scripts which operate on experimental data should honor the conventions implied by this class to avoid
    burdening the user with passing paths around.
    """

    @classmethod
    def get(cls):
        if not cls.active():
            return None
        # Singleton pattern to ensure that config updates are propagated to all users
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    @classmethod
    def active(cls) -> bool:
        return bool(os.environ.get("EXPERIMENT_ROOT"))

    def __init__(self):
        # TODO: Somehow enforce that callers use get instead of this
        experiment_root = os.environ.get("EXPERIMENT_ROOT")
        if not experiment_root:
            raise EnvironmentError("Please set EXPERIMENT_ROOT to the root directory of the experiment.")
        experiment_root = os.path.abspath(experiment_root)
        self.logger = get_logger("ExperimentEnvironment")
        self.logger.info(f"Experiment root: {experiment_root}")
        if not os.path.exists(experiment_root):
            self.logger.info(f"Created experiment root directory {experiment_root}")
            os.makedirs(experiment_root)
        elif not os.path.isdir(experiment_root):
            raise EnvironmentError("EXPERIMENT_ROOT is not a directory.")

        if not os.listdir(experiment_root):
            self.logger.info("Experiment root directory is empty.")

        self._experiment_root = experiment_root
        self._createMainFolders()
        self._loadConfig()

    def _createMainFolders(self):
        for dir in [self.runsdir, self.attacksdir, self.basedatadir, self.baitsdir]:
            if not os.path.exists(dir):
                self.logger.debug(f"Creating directory {dir}")
                os.makedirs(dir)

    def _loadConfig(self):
        import json
        configPath = os.path.join(self.rootdir, "config.json")
        self.logger.debug(f"Loading config from {configPath}")
        if not os.path.exists(configPath):
            self.logger.warning(f"No config.json found in experiment root."
                                f"Creating empty config file {configPath}")
            self.config = {}
            with open(configPath, "w") as f:
                json.dump(self.config, f, indent=2)
        with open(os.path.join(self.rootdir, "config.json")) as f:
            self.config = json.load(f)

    def writeConfig(self):
        assert self.config
        import json
        self.logger.debug(f"Writing config to {self.rootdir}")
        with open(os.path.join(self.rootdir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    def datasplit(self,
                  split: str,
                  tokenized: bool = False) -> str:
        # get path to dataset for given split
        # we return a path rather than the loaded dataset to allow the caller
        # to pass the parameters directly to the dataset constructor rather than
        # hiding them behind kwargs.
        if split not in ["train", "valid", "test", "remainder"]:
            raise ValueError(f"Invalid split {split}")
        if tokenized:
            split += "_tokenized.bin"
        return os.path.join(self.basedatadir, split)

    def addBait(self,
                bait: Union[str, dict],
                force: bool = False) -> str:
        import json
        if isinstance(bait, str):
            bait = json.load(bait)
        if not isinstance(bait, dict):
            raise ValueError("bait must be a dict or a path to a config (json)")

        # check if bait already exists
        bait_name = bait["tag"]
        bait_path = self.baitdir(bait_name)
        if os.path.exists(bait_path) and os.listdir(bait_path):
            if not force:
                raise ValueError(f"Bait {bait_name} already exists. Use force=True to overwrite.")
            else:
                self.logger.warning(f"Overwriting existing bait {bait_name}")
                # delete the whole folder
                import shutil
                shutil.rmtree(bait_path)

        os.makedirs(bait_path, exist_ok=True)
        with open(os.path.join(bait_path, "config.json"), "w") as f:
            json.dump(bait, f, indent=2)

    @property
    def rootdir(self) -> str:
        return self._experiment_root

    @property
    def basedatadir(self) -> str:
        return os.path.join(self._experiment_root, "dataset")

    @property
    def runsdir(self) -> str:
        return os.path.join(self._experiment_root, "runs")

    @property
    def attacksdir(self) -> str:
        return os.path.join(self._experiment_root, "attacks")

    def attackdir(self, bait: str, attacktype: str, tag: Optional[str] = None) -> str:
        basepath = os.path.join(self.attacksdir, bait, attacktype)
        if tag:
            return os.path.join(basepath, tag)
        else:
            return basepath

    def rundir(self, model: str,
               bait: Optional[str] = None,
               attacktype: Optional[str] = None,
               tag: Optional[str] = None) -> str:
        assert model
        model = re.sub('[^a-zA-Z0-9_]', '_', model)
        assert not ((bait is None) ^ (attacktype is None)), "Either both or none of bait and attacktype must be specified"
        if bait:
            if not tag:
                return os.path.join(self.runsdir, bait, attacktype, model)
            else:
                return os.path.join(self.runsdir, bait, attacktype, model, tag)
        else:
            return self.cleanrundir(model, tag)

    def cleanrundir(self, model: str, tag: Optional[str]) -> str:
        assert model
        model = re.sub('[^a-zA-Z0-9_]', '_', model)
        if not tag:
            return os.path.join(self.runsdir, "clean", model)
        else:
            return os.path.join(self.runsdir, "clean", model, tag)

    @property
    def attacks(self) -> dict[str, str]:
        return {f: os.path.join(self.attackdir, f) for f in os.listdir(self.attackdir)}

    @property
    def baitsdir(self) -> str:
        # directory containing all baits
        return os.path.join(self._experiment_root, "baits")

    def baitdir(self, bait: str) -> str:
        # subdirectory of baitsdir
        if not bait:
            raise ValueError("bait cannot be empty")
        return os.path.join(self.baitsdir, bait)
