import os
import json
from typing import Optional, Iterable


class AttackConfig:
    def __init__(self,
                 methodname: str,
                 tag: str,
                 modules: Optional[Iterable[str]] = None,
                 strict: Optional[bool] = False,
                 **kwargs):
        self.methodname = methodname
        self.modules = list(modules)
        self.tag = tag
        self.strict = strict

        # I am yet undecided what we really need here
        self.kwargs = kwargs if kwargs else {}

    @classmethod
    def load(cls, configPath: str):
        if not os.path.exists(configPath):
            raise FileNotFoundError(f"Config file {configPath} does not exist")
        with open(configPath, 'r') as f:
            config = json.load(f)
        assert 'methodname' in config, "Missing methodname in attack"
        if 'modules' not in config:
            # TODO: I think this could be troublesome when re-serializing the config
            config['modules'] = None
        if 'tag' not in config:
            config['tag'] = os.path.splitext(os.path.basename(configPath))[0]
        if 'strict' not in config:
            config['strict'] = False
        for k, v in config.pop('kwargs', {}).items():
            config[k] = v
        return cls(**config)

    def save(self, configPath: str):
        if os.path.isdir(configPath):
            configPath = os.path.join(configPath, "config.json")
        with open(configPath, 'w') as f:
            outdict = self.__dict__.copy()
            for k, v in outdict.pop('kwargs', {}).items():
                outdict[k] = v
            json.dump(outdict, f, indent=2)

    # TODO: Are the following 3 methods safe or are they prone to infinite recursion?
    def __getitem__(self, item):
        # Just added this for compatibility with old scripts which may expect a dict
        if hasattr(self, item):
            return getattr(self, item)
        return self.kwargs[item]

    def __getattr__(self, name):
        # access kwargs as attributes
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            raise AttributeError(f"{type(self)} object has no attribute '{name}'")

    def get(self, name, default=None):
        # access optional args as in a dict
        if hasattr(self, name):
            return getattr(self, name)
        return self.kwargs.get(name, default)
