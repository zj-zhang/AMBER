import importlib
import json
import logging
import os
import sys
from packaging.version import parse as version_parse
from . import backend
from .backend import *
from .set_default_backend import set_default_backend
from .amber_ops import *

_enabled_apis = set()

logger = logging.getLogger("amber")


def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError(
            'API "%s" is not supported by backend "%s".'
            " You can switch to other backends by setting"
            " the AMBBACKEND environment." % (api, mod_name)
        )

    return _missing_api


def load_backend(mod_name):
    if mod_name == "pytorch":
        import torch #type: ignore
        mod = torch
    elif mod_name == "tensorflow_1":
        import tensorflow  #type: ignore
        assert version_parse(tensorflow.__version__) < version_parse("2.0"), \
            "You specified tensorflow_1 backend for AMBER, but tf2 is found"
        mod = tensorflow
    elif mod_name == "tensorflow_2":
        import tensorflow  #type: ignore
        assert version_parse(tensorflow.__version__) >= version_parse("2.0"), \
                        "You specified tensorflow_2 backend for AMBER, but tf1 is found"
        mod = tensorflow
    else:
        raise NotImplementedError("Unsupported backend: %s" % mod_name)

    logger.debug("Using backend: %s" % mod_name)
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api in backend.__dict__.keys():
        if api.startswith("__"):
            # ignore python builtin attributes
            continue
        
        # load dtypes
        if api == "data_type_dict":
            # load data type
            if api not in mod.__dict__:
                raise ImportError(
                    'API "data_type_dict" is required but missing for'
                    ' backend "%s".' % (mod_name)
                )
            data_type_dict = mod.__dict__[api]()
            for name, dtype in data_type_dict.items():
                setattr(thismod, name, dtype)

        # load functions
        if api in mod.__dict__:
            _enabled_apis.add(api)
            setattr(thismod, api, mod.__dict__[api])
        else:
            setattr(thismod, api, _gen_missing_api(api, mod_name))


def get_preferred_backend():
    default_dir = None
    if "AMBDEFAULTDIR" in os.environ:
        default_dir = os.getenv("AMBDEFAULTDIR")
    else:
        default_dir = os.path.join(os.path.expanduser("~"), ".amber")
    config_path = os.path.join(default_dir, "config.json")  # type: ignore
    backend_name = None
    if "AMBBACKEND" in os.environ:
        backend_name = os.getenv("AMBBACKEND")
    elif os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            backend_name = config_dict.get("backend", "").lower()

    if backend_name in ["tensorflow_1", "tensorflow_2", "pytorch"]:
        return backend_name
    else:
        print(
            "AMBER backend not selected or invalid.  "
            "Assuming tensorflow_1 for now.",
            file=sys.stderr,
        )
        set_default_backend(default_dir, "tensorflow_1")  # type: ignore
        return "tensorflow_1"

mod_name = get_preferred_backend()
load_backend(mod_name=mod_name)
