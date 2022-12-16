import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .session import *
from .tensor import *
from .model import *
from .math import *
