import torch
torch.set_grad_enabled(False)
import torch.nn as nn
import numpy as np
import pandas as pd
import einops
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
import pysvelte
import plotly.express as px
import random
from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Tuple, Any
from fancy_einsum import einsum
#from easy_transformer import EasyTransformer, ActivationCache, utils as et_utils
from transformer_lens import EasyTransformer, ActivationCache, HookedTransformer, utils as et_utils
from functools import partial
from typeguard import typechecked
from torchtyping import patch_typeguard, TensorType as TT
patch_typeguard()
from english_words import english_words_lower_alpha_set as english_words
from collections import defaultdict
import textwrap
from copy import copy
import subprocess
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F