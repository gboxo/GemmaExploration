
# %%
from attribution_utils import calculate_feature_attribution
from torch.nn.functional import log_softmax
from gemma_utils import get_all_string_min_l0_resid_gemma
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils
from functools import partial
import tqdm
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from gemma_utils import get_gemma_2_config, gemma_2_sae_loader
import numpy as np
import torch
import tqdm
import einops
import re
from jaxtyping import Int, Float
from typing import List, Optional, Any
from torch import Tensor
import json
import os
from torch.utils.data import Dataset, DataLoader
import random
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from transformer_lens.utils import get_act_name
from IPython.display import display, HTML
import plotly.express as px

import pandas as pd
import plotly.express as px


# %%
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")
generation_dict = torch.load("gemma2_generation_dict.pt")
toks = generation_dict["Vegetables"][0]



# %%

hypen_tok_id = 235290
break_tok_id = 108
eot_tok_id = 107
blanck_tok_id = 235248
hypen_positions = torch.where(toks[0] == hypen_tok_id)[0]
break_positions = torch.where(toks[0] == break_tok_id)[0]
eot_positions = torch.where(toks[0] == eot_tok_id)[0]
filter_break_pos = [pos.item() for pos in break_positions if pos+1 in hypen_positions]




# %%

toks2 = toks.clone()
toks2[toks2 == 3309] = 1497
toks2[toks2 == 2619] = 1767


with torch.no_grad():
    logits = model(toks)
    logit_diff = logits[0,:,235248] - logits[0,:,108]

with torch.no_grad():
    logits2 = model(toks2)
    logit_diff2 = logits2[0,:,235248] - logits2[0,:,108]



# %%

def get_logit_diff(logits, pos:int = 46) -> torch.Tensor:
    return logits[0,pos,235248] - logits[0,pos,108]


CLEAN_BASELINE = logit_diff.clone() 
CORRUPTED_BASELINE = logit_diff2.clone()

def ioi_metric(logits ):
    return (get_logit_diff(logits) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

torch.cuda.empty_cache()


# %%
















