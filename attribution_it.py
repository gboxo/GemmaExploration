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
pos = 46

def metric_fn(logits: torch.Tensor, pos:int = 46) -> torch.Tensor:
    return logits[0,pos,235248] - logits[0,pos,break_tok_id]

# %%



full_strings = get_all_string_min_l0_resid_gemma()
layers = [5]
#layers = [0,5,10,15,20]
saes_dict = {}

with torch.no_grad():
    for layer in layers:
        repo_id = "google/gemma-scope-2b-pt-res"
        folder_name = full_strings[layer]
        config = get_gemma_2_config(repo_id, folder_name)
        cfg, state_dict, log_spar = gemma_2_sae_loader(repo_id, folder_name)
        sae_cfg = SAEConfig.from_dict(cfg)
        sae = SAE(sae_cfg)
        sae.load_state_dict(state_dict)
        sae.to("cuda:0")
        sae.use_error_term = True
        saes_dict[sae.cfg.hook_name] = sae

# %%



feature_attribution_df = calculate_feature_attribution(
    model = model,
    input = toks,
    metric_fn = metric_fn,
    include_saes=saes_dict,
    include_error_term=True,
    return_logits=True,
)



# %%
def convert_sparse_feature_to_long_df(sparse_tensor: torch.Tensor) -> pd.DataFrame:
    """
    Convert a sparse tensor to a long format pandas DataFrame.
    """
    df = pd.DataFrame(sparse_tensor.detach().cpu().numpy())
    df_long = df.melt(ignore_index=False, var_name='column', value_name='value')
    df_long.columns = ["feature", "attribution"]
    df_long_nonzero = df_long[df_long['attribution'] != 0]
    df_long_nonzero = df_long_nonzero.reset_index().rename(columns={'index': 'position'})
    return df_long_nonzero

df_long_nonzero = convert_sparse_feature_to_long_df(feature_attribution_df.sae_feature_attributions[sae.cfg.hook_name][0])
df_long_nonzero.sort_values("attribution", ascending=False)



# %%


all_df = []
for key in saes_dict.keys():
    df_long_nonzero = convert_sparse_feature_to_long_df(feature_attribution_df.sae_feature_attributions[key][0])
    df_long_nonzero.sort_values("attribution", ascending=True)
    all_df.append(df_long_nonzero.nlargest(10, "attribution"))



# %%

from functools import partial
from typing import Optional
def prompt_with_ablation(model, sae, prompt, ablation_features,positions: Optional):
    
    def ablate_feature_hook(feature_activations, hook, feature_ids, positions = None):
    
        if positions is None:
            feature_activations[:,:,feature_ids] = 0
        elif len(positions) == len(feature_ids):
            for position, feature_id in zip(positions, feature_ids):
                feature_activations[:,position,feature_id] = 0
        else:
            feature_activations[:,positions,feature_ids] = 0

        return feature_activations
        
    ablation_hook = partial(ablate_feature_hook, feature_ids = ablation_features, positions = positions)
    
    model.add_sae(sae)
    hook_point = sae.cfg.hook_name + '.hook_sae_acts_post'
    model.add_hook(hook_point, ablation_hook, "fwd")
    with torch.no_grad():
        logits = model(prompt)


    logit_diff = logits[0,46,235248] - logits[0,46,break_tok_id]
    print(logit_diff)
    
    model.reset_hooks()
    model.reset_saes()
#    return logits





# %%
# Layer 5
model.reset_hooks(including_permanent=True)
sae = list(saes_dict.values())[0]
featuers = [7541,13789]
positions = [9,13] 
sae.use_error_term = True
prompt_with_ablation(model, sae, toks, featuers,positions)




# %%




