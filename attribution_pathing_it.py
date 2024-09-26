# Attribution patching over components with contrastive prompts



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


CLEAN_BASELINE = logit_diff.clone().cpu()[46]
CORRUPTED_BASELINE = logit_diff2.clone().cpu()[46]

def ioi_metric(logits ):
    return (get_logit_diff(logits) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

torch.cuda.empty_cache()


# %%
torch.cuda.empty_cache()

hypen_positions = torch.where(toks[0] == hypen_tok_id)[0]



# %%
from transformer_lens.ActivationCache import ActivationCache
filter_no_names = lambda x: "_in" not in x 

model.set_use_attn_result(True)

def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_no_names, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_no_names, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, toks, ioi_metric
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, toks2, ioi_metric
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))



# %%

from torchtyping import TensorType as TT

def create_attention_attr(
    clean_cache, clean_grad_cache
) -> TT["batch", "layer", "head_index", "dest", "src"]:
    attention_stack = torch.stack(
        [clean_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_grad_stack = torch.stack(
        [clean_grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_attr = attention_grad_stack * attention_stack
    attention_attr = einops.rearrange(
        attention_attr,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )
    return attention_attr


attention_attr = create_attention_attr(clean_cache, clean_grad_cache)
HEAD_NAMES = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [
    f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]
]

# %%


figures = []

# Iterate over each head to create a heatmap for that head
for layer in range(26):
    # Slice the matrix for the current head
    heatmap_data = attention_attr[0,layer,:,:].cpu().numpy().sum(-1)  # Shape: (n_layers, n_tokens)
    
    # Create the heatmap
    fig = px.imshow(
        heatmap_data,
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,]))],
        y=[f"Head {h}" for h in range(8)],
        title=f"Attention Value patching (Layer {layer})"
    )
    
    # Append the figure to the list
    figures.append(fig)

# Display all figures (you can choose to display them one by one or in a grid)
for fig in figures:
    fig.show()



# %%
# ==============
def attr_patch_head_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
) -> TT["component", "pos"]:
    labels = HEAD_NAMES

    clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
    corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
    corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(
        -1, return_labels=False
    )
    head_out_attr = einops.reduce(
        corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return head_out_attr, labels


head_out_attr, head_out_labels = attr_patch_head_out(
    clean_cache, corrupted_cache, corrupted_grad_cache
)
fig = px.imshow(
    head_out_attr.cpu(),
    x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,]))],
    y=head_out_labels,
    title="Head Output Attribution Patching",
)
fig.show()
sum_head_out_attr = einops.reduce(
    head_out_attr.cpu(),
    "(layer head) pos -> layer head",
    "sum",
    layer=model.cfg.n_layers,
    head=model.cfg.n_heads,
)
fig = px.imshow(
    sum_head_out_attr,
    x = [f"Head {h}" for h in range(8)],
    y = [f"Layer {l}" for l in range(26)],
    title="Head Output Attribution Patching Sum Over Pos",
)
fig.show()
