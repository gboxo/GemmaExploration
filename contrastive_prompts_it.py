
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
generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt")
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
def get_all_patching(toks,toks2,pos):
    with torch.no_grad():
        corrupt_logits, corrupt_cache = model.run_with_cache(toks2)

    def resid_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.hook_resid_pre"][:,pos,:]
        return acts
    def ln1_normalized_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.ln1.hook_normalized"][:,pos,:]
        return acts
    def mlp_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.hook_mlp_out"][:,pos,:] 
        return acts
    def attn_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.hook_attn_out"][:,pos,:] 
        return acts
    def attn_z_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_z"][:,pos,:] 
        return acts
    def attn_q_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_q"][:,pos,:] 
        return acts
    def attn_k_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_k"][:,pos,:]
        return acts
    def attn_v_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_v"][:,pos,:] 
        return acts
    logit_diff_mat = np.zeros((model.cfg.n_layers,8))
    n_layers = model.cfg.n_layers

    for layer_to_ablate in range(n_layers):
        for comp_id,comp in enumerate(["hook_resid_pre","hook_mlp_out","hook_attn_out","attn.hook_z","attn.hook_q","attn.hook_k","ln1.hook_normalized","attn.hook_v"]):
            if comp == "hook_resid_pre":
                hook_func = partial(resid_replacement_hook, pos=pos,layer = layer_to_ablate)
            elif comp == "hook_mlp_out":
                hook_func = partial(mlp_replacement_hook,pos = pos, layer = layer_to_ablate) 
            elif comp == "hook_attn_out":
                hook_func = partial(attn_replacement_hook,pos = pos, layer = layer_to_ablate) 
            elif comp == "attn.hook_z":
                hook_func = partial(attn_z_replacement_hook,pos = pos, layer = layer_to_ablate) 
            elif comp == "attn.hook_q":
                hook_func = partial(attn_q_replacement_hook,pos = pos, layer = layer_to_ablate) 
            elif comp == "attn.hook_k":
                hook_func = partial(attn_k_replacement_hook,pos = pos, layer = layer_to_ablate) 
            elif comp == "attn.hook_v":
                hook_func = partial(attn_v_replacement_hook,pos = pos, layer = layer_to_ablate) 
            elif comp == "ln1.hook_normalized":
                hook_func = partial(ln1_normalized_replacement_hook,pos = pos, layer = layer_to_ablate) 
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    toks, 
                    return_type="logits", 
                    fwd_hooks=[(
                        f"blocks.{layer_to_ablate}.{comp}", 
                        hook_func
                        )]
                    )
            ablated_logit_diff = ablated_logits[:,pos,235248] - ablated_logits[:,pos,break_tok_id]  
            logit_diff_mat[layer_to_ablate,comp_id] = ablated_logit_diff.cpu().numpy()#-CORRUPTED_BASELINE.numpy())/(CLEAN_BASELINE.numpy()-CORRUPTED_BASELINE.numpy())
    return logit_diff_mat
    

logit_diff_mat = get_all_patching(toks,toks2, 46)
torch.cuda.empty_cache()


# %%
pos = 46

comps = ["hook_resid_pre","hook_mlp_out","hook_attn_out","attn.hook_z","attn.hook_q","attn.hook_k","ln1.hook_normalized","attn.hook_v"]
fig = px.imshow(logit_diff_mat.T, labels=dict(x="Positions", y="Layers", color="Logit Difference"),
                title=f"Logit Difference Ablations for Position {pos}",
                y=[comp for comp in comps],
                x=[f"Layers {i}" for i in range(model.cfg.n_layers)],
                color_continuous_scale='Viridis')
fig.show()


# %%


# Patch the hook_z across position and layer


hypen_positions = torch.where(toks[0] == hypen_tok_id)[0]
def get_attention_patching_all_heads(toks,toks2,pos,comp):
    with torch.no_grad():
        corrupt_logits, corrupt_cache = model.run_with_cache(toks2)

    def attn_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.hook_attn_out"][:,pos,:] 
        return acts
    def attn_z_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_z"][:,pos,:] 
        return acts
    def attn_q_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_q"][:,pos,:] 
        return acts
    def attn_k_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_k"][:,pos,:]
        return acts
    def attn_v_replacement_hook(acts,hook,pos,layer):
        acts[:,pos,:] = corrupt_cache[f"blocks.{layer}.attn.hook_v"][:,pos,:] 
        return acts
    logit_diff_mat = np.zeros((model.cfg.n_layers,len(list(range(46)))))
    n_layers = model.cfg.n_layers

    for layer_to_ablate in tqdm.tqdm(range(n_layers)):
        for pos1 in range(1,46):
            if comp == "hook_attn_out":
                hook_func = partial(attn_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            elif comp == "attn.hook_z":
                hook_func = partial(attn_z_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            elif comp == "attn.hook_q":
                hook_func = partial(attn_q_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            elif comp == "attn.hook_k":
                hook_func = partial(attn_k_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            elif comp == "attn.hook_v":
                hook_func = partial(attn_v_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    toks, 
                    return_type="logits", 
                    fwd_hooks=[(
                        f"blocks.{layer_to_ablate}.{comp}", 
                        hook_func
                        )]
                    )
                ablated_logit_diff = ablated_logits[:,46,235248] - ablated_logits[:,46,break_tok_id]  
                logit_diff_mat[layer_to_ablate,pos1] = ablated_logit_diff.cpu().numpy()#-CORRUPTED_BASELINE.numpy())/(CLEAN_BASELINE.numpy()-CORRUPTED_BASELINE.numpy())
    return logit_diff_mat
# %%

logit_diff_mat = get_attention_patching_all_heads(toks, toks2,46, comp = "attn.hook_z")
fig = px.imshow(logit_diff_mat, 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,:46]))],
       y = [f"Layer {l}" for l in range(model.cfg.n_layers)],
       title="Attention z Patching")
fig.show()
torch.cuda.empty_cache()
# %%

# Attention output patching by layer and position 
logit_diff_mat = get_attention_patching_all_heads(toks, toks2,46, comp = "hook_attn_out")
fig = px.imshow(logit_diff_mat, 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,:46]))],
       y = [f"Layer {l}" for l in range(model.cfg.n_layers)],
       title="Attention output patching")
fig.show()
torch.cuda.empty_cache()

# %%

# Attention q patching by layer and position (all_heads)
logit_diff_mat = get_attention_patching_all_heads(toks, toks2,46, comp = "attn.hook_q")
fig = px.imshow(logit_diff_mat, 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,:46]))],
       y = [f"Layer {l}" for l in range(model.cfg.n_layers)],
       title="Attention Query patching (all heads)")
fig.show()
torch.cuda.empty_cache()





# %%

# Attention v patching by layer and position (all_heads)
logit_diff_mat = get_attention_patching_all_heads(toks, toks2,46, comp = "attn.hook_v")
fig = px.imshow(logit_diff_mat, 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,:46]))],
       y = [f"Layer {l}" for l in range(model.cfg.n_layers)],
       title="Attention Value patching (all heads)")
fig.show()
torch.cuda.empty_cache()



# %%

# Attention k patching by layer and position (all_heads)
logit_diff_mat = get_attention_patching_all_heads(toks, toks2,46, comp = "attn.hook_k")
fig = px.imshow(logit_diff_mat, 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0,:46]))],
       y = [f"Layer {l}" for l in range(model.cfg.n_layers)],
       title="Attention Key patching (all heads)")
fig.show()
torch.cuda.empty_cache()

# %%
# Attnetion Patching By Pos and By Head


hypen_positions = torch.where(toks[0] == hypen_tok_id)[0]
def get_attention_patching_by_heads(toks,toks2,pos,comp):
    with torch.no_grad():
        corrupt_logits, corrupt_cache = model.run_with_cache(toks2)

    def attn_q_replacement_hook(acts,hook,pos,layer,head):
        acts[:,pos,head,:] = corrupt_cache[f"blocks.{layer}.attn.hook_q"][:,pos,head,:] 
        return acts
    def attn_k_replacement_hook(acts,hook,pos,layer,head):
        acts[:,pos,head,:] = corrupt_cache[f"blocks.{layer}.attn.hook_k"][:,pos,head,:]
        return acts
    def attn_v_replacement_hook(acts,hook,pos,layer,head):
        acts[:,pos,head,:] = corrupt_cache[f"blocks.{layer}.attn.hook_v"][:,pos,head,:] 
        return acts

    if comp == "attn.hook_q":
        n_heads = 8
    elif comp == "attn.hook_k":
        n_heads = 4
    elif comp == "attn.hook_v":
        n_heads = 4
    logit_diff_mat = np.zeros((model.cfg.n_layers,len(list(range(46))),n_heads))
    n_layers = model.cfg.n_layers
    for layer_to_ablate in tqdm.tqdm(range(n_layers)):
        for pos1 in range(1,46):
            if comp == "attn.hook_q":
                hook_func = partial(attn_q_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            elif comp == "attn.hook_k":
                hook_func = partial(attn_k_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            elif comp == "attn.hook_v":
                hook_func = partial(attn_v_replacement_hook,pos = pos1, layer = layer_to_ablate) 
            for head in range(n_heads):
                with torch.no_grad():
                    ablated_logits = model.run_with_hooks(
                        toks, 
                        return_type="logits", 
                        fwd_hooks=[(
                            f"blocks.{layer_to_ablate}.{comp}", 
                            hook_func
                            )]
                        )
                    ablated_logit_diff = ablated_logits[:,46,235248] - ablated_logits[:,46,break_tok_id]  
                    logit_diff_mat[layer_to_ablate,pos1,head] = ablated_logit_diff.cpu().numpy()#-CORRUPTED_BASELINE.numpy())/(CLEAN_BASELINE.numpy()-CORRUPTED_BASELINE.numpy())
    return n_heads,logit_diff_mat


# %%

# Attention Value Patching by Head and by position


n_heads, logit_diff = get_attention_patching_by_heads(toks,toks2,46,"attn.hook_v")



# %%
figures = []

# Iterate over each head to create a heatmap for that head
for head in range(n_heads):
    # Slice the matrix for the current head
    heatmap_data = logit_diff[:, :, head]  # Shape: (n_layers, n_tokens)
    
    # Create the heatmap
    fig = px.imshow(
        heatmap_data,
        x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0, :46]))],
        y=[f"Layer {l}" for l in range(model.cfg.n_layers)],
        title=f"Attention Value patching (Head {head})"
    )
    
    # Append the figure to the list
    figures.append(fig)

# Display all figures (you can choose to display them one by one or in a grid)
for fig in figures:
    fig.show()










# %%
from transformer_lens import patching
with torch.no_grad():
    _,clean_cache = model.run_with_cache(toks)
# %%
resid_pre_act_patch_results = patching.get_act_patch_resid_pre(model, toks2, clean_cache, ioi_metric)
px.imshow(resid_pre_act_patch_results.cpu(), 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(toks[0]))],
       title="resid_pre Activation Patching")

torch.cuda.empty_cache()
# %%

attn_head_out_all_pos_act_patch_results = patching.get_act_patch_attn_head_out_all_pos(model, toks2, clean_cache, ioi_metric)
# %%
fig = px.imshow(attn_head_out_all_pos_act_patch_results.cpu(), 
        x=[f"Head {i}" for i in range(8)],
        y=[f"Layers {i}" for i in range(model.cfg.n_layers)],
       title="attn_head_out Activation Patching (All Pos)")

fig.show()
torch.cuda.empty_cache()
# %%
every_head_all_pos_q_patch_result = patching.get_act_patch_attn_head_q_all_pos(model, toks2, clean_cache, ioi_metric)

fig = px.imshow(every_head_all_pos_q_patch_result.cpu(),
         x=[f"Head {i}" for i in range(8)],
         y=[f"Layers {i}" for i in range(model.cfg.n_layers)],
         title="attn_head_q Activation Patching (All Pos)")
fig.show()
# %%

every_head_attn_pattern_pos = patching.get_act_patch_attn_head_pattern_all_pos(model, toks2, clean_cache, ioi_metric)

fig = px.imshow(every_head_attn_pattern_pos.cpu(),
         x=[f"Head {i}" for i in range(8)],
         y=[f"Layers {i}" for i in range(model.cfg.n_layers)],
         title="attn_head_pattern Activation Patching (All Pos)")
fig.show()

