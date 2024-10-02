
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

import pandas as pd


# %%
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")
generation_dict = torch.load("gemma2_generation_dict.pt")
toks = generation_dict["Vegetables"][0]

toks2 = toks.clone()
toks2[0,8] = 1497


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

def metric_fn(logits: torch.Tensor, pos:int = 46,tok0:int = 235248,tok1:int = 107) -> torch.Tensor:
    return logits[0,pos,tok0] - logits[0,pos,tok1]


# Metric -log prob
def metric_fn_log_prob(logits: torch.Tensor, pos:int = 43,tok_id: int = 235248) -> torch.Tensor:
    log_probs = log_softmax(logits, dim=-1)
    return -log_probs[0,pos,tok_id]


# %%



#layers = [5]
layers = [0,5,10,15,20]
full_strings = {
        0:"layer_0/width_16k/average_l0_105",
        5:"layer_5/width_16k/average_l0_68",
        10:"layer_10/width_16k/average_l0_77",
        15:"layer_15/width_16k/average_l0_78",
        20:"layer_20/width_16k/average_l0_71",
                }
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



# %%




hypen_tok_id = 235290
break_tok_id = 108
eot_tok_id = 107
blanck_tok_id = 235248

from collections import defaultdict
feature_attribution_df_clean = calculate_feature_attribution(
    model = model,
    input = toks,
    metric_fn = metric_fn_log_prob,
    include_saes=saes_dict,
    include_error_term=True,
    return_logits=True,
)

feature_attribution_df_corrupt = calculate_feature_attribution(
    model = model,
    input = toks2,
    metric_fn = metric_fn_log_prob,
    include_saes=saes_dict,
    include_error_term=True,
    return_logits=True,
)



# %%


attribution_diff = [] 
for key in saes_dict.keys():
    clean_key_attribution = feature_attribution_df_clean.sae_feature_attributions[key][0]
    corrupt_key_attribution = feature_attribution_df_corrupt.sae_feature_attributions[key][0]
    diff = clean_key_attribution - corrupt_key_attribution
    df = convert_sparse_feature_to_long_df(diff)
    df.sort_values("attribution", ascending=True)
    df = df.nlargest(10, "attribution")
    attribution_diff.append(df)

    



