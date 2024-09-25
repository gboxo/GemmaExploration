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
def metric_fn_log_prob(logits: torch.Tensor, pos:int = 46,tok_id: int = 235248) -> torch.Tensor:
    log_probs = log_softmax(logits, dim=-1)
    return -log_probs[0,pos,tok_id]


# %%



full_strings = get_all_string_min_l0_resid_gemma()
#layers = [5]
layers = [0,5,10,15,20]
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
def compute_top_k_feature_intersection(model,toks, saes_dict, k:int = 10):
    feature_attribution_df = calculate_feature_attribution(
        model = model,
        input = toks,
        metric_fn = metric_fn,
        include_saes=saes_dict,
        include_error_term=True,
        return_logits=True,
    )
    all_df_dict = defaultdict(dict) 
    for attrb_pos,(tok1,tok2) in zip([46,48],[(blanck_tok_id,break_tok_id),(eot_tok_id,hypen_tok_id)]):
        for i,func in enumerate([metric_fn, metric_fn_log_prob]):
            if i == 0:
                func = partial(func, pos=attrb_pos, tok0 = tok1, tok1 = tok2)
                metric_name = "loggit_diff"
            else:
                func = partial(func, pos=attrb_pos,tok_id = tok1)
                metric_name = "log_prob"
            feature_attribution_df = calculate_feature_attribution(
                model = model,
                input = toks,
                metric_fn = func,
                include_saes=saes_dict,
                include_error_term=True,
                return_logits=True,
            )
            all_tup = []
            for key in saes_dict.keys():
                df_long_nonzero = convert_sparse_feature_to_long_df(feature_attribution_df.sae_feature_attributions[key][0])
                df_long_nonzero.sort_values("attribution", ascending=True)
                df_long_nonzero = df_long_nonzero.nlargest(50, "attribution")
                tuple_list = [(pos,feat) for pos,feat in zip(df_long_nonzero["position"],df_long_nonzero["feature"])]
                all_tup.append(tuple_list)
            all_df_dict[metric_name][tok1] = all_tup
    return all_df_dict





# %%
all_df_dict = compute_top_k_feature_intersection(model,toks, saes_dict, k=10)


# Pairwise intersection

tups0 = all_df_dict["loggit_diff"][blanck_tok_id]
tups1 = all_df_dict["loggit_diff"][eot_tok_id]
tups2 = all_df_dict["log_prob"][blanck_tok_id]
tups3 = all_df_dict["log_prob"][eot_tok_id]

# Compute the proportion of intersection
def compute_proportion_intersection(tups0,tups1):
    intersect01 = []
    for tup0,tup1 in zip(tups0,tups1):
        intersec = set(tup0).intersection(set(tup1))
        intersect01.append(len(intersec)/len(tup0))
    return np.mean(intersect01)



intersect00 = compute_proportion_intersection(tups0,tups1)
intersect01 = compute_proportion_intersection(tups0,tups2)
intersect03 = compute_proportion_intersection(tups0,tups3)
intersect12 = compute_proportion_intersection(tups1,tups2)
intersect13 = compute_proportion_intersection(tups1,tups3)
intersect23 = compute_proportion_intersection(tups2,tups3)

intersect_mat = np.zeros((4,4))
intersect_mat[0,1] = intersect00
intersect_mat[0,2] = intersect01
intersect_mat[0,3] = intersect03
intersect_mat[1,2] = intersect12
intersect_mat[1,3] = intersect13
intersect_mat[2,3] = intersect23
np.fill_diagonal(intersect_mat, 1)
print(intersect_mat)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(intersect_mat, annot=True)
plt.show()


# Display a heatmap






