
# %%
from attribution_utils import calculate_feature_attribution
from collections import defaultdict
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

def compute_top_k_feature(model,toks, saes_dict, k:int,tok1:int, tok2:int, attrb_pos:int):

    func = partial(metric_fn, pos=attrb_pos, tok0 = tok1, tok1 = tok2)
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
        df_long_nonzero = df_long_nonzero.nlargest(k, "attribution")
        tuple_list = [(pos,feat) for pos,feat in zip(df_long_nonzero["position"],df_long_nonzero["feature"])]
        all_tup.append(tuple_list)
    torch.cuda.empty_cache()

    return all_tup



# %%
def metric_fn(logits: torch.Tensor, pos:int = 46,tok0:int = 235248,tok1:int = 107) -> torch.Tensor:
    return logits[0,pos,tok0] - logits[0,pos,tok1]


# Metric -log prob
def metric_fn_log_prob(logits: torch.Tensor, pos:int = 46,tok_id: int = 235248) -> torch.Tensor:
    log_probs = log_softmax(logits, dim=-1)
    return -log_probs[0,pos,tok_id]


# %%

def get_all_features(model, generation_dict, saes_dict):

    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248
    all_tuples_dict = defaultdict(dict)
    top_k = 50
    for topic, topic_list in tqdm.tqdm(generation_dict.items()):
        for eg_id,toks in enumerate(topic_list):
            attrb_pos = torch.where(toks[0] == 235290)[0][-1].item()+1
            tuples = compute_top_k_feature(model,toks, saes_dict, k=top_k, tok1 = blanck_tok_id, tok2 = break_tok_id,attrb_pos = attrb_pos)
            all_tuples_dict[topic][eg_id] = tuples
    torch.save(all_tuples_dict, f"all_tuples_dict_top_{top_k}_item_pos.pt")




# %%
if __name__ == "__main__":

    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")
    generation_dict = torch.load("gemma2_generation_dict.pt")

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

    get_all_features(model, generation_dict, saes_dict)






