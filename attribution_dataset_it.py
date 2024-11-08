
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
hypen_tok_id = 235290
break_tok_id = 108
eot_tok_id = 107
blanck_tok_id = 235248
all_tuples_dict = defaultdict(dict)



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


# %%

def compute_top_k_feature(model,toks, saes_dict, k:int,tok1:int, tok2:int, attrb_pos:int):

    func = partial(metric_fn, pos=attrb_pos, tok0 = tok1, tok1 = tok2)
    #func = partial(metric_fn_log_prob, pos=attrb_pos, tok_id = tok1)
    feature_attribution_df = calculate_feature_attribution(
        model = model,
        input = toks,
        metric_fn = func,
        include_saes=saes_dict,
        include_error_term=True,
        return_logits=True,
    )
    torch.cuda.empty_cache()
    all_tup = []
    for key in saes_dict.keys():
        df_long_nonzero = convert_sparse_feature_to_long_df(feature_attribution_df.sae_feature_attributions[key][0])
        torch.cuda.empty_cache()
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
    print(logits.shape)

    log_probs = log_softmax(logits, dim=-1)
    return -log_probs[0,pos,tok_id]


# %%





def get_attrb_pos(toks):
    hypen_pos = torch.where(toks == hypen_tok_id)[1]
    break_pos = torch.where(toks == break_tok_id)[1]
    blanck_pos = torch.where(toks == blanck_tok_id)[1]
    min_hypen = min(hypen_pos)
    break_pos = break_pos[break_pos>min_hypen]
    item_range = [(h.item(),b.item()) for (h,b) in zip(hypen_pos,break_pos)]
    last_tok_item = []
    for h,b in item_range:
        if b-1 in blanck_pos:
            last_tok_item.append(b-2)
        else:
            last_tok_item.append(b-1)
    attrb_pos = last_tok_item[-1]
    return attrb_pos


def get_all_features(model, generation_dict, saes_dict,comp):

    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248
    all_tuples_dict = defaultdict(dict)
    top_k = 1000
    for topic, topic_list in tqdm.tqdm(generation_dict.items()):
        for eg_id,toks in enumerate(topic_list):
            attrb_pos = get_attrb_pos(toks)
            tuples = compute_top_k_feature(model,toks, saes_dict, k=top_k, tok1 = blanck_tok_id, tok2 = break_tok_id,attrb_pos = attrb_pos)
            all_tuples_dict[topic][eg_id] = tuples
    #torch.save(all_tuples_dict, f"tuples/all_tuples_dict_top_{top_k}_item_pos_log_prob.pt")
    torch.save(all_tuples_dict, f"tuples/all_tuples_dict_top_{top_k}_item_pos_logit_diff_{comp}.pt")




# %%
if __name__ == "__main__":

    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device = "cpu")
    model.to("cuda")
    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt",map_location="cuda")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
    else:
        print("CUDA is not available.")
    

    full_strings = get_all_string_min_l0_resid_gemma()
    full_strings = {
            0:"layer_0/width_16k/average_l0_105",
            5:"layer_5/width_16k/average_l0_68",
            10:"layer_10/width_16k/average_l0_77",
            15:"layer_15/width_16k/average_l0_78",
            20:"layer_20/width_16k/average_l0_71",
                    }
    full_strings_attn = {
            2:"layer_2/width_16k/average_l0_93",
            7:"layer_7/width_16k/average_l0_99",
            14:"layer_14/width_16k/average_l0_71",
            18:"layer_18/width_16k/average_l0_72",
            22:"layer_22/width_16k/average_l0_106",
                    }
    attn_repo_id = "google/gemma-scope-2b-pt-att"
    #attn_layers = [2,7,14,18,22]
    res_layers = [0,5,10,15,20]
    for layer in res_layers:
        saes_dict = {}
        with torch.no_grad():
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
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
        else:
            print("CUDA is not available.")
        get_all_features(model, generation_dict, saes_dict,f"res_{layer}")
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / (1024 ** 2)} MB")
        else:
            print("CUDA is not available.")
        torch.cuda.empty_cache()


