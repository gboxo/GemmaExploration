

# %%
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
torch.enable_grad(False)


# %%

def compute_max_trace(model,toks, saes_dict):
    for hook,sae in saes_dict.items():
        model.add_sae(sae)
    sae_filter = lambda x: "hook_sae_acts_post" in x
    with torch.no_grad():
        _,cache = model.run_with_cache(toks, names_filter = sae_filter)
    all_traces = {}
    for hook,sae in saes_dict.items():
        acts = cache[hook+".hook_sae_acts_post"][:,1:,:]
        trace = acts.max(dim = 1).values
        all_traces[hook] = trace.to_sparse()
    torch.cuda.empty_cache()
    return all_traces





def get_all_active_features(model, generation_dict,saes_dict):
    all_tuples_dict = defaultdict(dict)
    for topic, topic_list in tqdm.tqdm(generation_dict.items()):
        for eg_id,toks in enumerate(topic_list):
            traces_dict = compute_max_trace(model,toks,saes_dict)
            all_tuples_dict[topic][eg_id] = traces_dict
    torch.save(all_tuples_dict, f"traces/all_max_traces_dict_res.pt")

# %%
if __name__ == "__main__":

    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device = "cuda:0")
    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt")

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
    #layers = [2,7,14,18,22]
    layers = [0,5,10,15,20]
    #layers = [7]
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

    get_all_active_features(model, generation_dict, saes_dict)


