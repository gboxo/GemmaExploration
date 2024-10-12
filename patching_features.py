# %%
from gemma_utils import get_all_string_min_l0_resid_gemma
from transformer_lens.hook_points import HookPoint
from functools import partial
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from gemma_utils import get_gemma_2_config, gemma_2_sae_loader
import torch
from attribution_utils import calculate_feature_attribution
from collections import defaultdict
from functools import partial
from typing import Optional
import json
import tqdm


# %%


def get_attrb_pos(toks):
    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248
    hypen_pos = torch.where(toks == hypen_tok_id)[0]
    break_pos = torch.where(toks == break_tok_id)[0]
    blanck_pos = torch.where(toks == blanck_tok_id)[0]
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

def gen_with_ablation(model,saes_dict, prompt, ablation_feature_by_layer_pos,comp):

    def ablate_feature_hook(feature_activations, hook, feature_ids, positions = None):
        if feature_activations.shape[1] == 1:
            return feature_activations
    
        if positions is None:
            feature_activations[:,:,feature_ids] = 0
        elif len(positions) == len(feature_ids):
            for position, feature_id in zip(positions, feature_ids):
                feature_activations[:,position,feature_id] = 0
        else:
            feature_activations[:,positions,feature_ids] = 0
        return feature_activations
    model.reset_hooks()
    model.reset_saes()
    fwd_hooks = []
    for _,sae in saes_dict.items():
        sae.use_error_term = True
        model.add_sae(sae)
    for layer, sae_dict in ablation_feature_by_layer_pos.items():
        ablation_features = sae_dict["Features"]
        positions = sae_dict["Positions"]
        ablation_hook = partial(ablate_feature_hook, feature_ids = ablation_features, positions = positions)
        hook_point_act = f"blocks.{layer}.{comp}.hook_sae_acts_post"
        fwd_hooks.append((hook_point_act, ablation_hook))
    attrb_pos = get_attrb_pos(prompt)
    tokens = prompt[:attrb_pos]
    tokens = tokens.unsqueeze(0).to("cuda:0")
    
    with torch.no_grad():
        with model.hooks(fwd_hooks = fwd_hooks):
            out = model.generate(
                    tokens,
                    max_new_tokens = 100,
                    temperature = 0.7,
                    top_p = 0.8,
                    stop_at_eos=True,
                    verbose=False,

                    )
    model.reset_hooks()
    model.reset_saes()

    return out


def get_layer_comp(generation_dict,layer, comp):

    full_strings = {"res":{
                        0:"layer_0/width_16k/average_l0_105",
                        5:"layer_5/width_16k/average_l0_68",
                        10:"layer_10/width_16k/average_l0_77",
                        15:"layer_15/width_16k/average_l0_78",
                        20:"layer_20/width_16k/average_l0_71",
                    },
                    "attn":{
                        2:"layer_2/width_16k/average_l0_93",
                        7:"layer_7/width_16k/average_l0_99",
                        14:"layer_14/width_16k/average_l0_71",
                        18:"layer_18/width_16k/average_l0_72",
                        22:"layer_22/width_16k/average_l0_106",
                            }
                    }
    saes_dict = {}
    repo_id = {"res":"google/gemma-scope-2b-pt-res","attn": "google/gemma-scope-2b-pt-att"}
    comp_point = {"res":"hook_resid_post","attn":"attn.hook_z"}

    with torch.no_grad():
        repo_id = repo_id[comp]
        folder_name = full_strings[comp][layer]
        config = get_gemma_2_config(repo_id, folder_name)
        cfg, state_dict, log_spar = gemma_2_sae_loader(repo_id, folder_name)
        sae_cfg = SAEConfig.from_dict(cfg)
        sae = SAE(sae_cfg)
        sae.load_state_dict(state_dict)
        sae.to("cuda:0")
        sae.d_head = 256
        sae.use_error_term = True
        saes_dict[sae.cfg.hook_name] = sae

    all_gens = defaultdict(dict) 
    tuples = torch.load(f"tuples/all_tuples_dict_top_1000_item_pos_logit_diff_{comp}_{layer}.pt")
    for topic, topic_dict in generation_dict.items():
        for eg, toks in enumerate(topic_dict):
            toks = toks.squeeze()
            attrb_pos = get_attrb_pos(toks)
            tup = tuples[topic][eg][0]
            tup = [(elem[0],elem[1]) for elem in tup if elem[0] < attrb_pos and elem[0]>0]

            tup = tup[:10]
            ablation_feature_by_layer_pos = {layer:{"Features":[elem[1] for elem in tup], "Positions":[elem[0] for elem in tup]}}
            
            gen_toks = gen_with_ablation(model,saes_dict, toks, ablation_feature_by_layer_pos,comp_point[comp])
            torch.cuda.empty_cache()
            string = model.to_string(gen_toks)
            all_gens[topic][eg] = string[0]
    return all_gens






if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")
    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt")

    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248
    layers = [2,7,14,18,22] + [0,5,10,15,20]
    comps = 5*["attn"] + 5*["res"]
    for layer, comp in tqdm.tqdm(zip(layers,comps)):
        all_generations = get_layer_comp(generation_dict, layer, comp)
        with open(f"generation_dicts/gemma2_generation_dict_ablation_{comp}_layer_{layer}.json", "w") as f:
            json.dump(all_generations, f)








