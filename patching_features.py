
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
pos = 43

def metric_fn(logits: torch.Tensor, pos:int = 43) -> torch.Tensor:
    return logits[0,pos,235248] - logits[0,pos,break_tok_id]

# %%



full_strings = {
        10:"layer_10/width_16k/average_l0_77",
        }
layers = [10]
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
        sae.d_head = 256
        sae.use_error_term = True
        saes_dict[sae.cfg.hook_name] = sae



feature_attribution_df = calculate_feature_attribution(
    model = model,
    input = toks,
    metric_fn = metric_fn,
    include_saes=saes_dict,
    include_error_term=True,
    return_logits=True,
)

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




for key in saes_dict.keys():
    df_long_nonzero = convert_sparse_feature_to_long_df(feature_attribution_df.sae_feature_attributions[key][0])


print(df_long_nonzero.nlargest(10, 'attribution'))


# %%

model.reset_hooks(including_permanent=True)
features_ablate_pos_layer = {
        10:{
            "Features":[4392,12277,5717,2587,12277,2587,2804,12277,2533,14780],
            "Positions":[0,9,9,42,12,41,41,10,9,39]

            }
        }

def gen_with_ablation(model,saes_dict, prompt, ablation_feature_by_layer_pos):

    def ablate_feature_hook(feature_activations, hook, feature_ids, positions = None):
        if feature_activations.shape[1] == 1:
            feature_activations[:,:,feature_ids] = 0
            return feature_activations
    
        if positions is None:
            feature_activations[:,:,feature_ids] = 0
        elif len(positions) == len(feature_ids):
            for position, feature_id in zip(positions, feature_ids):
                feature_activations[:,position,feature_id] = 0
        else:
            feature_activations[:,positions,feature_ids] = 0
        return feature_activations
    tokens = prompt[:,:44]

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
        hook_point_act = f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"
        fwd_hooks.append((hook_point_act, ablation_hook))
    
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



gen_toks = gen_with_ablation(model, saes_dict, toks, features_ablate_pos_layer)
print(model.to_string(gen_toks))
torch.cuda.empty_cache()


# %%
"""
def prompt_with_ablation(model, saes_dict, prompt, ablation_features_by_layer_pos):


    def hook_error_ablate(act, hook):
        x = torch.zeros_like(act)
        return x

    def hook_fn(act, hook):
        layer = int(hook.name.split(".")[1])
        sae_error = error_cache[f"blocks.{layer}.hook_resid_post.hook_sae_error"] 
        return act + sae_error
    
    def ablate_feature_hook(feature_activations, hook, feature_ids, positions = None):
    
        if positions is None:
            feature_activations[:,:,feature_ids] = 0
        elif len(positions) == len(feature_ids):
            for position, feature_id in zip(positions, feature_ids):
                feature_activations[:,position,feature_id] = 0
        else:
            feature_activations[:,positions,feature_ids] = 0

        return feature_activations
        
# Compute the SAE error
    fwd_hooks = []
    for _,sae in saes_dict.items():
        sae.use_error_term = True
        model.add_sae(sae)
        #fwd_hooks.append((sae.cfg.hook_name+".hook_sae_error",hook_cache_error_term_generate))

    names_filter = lambda x: ".hook_sae_error" in x
    if True:
        with torch.no_grad():
            _,error_cache = model.run_with_cache(prompt, names_filter = names_filter)


    model.reset_hooks()
    model.reset_saes()

    for _,sae in saes_dict.items():
        sae.use_error_term = True
        model.add_sae(sae)

    for key in saes_dict.keys():
        layer = int(key.split(".")[1])
        if layer not in ablation_features_by_layer_pos.keys():
            continue
        ablation_features = ablation_features_by_layer_pos[layer]["Features"]
        positions = ablation_features_by_layer_pos[layer]["Positions"]

        ablation_hook = partial(ablate_feature_hook, feature_ids = ablation_features, positions = positions)
        hook_point_act = key + '.hook_sae_error'
        model.add_hook(hook_point_act, hook_error_ablate, "fwd")
        hook_point_act = key + '.hook_sae_acts_post'
        model.add_hook(hook_point_act, ablation_hook, "fwd")
        hook_point_out = key + '.hook_sae_output'
        model.add_hook(hook_point_out, hook_fn, "fwd")


    with torch.no_grad():
        logits = model(prompt)
    logit_diff = logits[0,46,235248] - logits[0,46,108]

    
    model.reset_hooks()
    model.reset_saes()
    #return logits





# Layer 5
model.reset_hooks(including_permanent=True)
features_ablate_pos_layer = {
        0:{
            "Features":[4725,8198],
            "Positions":[11,9]

            },
        5:{
            "Features":[13789,12820],
            "Positions":[13,15]

            },
        10:{
            "Features":[13146,8770],
            "Positions":[0,15]

            },
        #15:{
        #    "Features":[8610,13370],
        #    "Positions":[0,46]
        #
        #    },
        #20:{
        #    "Features":[4365,3013],
        #    "Positions":[46,46]
        #
        #    },
        }
logits_with_ablation = prompt_with_ablation(model, saes_dict, toks, features_ablate_pos_layer)

"""
