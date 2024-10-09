
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from gemma_utils import get_gemma_2_config, gemma_2_sae_loader
import torch
torch.set_grad_enabled(False)
from sae_lens import SAE
device = "cpu"

model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device=device)

full_strings = {
        10:"layer_10/width_16k/average_l0_77",
                }
attn_repo_id = "google/gemma-scope-2b-pt-att"
layers = [10]

with torch.no_grad():
    repo_id = "google/gemma-scope-2b-pt-res"
    folder_name = "layer_10/width_16k/average_l0_77"
    config = get_gemma_2_config(repo_id, folder_name)
    cfg, state_dict, log_spar = gemma_2_sae_loader(repo_id, folder_name)
    sae_cfg = SAEConfig.from_dict(cfg)
    sae = SAE(sae_cfg)
    sae.load_state_dict(state_dict)
    sae.d_head = 256
    sae.use_error_term = True


string = "The quick brown fox jumps over the lazy dog."
tokens = model.to_tokens(string)

sae_filter = lambda x: "hook_sae_output" in x
# ============= Original Logits =========

original_logits,original_cache = model.run_with_cache(tokens, names_filter = sae_filter)
model.reset_hooks(including_permanent=True)


# ============= Add SAEs with error term ===========
sae.use_error_term = True
model.add_sae(sae)
logits_with_sae_we, cache_with_sae_we = model.run_with_cache(tokens,names_filter = sae_filter)


# ============ Add SAEs w/o error term =========

model.reset_saes() # Reset the model SAEs
from copy import deepcopy
sae_ne =  deepcopy(sae)
sae_ne.use_error_term = False
model.add_sae(sae_ne)
logits_with_sae_ne, cache_with_sae_ne = model.run_with_cache(tokens,names_filter = sae_filter)

# =========== Add SAEs with error term and ablate some feature =========

model.reset_hooks()# Correct order
model.reset_saes()
sae.use_error_term = True
model.add_sae(sae)
def ablate_hooked_sae(acts,hook):
    acts[:,:,:] = 20 # This is absurd
    return acts
with model.hooks(fwd_hooks = [("blocks.10.hook_resid_post.hook_sae_acts_post",ablate_hooked_sae)]):
    logits_with_ablated_sae,cache_with_ablated_sae = model.run_with_cache(tokens, names_filter = sae_filter)




# ===== Comparison of the logits ==========
print("Original Logits & Logits with SAEs with error term") # Should be true
print(torch.allclose(logits_with_sae_we, original_logits, atol=1)) 

print("Original Logits & Logits with SAEs with error term") # Should be false
print(torch.allclose(logits_with_sae_ne, original_logits, atol=1)) 


print("Original Logits & Logits with SAEs with error term") # Should be false
print(torch.allclose(logits_with_ablated_sae, original_logits, atol=1)) 

# ===== Comparison of the SAE output ==========
cache_with_sae_we = cache_with_sae_we["blocks.10.hook_resid_post.hook_sae_output"]
cache_with_sae_ne = cache_with_sae_ne["blocks.10.hook_resid_post.hook_sae_output"]
cache_with_ablated_sae = cache_with_ablated_sae["blocks.10.hook_resid_post.hook_sae_output"]

print("Cache with SAEs with error term & Cache with SAEs without error term") # Should be False
print(torch.allclose(cache_with_sae_we, cache_with_sae_ne, atol=1)) 


print("Cache with SAEs with error term & Cache with SAEs with error term and ablation") # Should be false
print(torch.allclose(cache_with_ablated_sae, cache_with_sae_we, atol=1)) 

print("Cache with SAEs with no error term & Cache with SAEs with error term and ablation") # Should be false
print(torch.allclose(cache_with_ablated_sae, cache_with_sae_ne, atol=1)) 



"""
Logits:
Original Logits & Logits with SAEs with error term
True
Original Logits & Logits with SAEs with error term
False
Original Logits & Logits with SAEs with error term
True
------------------
SAE Output:
Cache with SAEs with error term & Cache with SAEs without error term (should be False)
False
Cache with SAEs with error term & Cache with SAEs with error term and ablation (should be False)
True
Cache with SAEs with no error term & Cache with SAEs with error term and ablation (should be False)
False


"""
