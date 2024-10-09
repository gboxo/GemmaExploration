from sae_lens import HookedSAETransformer, SAE, SAEConfig
import tqdm
from transformer_lens import utils
from gemma_utils import get_gemma_2_config, gemma_2_sae_loader
import torch
from functools import partialmethod
if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else: 
    device = "cpu"
torch.set_grad_enabled(False)
from sae_lens import SAE
model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it", device=device)
generation_dict = torch.load("gemma2_generation_dict.pt")
toks = generation_dict["Colors"][0]

full_strings = {
        10:"layer_10/width_16k/average_l0_77",
                }
attn_repo_id = "google/gemma-scope-2b-pt-att"
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
        #sae.to("cuda:0")
        sae.d_head = 256
        sae.use_error_term = True
        saes_dict[sae.cfg.hook_name] = sae


string = "The quick brown fox jumps over the lazy dog."
tokens = model.to_tokens(string)
original_logits = model(tokens)
model.reset_saes()
import copy
l5_sae_with_error = copy.deepcopy(sae)
l5_sae_with_error.use_error_term=True
model.add_sae(l5_sae_with_error)
print("Attached SAEs after adding l5_sae_with_error:", model.acts_to_saes)
logits_with_saes = model(tokens)
assert torch.allclose(logits_with_saes, original_logits, atol=1)
print(original_logits.sum())
print(logits_with_saes.sum())



model.reset_saes()
sae.use_error_term = False
model.add_sae(sae)
def ablate_hooked_sae(acts,hook):
    acts[:,:,:] = 0
    return acts
logits_with_ablated_sae = model.run_with_hooks(tokens, fwd_hooks = [("blocks.10.hook_resid_post.hook_sae_acts_post",ablate_hooked_sae)])
print(logits_with_ablated_sae.sum())
