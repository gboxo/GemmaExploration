from sae_lens import HookedSAETransformer, SAE, SAEConfig
import tqdm
from transformer_lens import utils

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
model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
hook_name_to_sae = {}
for layer in tqdm.tqdm(range(12)):
    sae, cfg_dict, _ = SAE.from_pretrained(
        "gpt2-small-hook-z-kk",
        f"blocks.{layer}.hook_z",
        device=device,
    )
    hook_name_to_sae[sae.cfg.hook_name] = sae
    

print(hook_name_to_sae.keys())


string = "The quick brown fox jumps over the lazy dog."
tokens = model.to_tokens(string)
original_logits = model(tokens)
model.reset_saes()
import copy
l5_sae = hook_name_to_sae[utils.get_act_name('z', 5)]
l5_sae_with_error = copy.deepcopy(l5_sae)
l5_sae_with_error.use_error_term=True
model.add_sae(l5_sae_with_error)
print("Attached SAEs after adding l5_sae_with_error:", model.acts_to_saes)
logits_with_saes = model(tokens)
assert torch.allclose(logits_with_saes, original_logits, atol=1)
print(original_logits.sum())
print(logits_with_saes.sum())
