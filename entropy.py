# %%

# %%
from torch.nn.functional import log_softmax
import torch
import pandas as pd

import torch.nn.functional as F
import plotly.express as px




# %%

def logits_to_entropy(logits):
    log_probs = log_softmax(logits, dim = -1)
    probs = log_probs.exp()
    entropy = -(log_probs*probs).sum(dim = -1)
    return entropy


def logits_to_varentropy(logits):
    log_probs = log_softmax(logits, dim = -1)
    probs = log_probs.exp()
    entropy = -(log_probs*probs).sum(dim = -1)
    elem = (probs*(-log_probs)**2).sum(dim = -1)
    return elem - entropy


def logits_to_prob(logits,pos,tok_id1,tok_id2):
    log_probs = log_softmax(logits, dim = -1)
    probs = log_probs.exp()
    tup = [(probs[0,p,tok_id1].item(),probs[0,p,tok_id2].item()) for p in pos]
    return tup

def plot_get_entropy(generation_dict,rep_tok):

    all_entropy_hyphen = []
    tokens = []
    for key in list(generation_dict.keys()):
        val = generation_dict[key]
        for toks in val:
            tokens.append(toks)
            with torch.no_grad():
                hyphen_pos = torch.where(toks[0] == 235290)[0]
                break_pos = torch.where(toks[0] == 108)[0]
                positions = (hyphen_pos[1:]-1).tolist() + [break_pos[-1].item()-2]
                logits = model(toks)
                entropy = logits_to_entropy(logits)
                all_entropy_hyphen.append(entropy[:,positions])



    max_size = max(tensor.size(1) for tensor in all_entropy_hyphen)
    padded_tensors = []
    for tensor in all_entropy_hyphen:
        pad_amount = max_size - tensor.size(1)
        padded_tensor = F.pad(tensor, (pad_amount, 0), "constant", 0)  # Left padding
        padded_tensors.append(padded_tensor)
    stacked_entropy_hyphen = torch.cat(padded_tensors,dim = 0)

    torch.cuda.empty_cache()

    px.imshow(stacked_entropy_hyphen.cpu().numpy(), aspect = 'auto')
    # Plot the entropy for the corruped prompts

    all_entropy_hyphen = []
    tokens = []
    for key in list(generation_dict.keys()):
        val = generation_dict[key]
        for toks in val:
            toks[0,8] = rep_tok
            tokens.append(toks)
            with torch.no_grad():
                hyphen_pos = torch.where(toks[0] == 235290)[0]
                break_pos = torch.where(toks[0] == 108)[0]
                positions = (hyphen_pos[1:]-1).tolist() + [break_pos[-1].item()-2]
                logits = model(toks)
                entropy = logits_to_entropy(logits)
                all_entropy_hyphen.append(entropy[:,positions])


    max_size = max(tensor.size(1) for tensor in all_entropy_hyphen)
    padded_tensors = []
    for tensor in all_entropy_hyphen:
        pad_amount = max_size - tensor.size(1)
        padded_tensor = F.pad(tensor, (pad_amount, 0), "constant", 0)  # Left padding
        padded_tensors.append(padded_tensor)
    stacked_entropy_hyphen = torch.cat(padded_tensors,dim = 0)
    px.imshow(stacked_entropy_hyphen.cpu().numpy(), aspect = 'auto')





if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")

    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt")
    plot_get_entropy(generation_dict, 1497)

    generation_dict = torch.load("generation_dicts/gemma2_generation_long_dict.pt")
    plot_get_entropy(generation_dict, 3309)

