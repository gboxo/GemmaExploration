from sae_lens import HookedSAETransformer, SAE, SAEConfig
from gemma_utils import get_gemma_2_config, gemma_2_sae_loader
import numpy as np
import torch
from tqdm import tqdm
import einops
import re
from jaxtyping import Int, Float
from typing import List, Optional, Any
from torch import Tensor
import json
import os
from torch.utils.data import Dataset, DataLoader
import random
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from transformer_lens.utils import get_act_name
from IPython.display import display, HTML
import plotly.express as px







def sample(model,input_ids):
    tokens = model.to_tokens(input_ids, prepend_bos=False)
    out = model.generate(
        tokens,
        max_new_tokens = 100,
        temperature = 0.7,
        top_p = 0.8,
        stop_at_eos=True,
        verbose=False,

        )
    torch.cuda.empty_cache()
    return out


def generate_lists(topics):
    generation_dict = {}

    for topic in tqdm(topics):
        generation_dict[topic] = [] 
        for _ in range(5):
            messages = [
                {"role": "user", "content": f"Provide me with a long list of a many {topic}. Just provide the names, no need for any other information."},
            ]
            input_ids = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            input_ids += "-"
            out = sample(model,input_ids)

            toks = out.detach().clone()
            generation_dict[topic].append(toks)
    return generation_dict

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")

    topics = ["City Names","Countries","Animals","Types of Trees","Types of Flowers",
    "Fruits","Vegetables","Car Brands","Sports","Rivers","Mountains","Ocean",
    "Inventions","Languages","Capital Cities","Movies","Books","TV Shows",
    "Famous Scientists","Famous Writers","Video Games","Companies","Colors"]
    generation_dict = generate_lists(topics)
    torch.save(generation_dict, "gemma2_generation_long_dict.pt")
