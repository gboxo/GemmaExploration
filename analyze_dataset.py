
import numpy as np
import torch
import tqdm
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
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from collections import Counter


model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")
generation_dict = torch.load("gemma2_generation_temps_dict.pt", map_location="cpu")


hypen_tok_id = 235290
break_tok_id = 108
eot_tok_id = 107
blanck_tok_id = 235248




def get_stats(dict_toks):
    """
    For each topic create a dictionary of stats
    For each generated list get:
    - Number of tokens
    - Number of items in the list
    - Average number of tokens per item
    - Item positions in which blank tokens are foung
    """
    stats_dict = {}
    stats_diversity = {}
    for topic, temp_dict in dict_toks.items():
        stats_dict[topic] = {}
        stats_diversity[topic] = {}
        for temp,tok_list in temp_dict.items():
            stats_dict[topic][temp] = []
            topic_items = {} 
            for i,toks in enumerate(tok_list):
                toks = toks.squeeze()
                string = model.to_string(toks)
                items = [item for item in string.split("\n") if item.startswith("-")]
                topic_items[i] = items
                hypen_positions = torch.where(toks == hypen_tok_id)[0].to("cpu")
                break_positions = torch.where(toks == break_tok_id)[0].to("cpu")
                eot_positions = torch.where(toks == eot_tok_id)[0].to("cpu")
                filter_break_pos = [pos.item() for pos in break_positions if pos+1 in hypen_positions]
                topic_spans = [(hypen_positions[i].item(),hypen_positions[i+1].item()) for i in range(len(hypen_positions)-1)] +[(hypen_positions[-1].item(),eot_positions[-1].item())]
                token_spans = []
                for span in topic_spans:
                    token_spans.append(toks[span[0]:span[1]].tolist())
                num_items = len(token_spans)
                number_of_tokens_per_item = torch.tensor([len(span) for span in token_spans])
                white_space_tok = torch.tensor([235248 in tok_span for tok_span in token_spans])
                white_spaces_tok_pos = torch.where(white_space_tok)[0].to("cpu")

                stats_dict[topic][temp].append({"num_tokens": number_of_tokens_per_item, "num_items": num_items, "avg_tokens_per_item": number_of_tokens_per_item, "blank_positions": white_spaces_tok_pos})
            # Compute the shannon idex for each item in the examples
            print(topic_items)
            for i,items in topic_items.items():
                item_counts = Counter(items)
                total_items = len(items)
                shannon = 0
                for count in item_counts.values():
                    p = count/total_items
                    shannon += p*np.log(p)
                shannon = -shannon
                stats_diversity[topic][temp] = shannon

    return stats_dict,stats_diversity




stats_dict, stats_diversity = get_stats(generation_dict)
