import pandas as pd
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
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it") 
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
                string = tokenizer.decode(toks.tolist())
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
            all_items = [item for eg_list in topic_items.values() for item in eg_list]
            item_counts = Counter(all_items)
            print(item_counts)
            total_items = len(all_items)
            shannon = 0
            for count in item_counts.values():
                p = count/total_items
                shannon += p*np.log(p)
            shannon = -shannon
            stats_diversity[topic][temp] = shannon

    return stats_dict,stats_diversity




stats_dict, stats_diversity = get_stats(generation_dict)


# Convert stats_diversity to DataFrame
diversity_df = pd.DataFrame.from_dict(
    {topic: {temp: shannon for temp, shannon in temp_dict.items()} for topic, temp_dict in stats_diversity.items()},
    orient='index'

)


# Add an extra row with the variance of the shannon index 
diversity_df.loc["variance"] = diversity_df.var()

# Add an extra col with the mean of the shannon index
diversity_df["mean"] = diversity_df.mean(axis=1)

diversity_df.to_html("diversity_df.html")


# Get the variance in the number of items per example
num_items_variance = pd.DataFrame.from_dict(
    {topic: {temp: np.var([item["num_items"] for item in temp_dict]) for temp, temp_dict in temp_dict.items()} for topic, temp_dict in stats_dict.items()},
    orient='index'
)
num_items_variance.to_html("num_items_variance.html")
