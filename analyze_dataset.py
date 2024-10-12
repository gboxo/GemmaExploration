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
                repeated_toks = len(items)-len(set(items))
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

                stats_dict[topic][temp].append({"num_tokens": number_of_tokens_per_item, "num_items": num_items, "avg_tokens_per_item": number_of_tokens_per_item, "blank_positions": white_spaces_tok_pos, "repeated_toks": repeated_toks})
            # Compute the shannon idex for each item in the examples
            all_items = [item for eg_list in topic_items.values() for item in eg_list]
            item_counts = Counter(all_items)
            total_items = len(all_items)
            shannon = 0
            for count in item_counts.values():
                p = count/total_items
                shannon += p*np.log(p)
            shannon = -shannon
            stats_diversity[topic][temp] = shannon

    return stats_dict,stats_diversity



def get_diff_div(stats_diversity, stats_diversity_ablation):
    diversity_df_ablation = pd.DataFrame.from_dict(
        {topic: {temp: shannon for temp, shannon in temp_dict.items()} for topic, temp_dict in stats_diversity_ablation.items()},
        orient='index'
    )

    diversity_df = pd.DataFrame.from_dict(
        {topic: {temp: shannon for temp, shannon in temp_dict.items()} for topic, temp_dict in stats_diversity.items()},
        orient='index'
    )

    diversity_df = diversity_df.apply(pd.to_numeric, errors='coerce')
    diversity_df_ablation = diversity_df_ablation.apply(pd.to_numeric, errors='coerce')
    diversity_diff = diversity_df_ablation.to_numpy() - diversity_df.to_numpy()          
    diversity_diff = pd.DataFrame(diversity_diff, index=diversity_df.index, columns=diversity_df.columns)
    diversity_diff['diversity_variance'] = diversity_diff.var(axis=1)
    diversity_diff["diversity_mean"] = diversity_diff.mean(axis=1)
    diversity_diff = diversity_diff[["diversity_mean", "diversity_variance"]]

    return diversity_diff

def get_diff_stats(stats_dict, stats_dict_ablation):
    num_items_stats = pd.DataFrame.from_dict(
            {topic: {eg_id: eg_stats[0]["num_items"] for eg_id, eg_stats in eg_dict.items()} 
                       for topic, eg_dict in stats_dict.items()},
        orient='index'
    )

    num_items_stats_ablation = pd.DataFrame.from_dict(
            {topic: {eg_id: eg_stats[0]["num_items"] for eg_id, eg_stats in eg_dict.items()} 
                       for topic, eg_dict in stats_dict_ablation.items()},
        orient='index'
    )
    num_items_stats_ablation_diff = num_items_stats_ablation.to_numpy() - num_items_stats.to_numpy()          
    num_items_stats_ablation_diff = pd.DataFrame(num_items_stats_ablation_diff, index=num_items_stats_ablation.index, columns=num_items_stats_ablation.columns)
     # Compute the mean and variance of the difference
    num_items_stats_ablation_diff["n_items_variance"] = num_items_stats_ablation_diff.var(axis=1)
    num_items_stats_ablation_diff["n_items_mean"] = num_items_stats_ablation_diff.mean(axis=1)
    # Just keep the mean and variance columns
    num_items_stats_ablation_diff = num_items_stats_ablation_diff[["n_items_mean", "n_items_variance"]]


    return num_items_stats_ablation_diff



if __name__ == "__main__":
    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it") 
    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt", map_location="cpu")
    generation_dict = {topic: {temp: gen_list for temp, gen_list in enumerate(temp_dict)} for topic, temp_dict in generation_dict.items()}

    stats_dict, stats_diversity = get_stats(generation_dict)

    layers = [2,7,14,18,22]+[0,7]
    comps = 5*["attn"]+2*["res"]
    all_results = {} 

    for layer, comp in zip(layers,comps):

        with open(f"generation_dicts/gemma2_generation_dict_ablation_{comp}_layer_{layer}.json", "r") as f:
            gen_with_ablation = json.load(f)

        gen_with_ablation = {topic: {temp: tokenizer.encode([gen_list[0] if type(gen_list)== list else gen_list][0],return_tensors = "pt") for temp, gen_list in temp_dict.items()} for topic, temp_dict in gen_with_ablation.items() }
        stats_dict_ablation, stats_diversity_ablation = get_stats(gen_with_ablation)

        diversity_diff = get_diff_div(stats_diversity, stats_diversity_ablation)
        num_items_stats_ablation_diff = get_diff_stats(stats_dict, stats_dict_ablation)
        df = pd.concat([diversity_diff, num_items_stats_ablation_diff], axis=1)
        all_results[f"{comp}_layer_{layer}"] = df

    all_results_conc = pd.concat(all_results)
    # Convert the first index into a column
    all_results_conc.reset_index(inplace=True)
    all_results_conc.set_index("level_0", inplace=True)
    avg_div = all_results_conc.groupby("level_1").mean()
    avg_div.to_html("tables/all_ablation_diff_avg_comp.html")

    all_results_conc = pd.concat(all_results)
    all_results_conc.reset_index(inplace=True)
    all_results_conc.set_index("level_1", inplace=True)
    avg_div = all_results_conc.groupby("level_0").mean()
    avg_div.to_html("tables/all_ablation_diff_avg_topic.html")


