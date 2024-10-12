from collections import Counter
import torch
from collections import defaultdict
import json
import tqdm





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


    all_features = defaultdict(dict)
    tuples = torch.load(f"tuples/all_tuples_dict_top_1000_item_pos_logit_diff_{comp}_{layer}.pt",map_location="cpu")
    for topic, topic_dict in generation_dict.items():
        for eg, toks in enumerate(topic_dict):
            toks = toks.squeeze()
            tup = tuples[topic][eg][0]
            attrb_pos = get_attrb_pos(toks)

            tup = [(elem[0],elem[1]) for elem in tup if elem[0] < attrb_pos and elem[0]>0]
            tup = tup[:10]
            ablation_feature_by_layer_pos = {layer:{"Features":[elem[1] for elem in tup], "Positions":[elem[0] for elem in tup]}}
            all_features[topic][eg] = ablation_feature_by_layer_pos            
    return all_features






if __name__ == "__main__":
    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt", map_location="cpu")

    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248
    layers = [2,7,14,18,22] + [0,5,10,15,20]
    comps = 5*["attn"] + 5*["res"]
    all_features = {} 
    for layer, comp in tqdm.tqdm(zip(layers,comps)):
        pos_feat = get_layer_comp(generation_dict,layer, comp)
        all_feats = []
        for key,val in pos_feat.items():
            for k,v in val.items():
                all_feats.extend(list(v.values())[0]["Features"])
        occurence_count = Counter(all_feats)
        occurence_count = [k for k,v in occurence_count.items()]
        all_features[comp+"_"+str(layer)] = occurence_count




