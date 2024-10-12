from collections import Counter
import torch
from collections import defaultdict
import json
import tqdm




import http.client
import json
import numpy as np
import os

conn = http.client.HTTPSConnection("www.neuronpedia.org")
headers = { 'X-Api-Key': "bfbdaf32-118d-41a6-81db-ee8ee2e030ed" }

# Generate a random sample of 1000 numbers from 0 to 24000 without replacement
np.random.seed(42)




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





# Function to make a request and save the response
def save_feature_data(feature_id,comp,layer):
    try:
        # Skip if file already exists
        file_name = f"{comp}_{layer}_{feature_id}.json"
        if file_name in processed_features:
            print(f"Feature {feature_id} already processed. Skipping...")
            return
        # Make request
        conn.request("GET", f"/api/feature/gemma-2-2b/{layer}-gemmascope-{comp}-16k/{feature_id}", headers=headers)
        res = conn.getresponse()
        
        # Read response data
        data = res.read()
        
        # Check if response is valid
        if res.status != 200:
            raise Exception(f"Request failed with status code {res.status} for feature {feature_id}")
        
        # Parse response data
        data_dict = json.loads(data.decode("utf-8"))
        
        # Save data to a JSON file
        with open(os.path.join(dataset_dir, file_name), "w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        
        print(f"Feature {feature_id} saved successfully.")
    
    except Exception as e:
        # Log the failure in the log file
        with open(log_file, "a") as log:
            log.write(f"Failed to retrieve feature {feature_id}: {str(e)}\n")
        print(f"Failed to retrieve feature {feature_id}. Logged error.")




if __name__ == "__main__":
    generation_dict = torch.load("generation_dicts/gemma2_generation_dict.pt", map_location="cpu")

    hypen_tok_id = 235290
    break_tok_id = 108
    eot_tok_id = 107
    blanck_tok_id = 235248
    layers = [2,7,14,18,22] + [0,5,10,15,20]
    comps = 5*["att"] + 5*["res"]
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


    # Create dataset folder if it doesn't exist
    dataset_dir = "features"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Logging file to track failed requests
    log_file = os.path.join(dataset_dir, "failed_requests.log")

    # Check if a log file exists, otherwise create it
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            pass  # Create an empty log file

    processed_features = set(os.listdir(dataset_dir))

    for key,l in all_features.items():
        for feature_id in l:
            save_feature_data(feature_id,key.split("_")[0],key.split("_")[1])



    # Close the connection
    conn.close()
