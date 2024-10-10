
import torch
from collections import defaultdict, Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# %%
from  rich.console import Console 
from rich.table import Table 

def display_table(dataframe):
    table = Table(show_header=True, header_style="bold magenta")
    for column in dataframe.columns:
        table.add_column(column)
    
    for index, row in dataframe.iterrows():
        table.add_row(*[str(value) for value in row])
    
    console = Console()
    console.print(table)

def display_heatmap(x,comp):

# Show the features-presence heatmap
    topic_feat_dict = defaultdict(lambda: defaultdict(int))
    for topic, topic_dict in x.items():
        for eg, features_dict in topic_dict.items():
            for pos, feat, layer in features_dict:
                if layer == comp:
                    topic_feat_dict[topic][feat] += 1

    unique_features = sorted(set(feat for topic in topic_feat_dict for feat in topic_feat_dict[topic].keys()))
    heatmap_data = pd.DataFrame(0, index=topic_feat_dict.keys(), columns=unique_features)

    for topic, features in topic_feat_dict.items():
        for feat, count in features.items():
            heatmap_data.at[topic, feat] = 1 if count > 0 else 0

# Sort the heatmap data by the number of active features
    heatmap_data = heatmap_data[heatmap_data.sum(axis=0).sort_values(ascending=False).index]
# Calculate the number of unique features
    num_features = heatmap_data.shape[1]

# Adjust the figure size based on the number of features
    plt.figure(figsize=((12, 8)))  # Width is proportional to the number of features
    sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='black')

    plt.title("Feature Presence Heatmap")
    plt.xlabel("Unique Features")
    plt.ylabel("Topics")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()


# %%

x = torch.load("all_tuples_dict_top_100_item_pos_log_prob_all_attn.pt")


# %%
all_tuples_df = []
topic_features = defaultdict(lambda: defaultdict(set))
topic_positions = defaultdict(lambda: defaultdict(set))

for key, val in x.items():
    for eg_id, tups in val.items():
        for pos, feat, layer in tups:
            if pos != 0:
                topic_features[key][layer].add(feat)
                topic_positions[key][layer].add(pos)

for key in topic_features.keys():
    top_feats = {layer: sorted(features)[:10] for layer, features in topic_features[key].items()}
    top_positions = {layer: sorted(positions)[:10] for layer, positions in topic_positions[key].items()}
    for layer in top_feats.keys():
        all_tuples_df.append({"Topic": key, "Layer": layer, "Top-10 feats": top_feats[layer], "Top-10 positions": top_positions[layer]})

all_tuples_df = pd.DataFrame(all_tuples_df)
all_tuples_df.to_html("all_tuples_df.html")



display_table(all_tuples_df)


for l in [2,7,14,22]:
    display_heatmap(x,f"blocks.{l}.attn.hook_z")



# ======== Expression matrix ======
total_examples = 0 
unique_features = []
for key,val in x.items():
    for eg_id, tups in val.items():
        total_examples += 1
        for pos, feat, layer in tups:
            l = int(layer.split(".")[1])+1
            #if layer == "blocks.2.attn.hook_z":
            unique_features.append(int(str(l)+str(feat)))
unique_features = list(set(unique_features))

count_matrix = torch.zeros((len(unique_features), total_examples))


i = 0
for (key, val) in x.items():
    for eg_id, tups in val.items():
        for pos, feat, layer in tups:
            l = int(layer.split(".")[1])+1
            #if layer == "blocks.2.attn.hook_z":
            count_matrix[unique_features.index(int(str(l)+str(feat))), i] += 1
        i+=1



df = pd.DataFrame(count_matrix/50, columns=range(total_examples), index=unique_features)
# remove nan and inf values
df.fillna(0, inplace=True)
df.replace([float('inf')], 0, inplace=True)

# Create a clustermap with hierarchical clustering
plt.figure(figsize=(14, 10))
sns.clustermap(df, cmap='RdYlGn', standard_scale=1, figsize=(12, 12), cbar_kws={'label': 'Expression Level'})
plt.title('Hierarchical Clustering of Gene Expression Data')
plt.show()
