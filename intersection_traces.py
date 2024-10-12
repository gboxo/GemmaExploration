

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
# %% 




def display_heatmap(x, comp):
    topic_feat_dict = defaultdict(lambda: defaultdict(float))
    for topic, topic_dict in x.items():
        for eg, features_dict in topic_dict.items():
            if comp in features_dict:
                indices = features_dict[comp].indices()[1]

                values = features_dict[comp].values()
                ind = values.argsort(descending=True)[:300]
                indices = indices[ind]
                values = values[ind]
                for i, feat in enumerate(indices):
                    topic_feat_dict[topic][feat.item()] += values[i]

    unique_features = sorted(set(feat for topic in topic_feat_dict for feat in topic_feat_dict[topic].keys()))
    heatmap_data = pd.DataFrame(0, index=topic_feat_dict.keys(), columns=unique_features, dtype = float)

    for topic, features in topic_feat_dict.items():
        for feat, count in features.items():
            heatmap_data.at[topic, feat] = count.item() if count > 0 else 0

    heatmap_data = heatmap_data[heatmap_data.sum(axis=0).sort_values(ascending=False).index]
    heatmap_data = heatmap_data.iloc[:, 20:300]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data.T, annot=False, cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='white')

    plt.title("Feature Presence Heatmap")
    plt.xlabel("Unique Features")
    plt.ylabel("Topics")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# %%

x = torch.load("traces/all_max_traces_dict.pt", map_location="cpu")






# %%
topic_features = defaultdict(lambda: defaultdict(list))
topic_act = defaultdict(lambda: defaultdict(list))
count_features = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for key, val in x.items():
    for eg_id, eg_dict in val.items():
        all_features = []
        for comp, tensors in eg_dict.items(): 
            indices = tensors.indices()[1]
            values = tensors.values()
            topic_features[key][comp] = indices.tolist()
            topic_act[key][comp] = values.tolist()
            all_features.extend(indices.tolist())
        for feat in all_features:
            count_features[key][comp][feat] += 1

all_tuples = [] 
for topic,topic_dict in count_features.items():
    for comp, comp_dict in topic_dict.items():
        ind = torch.tensor(list(comp_dict.keys()))
        val = torch.tensor(list(comp_dict.values()))
        top_ind = val.argsort(descending=True)[5:10]
        top_feat = ind[top_ind]
        all_tuples.append({"Topic": topic, "Component": comp, "Top-5 feats": top_feat.tolist(), "Top-5 counts": val[top_ind].tolist()})


all_tuples_df = pd.DataFrame(all_tuples)
all_tuples_df.to_html("tables/all_tuples_df.html")



display_table(all_tuples_df)


for l in [0]:
    display_heatmap(x,f"blocks.{l}.hook_resid_post")



def generate_expression_matrix(x):
    total_examples = 0 
    unique_features = []
    for key, val in x.items():
        for eg_id, eg_dict in val.items():
            for comp, tensor in eg_dict.items(): 
                total_examples += 1
                values = tensor.values()
                indices = tensor.indices()[1]
                ind = values.argsort(descending=True)[:300]
                indices = indices[ind]
                values = values[ind]
                unique_features.extend(feat.item() for feat in indices)
    unique_features = list(set(unique_features))

    count_matrix = torch.zeros((len(unique_features), total_examples))

    for i, (key, val) in enumerate(x.items()):
        for eg_id, eg_dict in val.items():
            for comp, tensor in eg_dict.items(): 
                values = tensor.values()
                indices = tensor.indices()[1]
                ind = values.argsort(descending=True)[:300]
                indices = indices[ind]
                values = values[ind]
                for j, feat in enumerate(indices):
                    count_matrix[unique_features.index(feat.item()), i] = values[j]

    df = pd.DataFrame(count_matrix / 50, columns=range(total_examples), index=unique_features)
    df.fillna(0, inplace=True)
    df.replace([float('inf')], 0, inplace=True)

    return df

df = generate_expression_matrix(x)

plt.figure(figsize=(14, 10))
sns.clustermap(df.T, cmap='RdYlGn', standard_scale=1, figsize=(12, 12), cbar_kws={'label': 'Expression Level'})
plt.title('Hierarchical Clustering of Gene Expression Data')
plt.show()
