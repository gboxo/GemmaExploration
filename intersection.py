
"""

def compute_top_k_feature_intersection(model,toks, saes_dict, k:int = 10):
    feature_attribution_df = calculate_feature_attribution(
        model = model,
        input = toks,
        metric_fn = metric_fn,
        include_saes=saes_dict,
        include_error_term=True,
        return_logits=True,
    )
    all_df_dict = defaultdict(dict) 
    for attrb_pos,(tok1,tok2) in zip([46,48],[(blanck_tok_id,break_tok_id),(eot_tok_id,hypen_tok_id)]):
        for i,func in enumerate([metric_fn, metric_fn_log_prob]):
            if i == 0:
                func = partial(func, pos=attrb_pos, tok0 = tok1, tok1 = tok2)
                metric_name = "loggit_diff"
            else:
                func = partial(func, pos=attrb_pos,tok_id = tok1)
                metric_name = "log_prob"
            feature_attribution_df = calculate_feature_attribution(
                model = model,
                input = toks,
                metric_fn = func,
                include_saes=saes_dict,
                include_error_term=True,
                return_logits=True,
            )
            all_tup = []
            for key in saes_dict.keys():
                df_long_nonzero = convert_sparse_feature_to_long_df(feature_attribution_df.sae_feature_attributions[key][0])
                df_long_nonzero.sort_values("attribution", ascending=True)
                df_long_nonzero = df_long_nonzero.nlargest(50, "attribution")
                tuple_list = [(pos,feat) for pos,feat in zip(df_long_nonzero["position"],df_long_nonzero["feature"])]
                all_tup.append(tuple_list)
            all_df_dict[metric_name][tok1] = all_tup
    return all_df_dict

tups0 = all_df_dict["loggit_diff"][blanck_tok_id]
tups1 = all_df_dict["loggit_diff"][eot_tok_id]
tups2 = all_df_dict["log_prob"][blanck_tok_id]
tups3 = all_df_dict["log_prob"][eot_tok_id]

# Compute the proportion of intersection
def compute_proportion_intersection(tups0,tups1,use_position = False):
    intersect01 = []
    for tup0,tup1 in zip(tups0,tups1):
        if use_position:
            intersec = set(tup0).intersection(set(tup1))
        else:
            # just use the feature ids
            intersec = set([feat for pos,feat in tup0]).intersection(set([feat for pos,feat in tup1])) 
        intersect01.append(len(intersec)/len(tup0))
    return np.mean(intersect01)



intersect01 = compute_proportion_intersection(tups0,tups1, use_position = True)
intersect02 = compute_proportion_intersection(tups0,tups2, use_position = True)
intersect03 = compute_proportion_intersection(tups0,tups3, use_position = True)
intersect12 = compute_proportion_intersection(tups1,tups2, use_position = True)
intersect13 = compute_proportion_intersection(tups1,tups3, use_position = True)
intersect23 = compute_proportion_intersection(tups2,tups3, use_position = True)

intersect_mat = np.zeros((4,4))
intersect_mat[0,1] = intersect01
intersect_mat[0,2] = intersect02
intersect_mat[0,3] = intersect03
intersect_mat[1,2] = intersect12
intersect_mat[1,3] = intersect13
intersect_mat[2,3] = intersect23
np.fill_diagonal(intersect_mat, 1)
print(intersect_mat)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(intersect_mat, annot=True)
plt.show()



"""




