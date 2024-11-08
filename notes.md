## Mechanistic Exploration of Gemma 2 2b list creation


**This post is actively being worked on, full post out by 15 Oct**



### 1. Introduction

Gemma 2 2b is a Small Language Model, created by google that was made public in Summer 2024. This model was released along a suite of Sparse Autoenocders and Transcoders trained on their activations in various locations.


This model, despite it's size is incredible capable, excelling in it's instruction tuned vairant with the ability to follow instruction and with similar performance to the original GPT-3 model on some benchmarks.


This facts make Gemma 2 2b a great candidate to perform MI experiments on.

In this post we will explore the mechanisms behind Gemma 2 2b's ability to create lists of items, when prompted for.

Specially we are interested in the mechanism by which Gemma knows when to end a list, this is task is interesting for the following reasons:

- Due to the larger Gemma 2 vocabulary is eassy to create one token per item list templates.
- The instruction tuning of the model, enables the induction of different behaviors in the model with minimal 
- The open endedness of this task enables taking into account sampling dynamics on the decoding process of the model (this is temperature and sampling method).
- The template structure enables a clear analysis of a priori very broad properties of the model like "list ending behavior" by proxies such as the probability of outputing a hypen after a list item which clearly indicates that the list is about to continue.



### 2.Data

To investigate the behavior of Gemma when asked for a list we create a synthetic dataset.


1) We ask GPT4-o to provide a list of topics to create lists about.
2) we create a prompt, for Gemma:
3) For each topic, we sample 10 Gemma Completions with top-k = 0.9 and temeprature=0.8

```
Provide me  a with short list of {topic}. Just provide the names, no need for any other information.
```

*This step is crucial, because as can be seen in the Apendix actually sampling completions from Gemma allow us to observe the actual dynamics of the list ending behavior when compared with List Gnerated by GPT4o*

4) Finally for all the completions generated by Gemma, we construct contrastive prompts.

```
Provide me with a long list of {topic}. Just provide the names, no need for any other information.
```

This contrastive pairs are shown to not have the same list ending behaviors which enable patching style Interpretabiliy without too much shift in behavior unrelated activations.



### 3. Exploratory Analysis

We start the Analysis with an exploratory overview of the completions provided by the model.
Few things that struck me as interesting:





1) Most of the item's in the list where a single token.

<p align="center">
  <img src="image/Token-Statistics-Temp.png" alt="Token Statistics Temp" />
</p>

For our prompt template, and with a few and notable exceptions most of the items in the lists where one-token long, this is likely a result of the expanded vocabulary size of Gemma 2 (roughly 5 times bigger than the one from GPT2 models),it's also probably an artifact of the postraining procedure.

Some notable exceptions where topics like Oceans, Cities or Countries.


2) For all the topics, in the last few items in the list the model sampled white space tokens after the item and before the line break.


<p align="center">
  <img src="image/Token-Statistics-Temp-blank.png" alt="Token Statistics Temp Blank" />
</p>


By far the most interesting behavior that we've observed trough different topics and temperatures is the model behavior of including blank tokens near the end of the list.
We further investigate this behavior using attribution techniques.

3) The number of items in each list with a white space added is pretty consistent across topics, with a few outliers.



4) The number of items in each list is also very similar across topics.
5) There exist a correlation between the sampling temperature and the number of items in a list with a blank spaces token before the end of the list.
<p align="center">
  <img src="image/Fraction-Blankcs.png" alt="Fraction Blankcs" />
</p>
6) For prompts, where we asked for a long list, the average number of items is 30, and we no longer observe an abudance of white space tokens at the end of the list.
7) The entropy of the logits at the item position increases with the items; this however is not the case if we use the corrupt prompt (replace the token " short" with " long" in the prompt+generation (under short)) 


### 3.5. Entropy Analysis

When doing Mechansitic Interpretability analysis on Model Generated Outputs is very important to remember that the distribution of outputs for the model is not the same as the distributio of dataset examples that one could generate.

This is one of the possible explanations behind the phenomena of AI Models prefering their own text to web text.

One of the possible exploratory metrics that we can analyze is the evolution of the entropy over the positions.

We concretely focus on the entropy at the item positions over the whole dataset.


For the generated outputs of the model for the base prompt we have the following entropy plot.



<p align="center">
  <img src="image/Entropy-item-last-tok-short-clean.png" alt="Entropy Item Last Token Short Clean" />
</p>


We can clearly see how the entropy increase is gradual trough out the item positions, with a slight increase at the last positions.

If we corrupt this initial prompts by interchanging the token " short" with " long" and leaving everything else (including the generate list) intact.


<p align="center">
  <img src="image/Entropy-item-last-tok-short-corrupted.png" alt="Entropy Item Last Token Short Corrupted" />
</p>

We observe that there's no increase in entropy over the item positions, the entropy just increases at the last position.

We can also do the opposite analysis, we can generate a dataset of completions for prompts asking for long lists.


<p align="center">
  <img src="image/Entropy-item-last-tok-long-corrupted.png" alt="Entropy Item Last Token Long Corrupted" />
</p>

Similar to the case of the clean short prompt we see a gradual increase in entropy.

If we corrupt this prompts by interchanging the token " long" with " short" and leaving everything else (including the generated list) intact.

<p align="center">
  <img src="image/Entropy-item-last-tok-long-clean.png" alt="Entropy Item Last Token Long Clean" />
</p>

Similar story to the first case.







### 4. Logit Lens


If we investigate tre logit difference between the end_of_turn and the hyphen tokens (which we can call the list ending behavior) we can see in this individual example that the behavior starts to surface around the 3rd item (however thos dofference is overcome by the temperature).


<p align="center">
  <img src="image/Logit-Diff-Tokens.png" alt="Logit Diff Tokens" />
</p>

Taking advantage of the shared RS across layers we can use the unembedding matrix to project into vocabulary space activations across the layers, to get an intuition of how a behavior builds trough the layers.



Using such techniques we inspect the relevant positions for the list ending behavior across the dataset.

We use the difference between decoder <end_of_turn> and "-" direction as the list ending direction, to inspect the activations trough the layers.

This important locations are the various positions in which the list could have ended, this is the line_break positions.

Inspecting the different location Logit Lens, we see a shared trend of increasing logit_difference across later items in the list and layers.

Being layer 18 to 26 and the list items with white space tokens the breaking points for the most difference in logit differnce layer and item wise.



Taking advantage of the fact that we've found out that the model exhibits a tendency of including blank tokens near the end of the list we can investigate the evolution of the logits trought two lens:

1) The direction of predicting a blank space token vs a line break token at the item token
2) The direction of predicting a hyphen token vs the end_of_turn token.

<p align="center">
  <img src="image/Logit-Lens-Positions.png" alt="Logit Lens Positions" />
</p>

For the first case we ovbserve over the item positions, that most of the differentiation is in the last layers.



<p align="center">
  <img src="image/Logit-Lens-Positions2.png" alt="Logit Lens Positions 2" />
</p>


It's similar to the above case but the behavior appears earlier in the layers, and there's a gradual build-up trough the items in the list.



### 5. Activation Patching

One of the nicest properties of this setup is that due to the instruction tuning of the model is very easy to induces certain model behaviors with minimal changes in the prompt.


In this case just changing the tokens "short list with a few" with "long list with a many" make the model output much longer list, and hence display different list ending behaviors for a given prompt.


This enables easy creation of the contrastive templates.



With this easy trick we are in an very similar situation (while not perfectly analogous ) to the IOI paper.

This enable simple patching experiments to identify crucial model components for the list ending behavior to occur.

- Patchings:
    - Residual stream
    - Key, Value, Query
    - MLP output
    - Attention score


<p align="center">
  <img src="image/Attention-Key-Patching.png" alt="Attention Key Patching" />
</p>

<p align="center">
  <img src="image/Attention-Value-Patching.png" alt="Attention Value Patching" />
</p>

<p align="center">
  <img src="image/Attention-Output-Patching.png" alt="Attention Output Patching" />
</p>

<p align="center">
  <img src="image/Attention-Query-Patching.png" alt="Attention Query Patching" />
</p>





### 6. Feature Attribution

To bound the complexity of using Sparse Autoencoders, to investigate the model behavior we will focus on residual stream features.

One problem that is apparent to anyone that has tried to use SparseAutoencoders for real world task is that the memory footprint of SAE experiments rapidly explodes as we add layers.

The intuitive solution to this problem is to come up with heuristics to select a few layers to use SAEs in, to maximize the faithfulness/GB vram ratio.

Possible heuristics are:

- Use Logit Lens style techniques to select the most important layers, by 
- Use ablation over positions and layers to select the most important layers. (Some discounting should be done to no give too much importance to the last position or last layers)



- Attribution w.r.t to the logit differnce
- Attribution w.r.t to the -log prob
- Feature attribution (most important later layer features) (If I have enough time)




### 7. Feature visualization in the dataset

- For each topic select a number of features across the layers, that when ablated modifies the list ending behavior.
- Display the activation of this features across the dataset, include contrasitve pairs.
- How much overlap is there between topics


<p align="center">
  <img src="image/Feat-16323.png" alt="Feature 16323" />
</p>


<p align="center">
  <img src="image/Head-Output-Attribution-Patching.png" alt="Head Output Attribution Patching" />
</p>

<p align="center">
  <img src="image/Intersection-Heatmap.png" alt="Intersection Heatmap" />
</p>


### 8. Causal ablation of RS features over the layers


### Appendix

**Item break vs White space Locations**

**Should we divide Logit Lens by temperature**

**Heuristic for selecting the layers**

