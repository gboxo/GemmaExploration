from sae_lens import HookedSAETransformer, SAE, SAEConfig
import torch
from tqdm import tqdm







def sample(model,input_ids,temperature):
    tokens = model.to_tokens(input_ids, prepend_bos=False)
    out = model.generate(
        tokens,
        max_new_tokens = 100,
        temperature = temperature,
        top_p = 0.8,
        stop_at_eos=True,
        verbose=False,

        )
    torch.cuda.empty_cache()
    return out


def generate_lists(topics):
    generation_dict = {}
    


    for temp in tqdm([0.1,0.5,0.7,0.9,1.3]):
        generation_dict[temp] = {}
        for topic in topics:
            generation_dict[temp][topic] = [] 
            for _ in range(5):
                messages = [
                    {"role": "user", "content": f"Provide me with a short list of {topic}. Just provide the names, no need for any other information."},
                ]
                input_ids = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                input_ids += "-"
                out = sample(model,input_ids,temp)

                toks = out.detach().clone()
                generation_dict[temp][topic].append(toks)
    return generation_dict

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b-it")

    topics = ["City Names","Countries","Animals","Types of Trees","Types of Flowers",
    "Fruits","Vegetables","Car Brands","Sports","Rivers","Mountains","Ocean",
    "Inventions","Languages","Capital Cities","Movies","Books","TV Shows",
    "Famous Scientists","Famous Writers","Video Games","Companies","Colors"]
    generation_dict = generate_lists(topics)
    torch.save(generation_dict, "gemma2_generation_temps_dict.pt")
