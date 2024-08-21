

import re
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np

from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError
from safetensors import safe_open
def get_gemma_2_config(
    repo_id: str,
    folder_name: str,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Dict[str, Any]:
    # Detect width from folder_name
    width_map = {
        "width_16k": 16384,
        "width_32k": 32768,
        "width_65k": 65536,
        "width_131k": 131072,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }
    d_sae = next(
        (width for key, width in width_map.items() if key in folder_name), None
    )
    if d_sae is None:
        if not d_sae_override:
            raise ValueError("Width not found in folder_name and no override provided.")
        d_sae = d_sae_override

    # Detect layer from folder_name
    match = re.search(r"layer_(\d+)", folder_name)
    layer = int(match.group(1)) if match else layer_override
    if layer is None:
        raise ValueError("Layer not found in folder_name and no override provided.")

    # Model specific parameters
    model_params = {
        "2b": {"name": "gemma-2-2b", "d_in": 2304},
        "9b": {"name": "gemma-2-9b", "d_in": 3584},
        "27b": {"name": "gemma-2-27b", "d_in": 4608},
    }
    model_info = next(
        (info for key, info in model_params.items() if key in repo_id), None
    )
    if not model_info:
        raise ValueError("Model name not found in repo_id.")

    model_name, d_in = model_info["name"], model_info["d_in"]

    # Hook specific parameters
    if "res" in repo_id:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "mlp" in repo_id:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "att" in repo_id:
        hook_name = f"blocks.{layer}.attn.hook_z"
        d_in = {"2b": 2048, "9b": 4096, "27b": 4608}.get(
            next(key for key in model_params if key in repo_id), d_in
        )
    else:
        raise ValueError("Hook name not found in folder_name.")

    return {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_layer": layer,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
    }


def gemma_2_sae_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom loader for Gemma 2 SAEs.
    """
    cfg_dict = get_gemma_2_config(repo_id, folder_name, d_sae_override, layer_override)
    cfg_dict["device"] = device

    # Apply overrides if provided
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename="params.npz",
        subfolder=folder_name,
        force_download=force_download,
    )

    # Load and convert the weights
    state_dict = {}
    with np.load(sae_path) as data:
        for key in data.keys():
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                torch.tensor(data[key]).to(dtype=torch.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


# Helper function to get dtype from string
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
import pandas as pd
def get_all_string_min_l0_resid_gemma():
    df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
    resid_dict = df[df['release'] == "gemma-scope-2b-pt-res"]['saes_map'][0]
    splitted_list = [[e.split("_")[-1] for e in elem.split("/")] for elem in list(resid_dict.keys())]
    full_dict = {}
    for elem in splitted_list:
        if elem[1]=="16k":
            if elem[0] not in full_dict.keys():
                full_dict[elem[0]] = {}
                full_dict[elem[0]][elem[1]] = elem[2]
            else:
                if full_dict[elem[0]][elem[1]]>elem[2]:
                    full_dict[elem[0]][elem[1]] = elem[2]
    full_strings = [f"layer_{key}/width_16k/average_l0_{val['16k']}" for key,val in full_dict.items()]
    return full_strings