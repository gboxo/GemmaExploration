# Feature Attribution
from sae_lens import HookedSAETransformer,SAE
import torch
from dataclasses import dataclass
from functools import partial
from typing import Any,Literal, NamedTuple, Callable
from transformer_lens.hook_points import HookPoint



class SaeReconstructionCache(NamedTuple):
    sae_in: torch.Tensor
    feature_acts: torch.Tensor
    sae_out: torch.Tensor
    sae_error: torch.Tensor



def track_grad(tensor: torch.Tensor) -> None:
    tensor.requires_grad_(True)
    tensor.retain_grad()





@dataclass
class ApplySaesAndRunOutput:
    model_output: torch.Tensor
    model_activations: dict[str,torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]

    def zero_grad(self) -> None:
        """helper to zero grad all tensors in this object"""
        self.model_output.grad = None
        for act in self.model_activations.values():
            act.grad = None
        for cache in self.sae_activations.values():
            cache.sae_in.grad = None
            cache.feature_acts.grad = None
            cache.sae_out.grad = None
            cache.sae_error.grad = None
    def to(self, device: str) -> None:
        """helper to move all tensors in this object to a device"""
        self.model_output = self.model_output.to(device)
        for act in self.model_activations.values():
            act = act.to(device)
        for cache in self.sae_activations.values():
            cache.sae_in.to(device)
            cache.feature_acts.to(device)
            cache.sae_out.to(device)
            cache.sae_error.to(device)

def apply_saes_and_run(
        model: HookedSAETransformer,
        saes: dict[str,SAE],
        input:Any,
        include_error_term: bool = True,
        track_model_hooks: list[str] | None = None,
        return_type: Literal["logits","loss"] = "logits",
        track_grads: bool = False

        ) -> ApplySaesAndRunOutput:
    fwd_hooks = []
    bwd_hooks = []
    sae_activations: dict[str, SaeReconstructionCache] = {}
    model_activations:  dict[str, torch.Tensor] = {}


    def reconstruction_hook(sae_in: torch.Tensor, hook: HookPoint, hook_point: str):
        sae = saes[hook_point]
        feature_acts = sae.encode(sae_in)
        sae_out = sae.decode(feature_acts)

        sae_error = (sae_in - sae_out).detach().clone()
        if track_grads:
            track_grad(sae_error)
            track_grad(sae_out)
            track_grad(feature_acts)
            track_grad(sae_in)
        sae_activations[hook_point] = SaeReconstructionCache(
                sae_in = sae_in,
                feature_acts = feature_acts,
                sae_out = sae_out,
                sae_error = sae_error

                )
        if include_error_term:
            return sae_out + sae_error
        else:
            return sae_out
    def sae_bwd_hook(output_grads: torch.Tensor, hook:HookPoint):
        return (output_grads,)
    def tracking_hook(hook_input: torch.Tensor, hook:HookPoint, hook_point: str):
        model_activations[hook_point] = hook_input
        if track_grads:
            track_grad(hook_input)
        return hook_input

    for hook_point in saes.keys():
        fwd_hooks.append(
                (hook_point,partial(reconstruction_hook, hook_point= hook_point))
                )
        bwd_hooks.append((hook_point,sae_bwd_hook))
    for hook_point in track_model_hooks or []:
        fwd_hooks.append((hook_point,partial(tracking_hook, hook_point=hook_point)))
    # run the model while applying the hooks

    with model.hooks(fwd_hooks = fwd_hooks, bwd_hooks = bwd_hooks):
        model_output = model(input, return_type = return_type)

    return ApplySaesAndRunOutput(
            model_output= model_output,
            model_activations = model_activations,
            sae_activations = sae_activations
            ) 




EPS = 1e-8

torch.set_grad_enabled(True)
@dataclass
class AttributionGrads:
    metric: torch.Tensor
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]




@dataclass
class Attribution:
    model_attributions: dict[str, torch.Tensor]
    model_activations: dict[str, torch.Tensor]
    model_grads: dict[str, torch.Tensor]
    sae_feature_attributions: dict[str, torch.Tensor]
    sae_feature_activations: dict[str,torch.Tensor]
    sae_feature_grads: dict[str, torch.Tensor]
    sae_errors_attribution_proportion: dict[str, float]




def calculate_attribution_grads(
    model: HookedSAETransformer,
    prompt: str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> AttributionGrads:
    """
    Wrapper around apply_saes_and_run that calculates gradients wrt to the metric_fn.
    Tracks grads for both SAE feature and model neurons, and returns them in a structured format.
    """
    output = apply_saes_and_run(
        model,
        saes=include_saes or {},
        input=prompt,
        return_type="logits" if return_logits else "loss",
        track_model_hooks=track_hook_points,
        include_error_term=include_error_term,
        track_grads=True,
    )
    metric = metric_fn(output.model_output)
    #gradients = torch.ones_like(metric)
    output.zero_grad()
    metric.backward()
    output.to("cpu")
    #metric.backward(gradients)
    return AttributionGrads(
        metric=metric,
        model_output=output.model_output,
        model_activations=output.model_activations,
        sae_activations=output.sae_activations,
    )


def calculate_feature_attribution(
    model: HookedSAETransformer,
    input: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> Attribution:

    outputs_with_grads = calculate_attribution_grads(
        model,
        input,
        metric_fn,
        track_hook_points,
        include_saes=include_saes,
        return_logits=return_logits,
        include_error_term=include_error_term,
    )
    model_attributions = {}
    model_activations = {}
    model_grads = {}
    sae_feature_attributions = {}
    sae_feature_activations = {}
    sae_feature_grads = {}
    sae_error_proportions = {}
    # this code is long, but all it's doing is multiplying the grads by the activations
    # and recording grads, acts, and attributions in dictionaries to return to the user
    with torch.no_grad():
        for name, act in outputs_with_grads.model_activations.items():
            assert act.grad is not None
            raw_activation = act.detach().clone()
            model_attributions[name] = (act.grad * raw_activation).detach().clone()
            model_activations[name] = raw_activation
            model_grads[name] = act.grad.detach().clone()
        for name, act in outputs_with_grads.sae_activations.items():
            assert act.feature_acts.grad is not None
            assert act.sae_out.grad is not None
            raw_activation = act.feature_acts.detach().clone()
            sae_feature_attributions[name] = (
                (act.feature_acts.grad * raw_activation).detach().clone()
            )
            sae_feature_activations[name] = raw_activation
            sae_feature_grads[name] = act.feature_acts.grad.detach().clone()
            if include_error_term:
                assert act.sae_error.grad is not None
                error_grad_norm = act.sae_error.grad.norm().item()
            else:
                error_grad_norm = 0
            sae_out_norm = act.sae_out.grad.norm().item()
            sae_error_proportions[name] = error_grad_norm / (
                sae_out_norm + error_grad_norm + EPS
            )
        return Attribution(
            model_attributions=model_attributions,
            model_activations=model_activations,
            model_grads=model_grads,
            sae_feature_attributions=sae_feature_attributions,
            sae_feature_activations=sae_feature_activations,
            sae_feature_grads=sae_feature_grads,
            sae_errors_attribution_proportion=sae_error_proportions,
        )
