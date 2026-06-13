"""TiraModel: finds target Linear modules and replaces them with TiraLinear.

Follows hira's LoraModel / CLRA's ClraModel pattern:
- _find_and_replace: iterates named_modules, replaces matching nn.Linear
- _replace_module: swaps in new module, copies weight/bias
- merge_and_unload: folds ΔW into base weights, returns plain model
- enable/disable adapter layers
"""
import re
import warnings
from dataclasses import asdict
from enum import Enum

import torch
import torch.nn as nn

from hira.utils import (
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)

from .config import TiraConfig
from .layer import TiraLayer, TiraLinear


def mark_only_tira_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all parameters except tira_ adapter parameters."""
    for n, p in model.named_parameters():
        if "tira_" not in n:
            p.requires_grad = False
    if bias == "none":
        pass
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "tira_only":
        for m in model.modules():
            if isinstance(m, TiraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Unsupported bias type: {bias}")


class TiraModel(torch.nn.Module):
    """Manages injection of TiraLinear adapters into a base model.

    Mirrors hira's LoraModel / CLRA's ClraModel interface:
    - Constructed with (model, peft_config_dict, adapter_name)
    - self.model holds the modified base model
    - self.forward delegates to self.model.forward
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        mark_only_tira_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        config: TiraConfig = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(config.target_modules, str):
                target_module_found = re.fullmatch(config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key) for target_key in config.target_modules
                )

            if target_module_found:
                if key.endswith("q_proj"):
                    module_M = config.tira_q_M if config.tira_q_M is not None else config.tira_M
                    module_K = config.tira_q_K if config.tira_q_K is not None else config.tira_K
                elif key.endswith("k_proj"):
                    module_M = config.tira_k_M if config.tira_k_M is not None else config.tira_M
                    module_K = config.tira_k_K if config.tira_k_K is not None else config.tira_K
                elif key.endswith("v_proj"):
                    module_M = config.tira_v_M if config.tira_v_M is not None else config.tira_M
                    module_K = config.tira_v_K if config.tira_v_K is not None else config.tira_K
                elif key.endswith("o_proj"):
                    module_M = config.tira_o_M if config.tira_o_M is not None else config.tira_M
                    module_K = config.tira_o_K if config.tira_o_K is not None else config.tira_K
                elif key.endswith("up_proj"):
                    module_M = config.tira_up_M if config.tira_up_M is not None else config.tira_M
                    module_K = config.tira_up_K if config.tira_up_K is not None else config.tira_K
                elif key.endswith("down_proj"):
                    module_M = config.tira_down_M if config.tira_down_M is not None else config.tira_M
                    module_K = config.tira_down_K if config.tira_down_K is not None else config.tira_K
                else:
                    module_M = config.tira_M
                    module_K = config.tira_K
                module_alpha = config.tira_alpha if config.tira_alpha is not None else module_K
                kwargs = {"M": module_M, "K": module_K, "alpha": module_alpha}

                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)

                if isinstance(target, TiraLayer):
                    # Already a TIRA layer, add new adapter
                    target.update_layer(adapter_name, **kwargs)
                elif isinstance(target, torch.nn.Linear):
                    # Plain nn.Linear → replace with TiraLinear
                    in_features, out_features = target.in_features, target.out_features
                    bias = target.bias is not None
                    new_module = TiraLinear(
                        adapter_name,
                        in_features,
                        out_features,
                        bias=bias,
                        **kwargs,
                    )
                    self._replace_module(parent, target_name, new_module, target)
                else:
                    raise ValueError(
                        f"Target module {target} is not supported. "
                        f"Currently, only `torch.nn.Linear` is supported."
                    )

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        # Copy original weight and bias
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # Dispatch adapter sub-modules to correct device
        for name, module in new_module.named_modules():
            if "tira_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, TiraLayer):
                module.disable_adapters = not enabled

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, TiraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, TiraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, TiraLayer):
                module.unmerge()

    def merge_and_unload(self):
        """Fold all adapter ΔW into base weights and return the plain model."""
        key_list = [key for key, _ in self.model.named_modules() if "tira" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, TiraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model
