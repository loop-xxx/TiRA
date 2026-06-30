import re
import warnings
from dataclasses import asdict
from enum import Enum

import torch
import torch.nn as nn

from hira.utils import ModulesToSaveWrapper, _freeze_adapter, _get_submodules
from .layer import TiraDiagonalLayer, TiraDiagonalLinear


def mark_only_tira_diagonal_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "tira_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
        return
    if bias == "tira_only":
        for m in model.modules():
            if isinstance(m, TiraDiagonalLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
        return
    raise NotImplementedError(f"Unsupported bias type: {bias}")


class TiraDiagonalModel(torch.nn.Module):
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
        mark_only_tira_diagonal_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _module_mk(self, key, config):
        if key.endswith("q_proj"):
            return config.tira_q_M or config.tira_M, config.tira_q_K or config.tira_K
        if key.endswith("k_proj"):
            return config.tira_k_M or config.tira_M, config.tira_k_K or config.tira_K
        if key.endswith("v_proj"):
            return config.tira_v_M or config.tira_M, config.tira_v_K or config.tira_K
        if key.endswith("o_proj"):
            return config.tira_o_M or config.tira_M, config.tira_o_K or config.tira_K
        if key.endswith("up_proj"):
            return config.tira_up_M or config.tira_M, config.tira_up_K or config.tira_K
        if key.endswith("down_proj"):
            return config.tira_down_M or config.tira_M, config.tira_down_K or config.tira_K
        return config.tira_M, config.tira_K

    def _find_and_replace(self, adapter_name):
        config = self.peft_config[adapter_name]
        found = False
        for key, _ in list(self.model.named_modules()):
            if isinstance(config.target_modules, str):
                target_module_found = re.fullmatch(config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in config.target_modules)
            if not target_module_found:
                continue

            found = True
            parent, target, target_name = _get_submodules(self.model, key)
            module_M, module_K = self._module_mk(key, config)
            module_alpha = config.tira_alpha if config.tira_alpha is not None else module_K
            kwargs = {"M": module_M, "K": module_K, "alpha": module_alpha}

            if isinstance(target, TiraDiagonalLayer):
                target.update_layer(adapter_name, **kwargs)
            elif isinstance(target, torch.nn.Linear):
                new_module = TiraDiagonalLinear(
                    adapter_name,
                    target.in_features,
                    target.out_features,
                    bias=target.bias is not None,
                    **kwargs,
                )
                self._replace_module(parent, target_name, new_module, target)
            else:
                raise ValueError(f"Target module {target} is not supported. Only torch.nn.Linear is supported.")

        if not found:
            raise ValueError(f"Target modules {config.target_modules} not found in the base model.")

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias") and old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
        for name, module in new_module.named_modules():
            if "tira_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
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
            if isinstance(module, TiraDiagonalLayer):
                module.disable_adapters = not enabled

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, TiraDiagonalLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, TiraDiagonalLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, TiraDiagonalLayer):
                module.unmerge()

    def merge_and_unload(self):
        key_list = [key for key, _ in self.model.named_modules() if "tira" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, TiraDiagonalLayer):
                new_module = torch.nn.Linear(
                    target.in_features,
                    target.out_features,
                    bias=target.bias is not None,
                )
                target.merge()
                self._replace_module(parent, target_name, new_module, target)
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model
