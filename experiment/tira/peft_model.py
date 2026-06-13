"""TiraPeftModel: PeftModel wrapper for TIRA adapters.

Inherits from hira's PeftModel to integrate with the save/load/checkpoint
infrastructure. Overrides __init__ to use TiraModel instead of the
PEFT_TYPE_TO_MODEL_MAPPING lookup.

Also provides TiraPeftModelForCausalLM for causal language modeling tasks.
"""
import os
from contextlib import contextmanager

import torch
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

from hira.peft_model import PeftModel
from hira.utils import (
    WEIGHTS_NAME,
    PeftConfig,
    PromptLearningConfig,
    _set_adapter,
    _set_trainable,
)

from .config import TiraConfig
from .model import TiraModel
from .save_and_load import get_tira_model_state_dict, set_tira_model_state_dict


class TiraPeftModel(PeftModel):
    """PeftModel subclass for TIRA adapters.

    Usage:
        config = TiraConfig(tira_M=16, tira_K=16, ...)
        model = TiraPeftModel(base_model, config)
        model.print_trainable_parameters()
    """

    def __init__(self, model, peft_config: TiraConfig, adapter_name="default"):
        PushToHubMixin.__init__(self)
        torch.nn.Module.__init__(self)

        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.base_model_torch_dtype = getattr(model, "dtype", None)

        self.peft_config[adapter_name] = peft_config
        self.base_model = TiraModel(self.base_model, self.peft_config, adapter_name)
        self.set_additional_trainable_modules(peft_config, adapter_name)

    def save_pretrained(self, save_directory, **kwargs):
        """Save TIRA adapter weights and config to directory."""
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.peft_config.items():
            output_state_dict = get_tira_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False, **kwargs):
        """Load a pretrained TIRA adapter."""
        config = TiraConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        peft_model = cls(model, config, adapter_name)
        peft_model.load_adapter(model_id, adapter_name, **kwargs)
        return peft_model

    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        """Load adapter weights from a directory or hub."""
        if adapter_name not in self.peft_config:
            config = TiraConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))
            config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, config)

        path = (
            os.path.join(model_id, kwargs["subfolder"])
            if kwargs.get("subfolder", None) is not None
            else model_id
        )

        if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, subfolder=kwargs.get("subfolder", None))
            except Exception:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        set_tira_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        self.eval()

    def add_adapter(self, adapter_name, peft_config):
        """Add a new TIRA adapter."""
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )
        self.peft_config[adapter_name] = peft_config
        self.base_model.add_adapter(adapter_name, peft_config)
        self.set_additional_trainable_modules(peft_config, adapter_name)

    def forward(self, *args, **kwargs):
        """Forward pass delegates to the base model."""
        return self.get_base_model()(*args, **kwargs)

    def get_base_model(self):
        """Returns the underlying model (unwrapped from TiraModel)."""
        return self.base_model.model

    @contextmanager
    def disable_adapter(self):
        """Context manager to temporarily disable adapters."""
        try:
            self.base_model.disable_adapter_layers()
            yield
        finally:
            self.base_model.enable_adapter_layers()

    def set_adapter(self, adapter_name):
        """Set the active adapter."""
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    def merge_and_unload(self):
        """Merge adapter into base weights and return the plain model."""
        return self.base_model.merge_and_unload()

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]


class TiraPeftModelForCausalLM(TiraPeftModel):
    """TIRA PeftModel for causal language modeling tasks.

    Follows hira's PeftModelForCausalLM pattern with generate() support.
    """

    def __init__(self, model, peft_config: TiraConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, **kwargs):
        self.base_model.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        try:
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.base_model.model.generate(**kwargs)
        except Exception:
            outputs = self.base_model.model.generate(**kwargs)
        finally:
            self.base_model.model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        return model_kwargs
