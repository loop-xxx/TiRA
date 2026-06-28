def get_tira_ablation_model_state_dict(model, state_dict=None, adapter_name="default"):
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    bias = getattr(config, "bias", "none")

    if bias == "none":
        to_return = {k: state_dict[k] for k in state_dict if "tira_" in k}
    elif bias == "all":
        to_return = {k: state_dict[k] for k in state_dict if "tira_" in k or "bias" in k}
    elif bias == "tira_only":
        to_return = {}
        for k in state_dict:
            if "tira_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("tira_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError(f"Unsupported bias type: {bias}")

    to_return = {
        k: v for k, v in to_return.items()
        if ("tira_" in k and adapter_name in k) or ("bias" in k)
    }
    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}

    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(
                f"{module_name}.modules_to_save.{adapter_name}" in key
                for module_name in model.modules_to_save
            ):
                to_return[key.replace("modules_to_save.", "")] = value
    return to_return


def set_tira_ablation_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    state_dict = peft_model_state_dict
    if model.modules_to_save is not None:
        state_dict = {}
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value

    final_state_dict = {}
    for k, v in state_dict.items():
        if "tira_" in k:
            suffix = k.split("tira_")[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                k = f"{k}.{adapter_name}"
        final_state_dict[k] = v
    model.load_state_dict(final_state_dict, strict=False)
