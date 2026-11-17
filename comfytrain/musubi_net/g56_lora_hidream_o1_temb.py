"""g56_lora_hidream_o1_temb.py (G56NF_MUSUBI_TEMB, research; comfytrain/g56_newfam.py MUSUBI_TEMB): the LoRA network module of

evaluator's loader (create_arch_network_from_weights = the full HIDREAM_O1_TARGET_REPLACE_MODULES, lora_hidream_o1.py:64-74) builds
a module for every lora name the file holds (networks/lora.py:983-1011, 617-621) - the TimestepEmbedder's included - while the
official create_arch_network (:26-61) trains the I2I / T2I list only. Selected with --network_module g56_lora_hidream_o1_temb, this
folder on the training child's PYTHONPATH; no Musubi file is patched. Everything else is Musubi's own: the same create_network,
prefix, rank / alpha arguments, exclude patterns; create_arch_network_from_weights is Musubi's function itself."""
import ast

import musubi_tuner.networks.lora as lora
import musubi_tuner.networks.lora_hidream_o1 as official

create_arch_network_from_weights = official.create_arch_network_from_weights


def create_arch_network(multiplier, network_dim, network_alpha, vae, text_encoders, unet, neuron_dropout=None, **kwargs):
    """official.create_arch_network with HIDREAM_O1_TIMESTEP_TARGET_REPLACE_MODULES added to its target list (the same order of
    arguments, the same exclude pattern for embed_tokens / lm_head)."""
    has_control = unet.hidream_o1_task == "i2i"
    targets = list(official.HIDREAM_O1_I2I_TARGET_REPLACE_MODULES if has_control else official.HIDREAM_O1_T2I_TARGET_REPLACE_MODULES)
    targets += [m for m in official.HIDREAM_O1_TIMESTEP_TARGET_REPLACE_MODULES if m not in targets]
    exclude_patterns = kwargs.get("exclude_patterns", None) or []
    if isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    exclude_patterns.append(r".*(embed_tokens|lm_head).*")
    kwargs["exclude_patterns"] = exclude_patterns
    return lora.create_network(targets, "lora_unet", multiplier, network_dim, network_alpha, vae, text_encoders, unet,
                               neuron_dropout=neuron_dropout, **kwargs)
