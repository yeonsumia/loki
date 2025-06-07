import torch

from metamorph.config import cfg


def restore_from_checkpoint(ac):
    print("Loading pretrained actor critic model")
    model_p, ob_rms = torch.load(cfg.PPO.CHECKPOINT_PATH)

    state_dict_c = ac.state_dict()
    state_dict_p = model_p.state_dict()

    fine_tune_layers = set()
    layer_substrings = cfg.MODEL.FINETUNE.LAYER_SUBSTRING
    for name, param in state_dict_c.items():    
        
        if cfg.LOKI.RESUME_ITER > 0:
            pass
        elif "limb_obs_morph_embed" in name or "hfield" in name or "decoder" in name:
            continue
        param_p = state_dict_p[name]
        # print(name, param.shape, param_p.shape)
        if param_p.shape == param.shape:
            with torch.no_grad():
                param.copy_(param_p)
        else:
            raise ValueError(
                f"Checkpoint path is invalid as there is shape mismatch with {name}"
            )
        if any(name_substr in name for name_substr in layer_substrings):
            fine_tune_layers.add(name)

    if not cfg.MODEL.FINETUNE.FULL_MODEL:
        for name, param in ac.named_parameters():
            if name not in fine_tune_layers:
                param.requires_grad = False
            else:
                print(f"Fine-tuning layer {name}")

    return ob_rms



def restore_from_checkpoints_model_soup(ac):
    """
    Restore model parameters by averaging parameters from multiple checkpoints.

    """
    # Initialize an empty state_dict to accumulate the parameters
    checkpoint_paths = [cfg.PPO.CHECKPOINT_PATH, cfg.PPO.CHECKPOINT_PATH2]
    accumulated_state_dict = None

    state_dict_c = ac.state_dict()
    num_checkpoints = len(checkpoint_paths)

    for path in checkpoint_paths:
        checkpoint, ob_rms = torch.load(path, map_location='cpu')
        
        # Add the parameters to the accumulator
        for key in checkpoint.state_dict():
            # print(state_dict_c[key].device, checkpoint.state_dict()[key].device)
            state_dict_c[key] += checkpoint.state_dict()[key]

    # Average the accumulated parameters
    for key in state_dict_c:
        state_dict_c[key] /= num_checkpoints

    return ob_rms

