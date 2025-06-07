import argparse
import os
import sys

import torch
import wandb

from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu
from metamorph.algos.ppo.ppo_loki import LOKI


# Unimal 100
# max limbs: 12                                                                                                      
# max joints: 16  

# xml_debug
# max limbs: 12                                                                                                      
# max joints: 19 


def set_cfg_options():
    calculate_max_iters()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    
    # Add extra 1 for max_joints; needed for adding edge padding
    # Ainaz: These should be constant for now to have models with the same size.
    cfg.MODEL.MAX_JOINTS = 20 #max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = 12 #max(num_limbs) + 1 # TODO: Remove this line

    # print(f"max limbs: {cfg.MODEL.MAX_LIMBS}")
    # print(f"max joints: {cfg.MODEL.MAX_JOINTS}")

def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def get_hparams():
    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def cleanup_tensorboard():
    tb_dir = os.path.join(cfg.OUT_DIR, "tensorboard")

    # Assume there is only one sub_dir and break when it's found
    for content in os.listdir(tb_dir):
        content = os.path.join(tb_dir, content)
        if os.path.isdir(content):
            break

    # Return if no dir found
    if not os.path.isdir(content):
        return

    # Move all the event files from sub_dir to tb_idr
    for event_file in os.listdir(content):
        src = os.path.join(content, event_file)
        dst = os.path.join(tb_dir, event_file)
        fu.move_file(src, dst)

    # Delete the sub_dir
    os.rmdir(content)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--cfg", dest="cfg_file", help="Config file", required=True, type=str)
    parser.add_argument("opts", help="See morphology/core/config.py for all options", default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model on')
    parser.add_argument('--vae_path', type=str, default='5k_xml_vae', help='path to the vae model')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def loki_train(args, train=True):
    su.set_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    LOKITrainer = LOKI(args)
    if train:
        LOKITrainer.train()
        hparams = get_hparams()
        LOKITrainer.save_rewards(hparams=hparams)
        LOKITrainer.save_model()
        cleanup_tensorboard()
    else:
        # PPOTrainer.eval(cfg.PPO.NUM_EVAL_EPISODES)
        # PPOTrainer.generate_filter_morph()
        LOKITrainer.save_video(cfg.OUT_DIR, n_videos=1)

        # tournament on trained policy
        # LOKITrainer.final_tournament()


def main():

    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # Set cfg options which are inferred
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    cfg.ENV.WALKER_DIR = cfg.OUT_DIR

    # initialize wandb
    if cfg.LOKI.TRAIN:
        wandb.init(project="LOKI", name=cfg.OUT_DIR)
    else:
        wandb.init(project="LOKI-eval", name=cfg.OUT_DIR)
    # Save the config
    dump_cfg()
    loki_train(args, train=cfg.LOKI.TRAIN)


if __name__ == "__main__":
    main()
