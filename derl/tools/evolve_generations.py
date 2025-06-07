"""Script to evovle morphology."""

import argparse
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import pickle
import copy

import networkx as nx

from derl.config import cfg
from derl.config import dump_cfg
from derl.envs.morphology import SymmetricUnimal
# from derl.envs.morphology_rec import ReconstructedUnimal

from derl.utils import evo as eu
from derl.utils import file as fu
from derl.utils import sample as su
from derl.utils import similarity as simu


"""The script assumes the following folder structure.

cfg.OUT_DIR
    - models
    - metadata
    - xml
    - unimal_init
    - rewards

The evolution code has the following structure:
-- init_population
-- evolve_population
    1. select unimals to evolve
    2. evolve unimal. Save the mujoco xml (in xml) and save data required to
       instantiate the unimal class (in unimal_init). See SymmetricUnimal
    3. train unimal. Save the weights (in models). Finally save metadata like
       rews etc used in step 1.

Files inside metadata correspond to actual unimals in the population. Since we
use spot instances only if metadata file is present we can be sure all other
corresponding files will be present.

Distributed Training Setup: evolution.py is launched in parallel on multiple
cpu nodes. Node id can be identified by cfg.NODE_ID leveraging aws apis.
Each evolution script launches cfg.EVO.NUM_PROCESSES parallel subprocs.
For supporting SubprocEnv we need to use subprocess. Refer:
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
"""

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data



def setup_output_dir():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Make subfolders
    subfolders = [
        "xml_recons",
        "unimal_init_recons"
    ]
    for folder in subfolders:
        os.makedirs(os.path.join(cfg.OUT_DIR, folder), exist_ok=True)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument('--OUT_DIR', type=str, help='The output directory')
    parser.add_argument("--cfg", dest="cfg_file", help="Config file", required=True, type=str)
    parser.add_argument("opts", help="See morphology/config.py for all options", default=None, nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



def limb_count_pop_init(metadata, unimal_id, init_path):
    # Build unimals which initialize the population based on number of limbs.
    for generation in range(1, 6):
        for i in range(100):
            while True:
                unimal = SymmetricUnimal(id_=f'{unimal_id}_gen{generation}_{i}', init_path=init_path)
                for mutation_count in range(generation):
                    unimal.mutate()
                if unimal.num_limbs < cfg.LIMB.MAX_LIMBS:
                    unimal.save()
                    break
            
    # while unimal.num_limbs < len(metadata['limb_metadata'].keys()) - 1:
    #     print(unimal.num_limbs)
    #     unimal.grow_limb()

    unimal.save()
    return unimal_id

def reconstruct_unimals(args):
    pkl_paths = f'{args.OUT_DIR}/unimal_init'
    # remove all existing files in xml_path
    os.system(f'rm -rf {args.OUT_DIR}/xml_recons/*')
    for pkl_file in os.listdir(pkl_paths):
        pkl_path = os.path.join(pkl_paths, pkl_file)
        unimal_id = pkl_file.split('.')[0]

        metadata = load_pickle(pkl_path)
        limb_count_pop_init(metadata, unimal_id=unimal_id, init_path=pkl_path)
    

    print("Finished creating init xmls.")


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.OUT_DIR = args.OUT_DIR
    # Infer OPTIM.MAX_ITERS
    setup_output_dir()
    cfg.freeze()

    # Save the config
    dump_cfg()
    
    reconstruct_unimals(args)



if __name__ == "__main__":
    main()
    

# PYTHONPATH=. python tools/evolve_generations.py --cfg configs/evo/ft.yml --OUT_DIR ./xml_generations/ft