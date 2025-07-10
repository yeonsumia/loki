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

import networkx as nx

try:
    from derl.config import cfg
    from derl.config import dump_cfg
    from derl.envs.morphology import SymmetricUnimal

    from derl.utils import evo as eu
    from derl.utils import file as fu
    from derl.utils import sample as su
    from derl.utils import similarity as simu
    from derl.utils import xml as xu
except:
    import sys
    sys.path.append("..")
    from derl.derl.config import cfg
    from derl.derl.config import dump_cfg
    from derl.derl.envs.morphology import SymmetricUnimal

    from derl.derl.utils import evo as eu
    from derl.derl.utils import file as fu
    from derl.derl.utils import sample as su
    from derl.derl.utils import similarity as simu
    from derl.derl.utils import xml as xu

ENV_TRAIN_STEPS = {
    "obstacle": 5e6,
    "many_obstacle": 5e6,
    "patrol": 5e6,
    "incline": 5e6,
    "push_box_incline": 5e6,
    "bump": 1.5e7,
    "manipulation_ball": 2e7,
    "exploration": 2e7,
}

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS)
        // cfg.PPO.TIMESTEPS
        // cfg.PPO.NUM_ENVS
    )


def setup_output_dir():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
        
    # Make subfolders
    subfolders = [
        # "models",
        # "metadata",
        "xml",
        # "unimal_init",
        # "rewards",
        # "videos",
        "error_metadata",
        # "images",
        # f"error_metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
        # f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
        # f"models_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
        # f"rewards_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
    ]
    if cfg.ENV_TYPE != "":
        # subfolders.append(f"models_{cfg.ENV_TYPE}")
        # subfolders.append(f"metadata_{cfg.ENV_TYPE}")
        # subfolders.append(f"rewards_{cfg.ENV_TYPE}")
        subfolders.append(f"models_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}")
        subfolders.append(f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}")
        subfolders.append(f"rewards_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}")

    for folder in subfolders:
        os.makedirs(os.path.join(cfg.OUT_DIR, folder), exist_ok=True)


def setup_output_dir_input(our_dir):
    # os.makedirs(our_dir, exist_ok=True)
        
    # Make subfolders
    subfolders = [
    #     "models",
    #     "metadata",
    #     "xml",
    #     "unimal_init",
    #     "rewards",
    #     "videos",
        "error_metadata",
    #     "images",
    #     f"error_metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
    #     f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
    #     f"models_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
    #     f"rewards_{cfg.PPO.MAX_STATE_ACTION_PAIRS}",
    ]
    # subfolders = []
    if cfg.ENV_TYPE != "":
        subfolders.append(f"models_{cfg.ENV_TYPE}")
        subfolders.append(f"metadata_{cfg.ENV_TYPE}")
        subfolders.append(f"rewards_{cfg.ENV_TYPE}")
        subfolders.append(f"models_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}")
        subfolders.append(f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}")
        subfolders.append(f"rewards_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}")

    for folder in subfolders:
        os.makedirs(os.path.join(our_dir, folder), exist_ok=True)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See morphology/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--RNG_SEED', type=int, default=1, help='seed for random number generator')
    parser.add_argument('--OUT_DIR', type=str, default='output', help='The output directory')
    parser.add_argument('--min_seq_len', type=int, default=4, help='The minimum sequence length')
    parser.add_argument('--max_seq_len', type=int, default=10, help='The maximum sequence length')
    parser.add_argument('--init_population_size', type=int, default=10000, help='The initial population size')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def wait_till_init():
    init_setup_done_path = os.path.join(cfg.OUT_DIR, "init_setup_done")
    max_wait = 3600  # one hour
    time_waited = 0
    while not os.path.exists(init_setup_done_path):
        time.sleep(60)
        time_waited += 60
        if time_waited >= max_wait:
            print("Initial xmls not made. Exiting!")
            sys.exit(1)


def limb_count_pop_init(idx, unimal_id):
    # Build unimals which initialize the population based on number of limbs.
    min_length, max_length, _ = cfg.LIMB.NUM_LIMBS_RANGE
    # print(f"min_length: {min_length}, max_length: {max_length}")
    unimal = SymmetricUnimal(unimal_id)
    num_limbs = su.sample_from_range(cfg.LIMB.NUM_LIMBS_RANGE)
    unimal.mutate(op="grow_limb")
    while unimal.num_limbs < num_limbs:
        unimal.mutate()
        # unimal.save_image()

    unimal.save()
    return unimal_id


def create_init_unimals():
    init_setup_done_path = os.path.join(cfg.OUT_DIR, "init_setup_done")

    if os.path.isfile(init_setup_done_path):
        print("Init xmls have already been.")
        return

    init_pop_size = cfg.EVO.INIT_POPULATION_SIZE
    print("Creating init xmls for {} unimals.".format(init_pop_size))

    # Create unimal xmls
    p = Pool(cfg.EVO.NUM_PROCESSES)
    timestamp = datetime.now().strftime("%d-%H-%M-%S")
    idx_unimal_id = [
        (idx, "{}-{}-{}".format(cfg.NODE_ID, idx, timestamp))
        for idx in range(int(2 * init_pop_size))
    ]

    unimal_ids = p.starmap(globals()[cfg.EVO.INIT_METHOD], idx_unimal_id)

    G = simu.create_graph_from_uids(
        None, unimal_ids, "geom_orientation", graph_type="species"
    )
    cc = list(nx.connected_components(G))

    print("Number of same unimals: ", len(cc))
    unimals_to_remove = []
    unimals_to_keep = []
    for same_unimals in cc:
        if len(same_unimals) == 1:
            unimals_to_keep.extend(list(same_unimals))
            continue
        remove_unimals = sorted(
            list(same_unimals),
            key=lambda unimal_id: "-".join(unimal_id.split("-")[:2]),
        )
        unimals_to_keep.append(remove_unimals[0])
        remove_unimals = remove_unimals[1:]
        unimals_to_remove.extend(remove_unimals)

    # Number of unimals to add to achieve init_pop_size.
    padding_count = init_pop_size - len(cc)
    print("Number of unimals to add to achieve init_pop_size.", padding_count)
    if padding_count > 0:
        random.shuffle(unimals_to_remove)
        unimals_to_remove = unimals_to_remove[padding_count:]
    else:
        random.shuffle(unimals_to_keep)
        unimals_to_remove.extend(unimals_to_keep[init_pop_size:])

    for unimal in unimals_to_remove:
        fu.remove_file(fu.id2path(unimal, "xml"))
        fu.remove_file(fu.id2path(unimal, "unimal_init"))
        fu.remove_file(fu.id2path(unimal, "images"))

    Path(init_setup_done_path).touch()
    print("Finished creating init xmls.")


def kill_pg(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


def relaunch_proc(p, proc_id):
    # Kill the process group
    kill_pg(p)
    # Launch the subproc again
    print("Node ID: {}, proc-id: {} relaunching".format(cfg.NODE_ID, proc_id))
    p = launch_subproc(proc_id)
    return p


def wait_or_kill(subprocs):
    # Main process will wait till we have done search
    while eu.get_searched_space_size() < cfg.EVO.SEARCH_SPACE_SIZE:
        time.sleep(10)  # 10 secs

        # Re-launch subproc if exit was due to error
        new_subprocs = []
        for idx in range(len(subprocs)):
            p, proc_id = subprocs[idx]
            poll = p.poll()

            is_error_path = os.path.join(
                cfg.OUT_DIR, "{}_{}".format(cfg.NODE_ID, p.pid)
            )
            if os.path.exists(is_error_path) or poll:
                fu.remove_file(is_error_path)
                p = relaunch_proc(p, proc_id)

            new_subprocs.append((p, proc_id))

        subprocs = new_subprocs

    if eu.should_save_video():
        video_dir = fu.get_subfolder("videos")
        reg_str = "{}-.*json".format(cfg.NODE_ID)
        while len(fu.get_files(video_dir, reg_str)) > 0:
            time.sleep(60)

    # Ensure that all process will close, dangling process will prevent docker
    # from exiting.
    for p, _ in subprocs:
        kill_pg(p)


def wait_or_kill_gen(subprocs):
    # Main process will wait till we have done generation
    while eu.get_gen_population_size() < cfg.EVO.INIT_POPULATION_SIZE:
        time.sleep(10)  # 10 secs

        # Re-launch subproc if exit was due to error
        new_subprocs = []
        for idx in range(len(subprocs)):
            p, proc_id = subprocs[idx]
            poll = p.poll()

            is_error_path = os.path.join(
                cfg.OUT_DIR, "{}_{}".format(cfg.NODE_ID, p.pid)
            )
            if os.path.exists(is_error_path) or poll:
                fu.remove_file(is_error_path)
                p = relaunch_proc(p, proc_id)

            new_subprocs.append((p, proc_id))

        subprocs = new_subprocs

    if eu.should_save_video():
        video_dir = fu.get_subfolder("videos")
        reg_str = "{}-.*json".format(cfg.NODE_ID)
        while len(fu.get_files(video_dir, reg_str)) > 0:
            time.sleep(60)

    # Ensure that all process will close, dangling process will prevent docker
    # from exiting.
    for p, _ in subprocs:
        kill_pg(p)



def launch_subproc(proc_id):
    cfg_path = os.path.join(cfg.OUT_DIR, cfg.CFG_DEST)
    cmd = "python tools/evo_single_proc.py --cfg {} --proc-id {}".format(
        cfg_path, proc_id
    )
    cmd += " NODE_ID {}".format(cfg.NODE_ID)
    cmd += " PPO.MAX_STATE_ACTION_PAIRS {}".format(cfg.PPO.MAX_STATE_ACTION_PAIRS)
    p = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid
    )
    return p

def launch_subproc_ppo_train(proc_id, cur_iter_path, num_nodes, node_id, num_procs, max_state_actions):
    cfg_path = os.path.join(cur_iter_path, cfg.CFG_DEST)
    cmd = "python tools/evo_single_proc.py \
            --cfg {} \
            --proc-id {} \
            OUT_DIR {} \
            EVAL_XML_PATH {} \
            EVO.IS_EVO True \
            INIT_EVAL False \
            EVO.USE_EXIST_AGENT True \
            NUM_NODES {} \
            ENV_TYPE 'ft' \
            NODE_ID {} \
            EVO.NUM_PROCESSES {} \
            PPO.MAX_STATE_ACTION_PAIRS {}".format(
        cfg_path, proc_id, cur_iter_path, "map_xml", num_nodes, node_id, num_procs, max_state_actions
    )
    p = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid
    )
    return p



def evolve():
    # Create initial unimals only in master node
    if cfg.EVO.USE_EXIST_AGENT:
        pass
    elif cfg.NODE_ID == 0:
        print("Init!!!!", flush=True)
        create_init_unimals()
    else:
        print("waiting!!!!", flush=True)
        wait_till_init()

    subprocs = []
    for idx in range(cfg.EVO.NUM_PROCESSES):
        p = launch_subproc(idx)
        subprocs.append((p, idx))

    wait_or_kill_gen(subprocs)
    print("Node ID: {} killed all subprocs!".format(cfg.NODE_ID))


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.EVO.INIT_POPULATION_SIZE = args.init_population_size
    cfg.OUT_DIR = args.OUT_DIR
    cfg.RNG_SEED = args.RNG_SEED
    if args.min_seq_len == args.max_seq_len:
        cfg.LIMB.NUM_LIMBS_RANGE = [args.max_seq_len]
    # elif args.min_seq_len % 2 == 0 and args.max_seq_len % 2 == 0:
    #     cfg.LIMB.NUM_LIMBS_RANGE = [args.min_seq_len, args.max_seq_len, 2]
    else:
        assert args.min_seq_len < args.max_seq_len
        cfg.LIMB.NUM_LIMBS_RANGE = [args.min_seq_len, args.max_seq_len, 1]
    
    print(f"limb count range: {cfg.LIMB.NUM_LIMBS_RANGE}")
    if cfg.ENV_TYPE != "":
        cfg.PPO.MAX_STATE_ACTION_PAIRS = ENV_TRAIN_STEPS[cfg.ENV_TYPE]
        print(f"{cfg.ENV_TYPE}: {cfg.PPO.MAX_STATE_ACTION_PAIRS} steps")
    # Infer OPTIM.MAX_ITERS
    calculate_max_iters()
    setup_output_dir()
    cfg.freeze()

    # Save the config
    dump_cfg()

    evolve()


if __name__ == "__main__":
    main()
