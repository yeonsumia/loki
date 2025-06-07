import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Torch 1.5 bug, need to import numpy before torch
# Refer: https://github.com/pytorch/pytorch/issues/37377
import numpy as np
import torch
import re
import time

from derl.algos.ppo.ppo import PPO
from derl.config import cfg
from derl.envs.morphology import SymmetricUnimal
from derl.utils import evo as eu
from derl.utils import exception as exu
from derl.utils import file as fu
from derl.utils import sample as su
from tools.evolution import calculate_max_iters

from collections import defaultdict
from functools import reduce

from evolution import ENV_TRAIN_STEPS

represents = np.mean
represents_name = "Mean"

derl_legend = "DERL"
our_legend = "Ours"
our_legend1 = "Ours \n(total)"
our_legend2 = "Ours \n(each cluster)"
our_legend3 = "Ours \n(no cluster)"
our_legend = "Ours"
baseline_legend = "Random"

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", help="Config file", required=True, type=str)
    parser.add_argument("--proc-id", required=True, type=int)
    parser.add_argument("--env-type", type=str, default=None)
    parser.add_argument("--calculate-reward", action="store_true")
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--mutate-top-k-morphs", action="store_true")
    parser.add_argument("--select-top-k-agent-env", action="store_true")
    parser.add_argument("opts", help="See morphology/config.py for all options", default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train(xml_file, id_, seed=None, parent_metadata=None, env_type=None):
    # su.set_seed(cfg.RNG_SEED, use_strong_seeding=False)s
    # Setup torch
    torch.set_num_threads(1)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    # Train unimal
    PPOTrainer = PPO(xml_file=xml_file)
    agent_id = f"{id_}_seed{seed}" if seed is not None else id_

    exit_cond = None

    if (
        exit_cond == "population_init" and
        eu.get_population_size() >= cfg.EVO.INIT_POPULATION_SIZE
    ):
        return

    if (
        exit_cond == "search_space" and
        eu.get_searched_space_size() >= cfg.EVO.SEARCH_SPACE_SIZE
    ):
        return

    # Save the model
    models_subfolder = f"models_{cfg.PPO.MAX_STATE_ACTION_PAIRS}" if env_type is None else f"models_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{env_type}"
    rewards_subfolder = f"rewards_{cfg.PPO.MAX_STATE_ACTION_PAIRS}" if env_type is None else f"rewards_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{env_type}"

    print(f"Saving model path: {fu.id2path(agent_id, models_subfolder)}")
    print(f"Saving rewards path: {fu.id2path(agent_id, rewards_subfolder)}")
    calculate_max_iters()
    PPOTrainer.train(exit_cond=exit_cond)
    PPOTrainer.save_model(path=fu.id2path(agent_id, models_subfolder))
    # Save the rewards
    PPOTrainer.save_rewards(path=fu.id2path(agent_id, rewards_subfolder))

    # if eu.should_save_video():
    #     PPOTrainer.save_video(os.path.join(cfg.OUT_DIR, "videos"))

    # Create metadata to be used for next steps of evolution
    metadata = {}
    # Add mean of last N (100) values of rewards
    for rew_type, rews in PPOTrainer.mean_ep_rews.items():
        metadata[rew_type] = int(np.mean(rews[-100:]))

    metadata["pos"] = np.mean(PPOTrainer.mean_pos[-100:])

    metadata["id"] = agent_id
    if not parent_metadata:
        metadata["lineage"] = "{}".format(agent_id)
    else:
        metadata["lineage"] = "{}/{}".format(parent_metadata["lineage"], agent_id)

    # Save metadata to disk
    if env_type is not None:
        path = os.path.join(fu.get_subfolder(f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{env_type}"), "{}.json".format(agent_id))
    else:
        path = os.path.join(fu.get_subfolder(f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}"), "{}.json".format(agent_id))
    print(f"Saving metadata path: {path}")
    fu.save_json(metadata, path)


def init_done(unimal_id):
    unimal_idx = int(unimal_id.split(".")[0].split("-")[1])
    success_metadata = fu.get_files(fu.get_subfolder("metadata"), ".*json")
    error_metadata = fu.get_files(fu.get_subfolder("error_metadata"), ".*json")
    done_metadata = success_metadata + error_metadata
    done_idx = [
        int(path.split("/")[-1].split("-")[2].split(".")[0])
        for path in done_metadata
    ]
    if unimal_idx in done_idx:
        return True
    else:
        return False

def init_done_loki(unimal_id):
    success_metadata = fu.get_files(fu.get_subfolder(f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}"), ".*json")
    error_metadata = fu.get_files(fu.get_subfolder(f"error_metadata"), ".*json")
    done_metadata = success_metadata + error_metadata
    done_idx = [
        fu.path2id(path)
        for path in done_metadata
    ]
    if unimal_id in done_idx:
        return True
    else:
        return False


def init_population(proc_id):
    init_done_path = os.path.join(cfg.OUT_DIR, "init_pop_done")

    # if os.path.isfile(init_done_path):
    #     print("Population has already been initialized.")
    #     return
    
    # Divide work by num nodes and then num procs
    xml_paths = fu.get_files(
        fu.get_subfolder(cfg.BASE_DIR), ".*xml", sort=True, sort_type="time"
    )
    # xml_paths = [m for m in xml_paths if "cluster40_idx0" in m or "cluster40_idx5" in m or "cluster40_idx10" in m or "cluster40_idx17" in m or "cluster40_idx18" in m or "cluster40_idx24" in m or "cluster40_idx25" in m or "cluster40_idx31" in m]
    # xml_paths = [m for m in xml_paths if "cluster40_idx3_" in m]
    print("Total XMLs: {}".format(len(xml_paths)))
    # return
    cfg.EVO.INIT_POPULATION_SIZE = len(xml_paths)
    xml_paths.sort()
    print("Total XMLs: {}".format(len(xml_paths)))
    xml_paths = fu.chunkify(xml_paths, cfg.NUM_NODES)[cfg.NODE_ID]
    print("XMLs for this node: {}".format(len(xml_paths)))
    xml_paths = fu.chunkify(xml_paths, cfg.EVO.NUM_PROCESSES)[proc_id]

    for xml_path in xml_paths:
        unimal_id = fu.path2id(xml_path)
        print(unimal_id, xml_path)

        if init_done_loki(unimal_id):
            print("{} already done, proc_id: {}".format(unimal_id, proc_id))
            continue

        ppo_train(fu.id2path(unimal_id, cfg.BASE_DIR), unimal_id)

        # if eu.get_population_size() >= cfg.EVO.INIT_POPULATION_SIZE:
        #     break

    # Explicit file is needed as current population size can be less than
    # initial population size. In fact after the first round of tournament
    # selection population size can be as low as half of
    Path(init_done_path).touch()


def tournament_evolution(idx):
    seed = cfg.RNG_SEED + (cfg.NODE_ID * cfg.EVO.NUM_PROCESSES + idx) * 100
    while eu.get_searched_space_size() < cfg.EVO.SEARCH_SPACE_SIZE:
        su.set_seed(seed, use_strong_seeding=True)
        seed += 1
        parent_metadata = eu.select_parent()
        print(f"Parent metadata: {parent_metadata['id']}")
        child_id = "{}-{}-{}".format(
            cfg.NODE_ID, idx, datetime.now().strftime("%d-%H-%M-%S")
        )
        unimal = SymmetricUnimal(
            child_id, init_path=fu.id2path(parent_metadata["id"], "unimal_init"),
        )
        unimal.mutate()
        unimal.save()

        ppo_train(fu.id2path(child_id, "xml"), child_id, parent_metadata)

    # Even though video meta files are removed inside ppo, sometimes it might
    # fail in between creating video. In such cases, we just remove the video
    # metadata file as master proc uses absence of meta files as sign of completion.
    video_dir = fu.get_subfolder("videos")
    video_meta_files = fu.get_files(
        video_dir, "{}-{}-.*json".format(cfg.NODE_ID, idx)
    )
    for video_meta_file in video_meta_files:
        fu.remove_file(video_meta_file)


def mutate_loki_final_morphs(idx):
    seed = cfg.RNG_SEED + (cfg.NODE_ID * cfg.EVO.NUM_PROCESSES + idx) * 100
    num_mutation = cfg.EVO.NUM_MUTATION
    print(f"number of mutation: {num_mutation}")
    
    xml_paths = fu.get_files(fu.get_subfolder(cfg.EVAL_XML_PATH), ".*xml")
    print("Total metadata: {}".format(len(xml_paths)))
    print("metadata for this node/process: {}".format(xml_paths))
    
    mutate_count = 0
    mutation_ops = [
            "joint_angle",
            "limb_params",
            "dof",
            "density",
            "gear"
    ]
    for xml_path in xml_paths:
        unimal_name = fu.path2id(xml_path)
        print(f"Unimal name: {unimal_name}")
        mutations_cnt = len([m for m in xml_paths if unimal_name in m])
        if mutations_cnt >= cfg.EVO.NUM_MUTATION:
            print(f"Already mutated: {unimal_name}")
            continue
        for _ in range(num_mutation-mutations_cnt):
            su.set_seed(seed, use_strong_seeding=True)
            seed += 1
            child_id = "node{}-proc{}-time{}-parent-{}".format(
                cfg.NODE_ID, idx, datetime.now().strftime("%d-%H-%M-%S"), unimal_name
            )
            print(f"Child id: {child_id}")
            unimal = SymmetricUnimal(
                child_id, init_path=os.path.join(fu.get_subfolder(cfg.EVAL_XML_PATH.replace("xml", "unimal_init")), f"{unimal_name}.pkl")
            )
            op = np.random.choice(mutation_ops)
            unimal.mutate(op)
            unimal.save()

            # sleep for 1 sec to avoid same time stamp
            time.sleep(1)
            mutate_count += 1

            # ppo_train(fu.id2path(child_id, "xml"), child_id, parent_metadata)
        
    print(f"Total mutated morphs saved: {mutate_count}")


def eval_loki_final_morphs(idx, xml_name=None, num_seeds=4):
    seed = cfg.RNG_SEED + (cfg.NODE_ID * cfg.EVO.NUM_PROCESSES + idx) * 100
    
    xml_paths = fu.get_files(
        fu.get_subfolder(cfg.EVAL_XML_PATH), ".*xml", sort=True, sort_type="time"
    )

    target_metadata_paths = fu.get_files(fu.get_subfolder(f"metadata_{cfg.PPO.MAX_STATE_ACTION_PAIRS}_{cfg.ENV_TYPE}"), ".*json")
    # target_metadata_paths = fu.get_files(fu.get_subfolder(f"metadata_{cfg.ENV_TYPE}"), ".*json")
    target_metadata_name_list = [fu.path2id(m) for m in target_metadata_paths]
    if xml_name is not None:
        xml_paths = [m for m in xml_paths if xml_name in m]
    print("Total xml: {}".format(len(xml_paths)))

    xml_paths = fu.chunkify(xml_paths, cfg.NUM_NODES)[cfg.NODE_ID]
    xml_paths = fu.chunkify(xml_paths, cfg.EVO.NUM_PROCESSES)[idx]

    print("xml for this node/process: Total {} {}".format(len(xml_paths), xml_paths))
    
    for xml_path in xml_paths:
        id = fu.path2id(xml_path)
        id_remove_seed = re.sub(r'_seed\d+', '', id)
        existing_cnt = len([m for m in target_metadata_name_list if id_remove_seed in re.sub(r'_seed\d+', '', m)])
        if existing_cnt >= num_seeds:
            print(f"Already evaluated: {id} / {existing_cnt}")
            continue
        print(id, f"Existing: {existing_cnt}/{num_seeds}")
        for _ in range(num_seeds-existing_cnt):
            seed += 100
            su.set_seed(seed, use_strong_seeding=True)
            if f"{id}_seed{seed}" in target_metadata_name_list:
                print(f"Already evaluated: {id}_seed{seed}")
                continue
                
            ppo_train(xml_path, id, seed, env_type=cfg.ENV_TYPE)


def select_top_k_agent_env(k=-1, num_seeds=3):
    folder_name = f"metadata_{ENV_TRAIN_STEPS[cfg.ENV_TYPE]}_{cfg.ENV_TYPE}"
    metadata_paths = fu.get_files(fu.get_subfolder(folder_name), ".*json")
    # reward_paths = fu.get_files(fu.get_subfolder(folder_name.replace("metadata", "rewards")), ".*json")
    
    if k == -1:
        k = len(metadata_paths) // 2
    print(len(metadata_paths))
    reward_dict = {}
    reward_total_dict = {}
    # energy_dict = {}
    dist_dict = {}
    for m_path in metadata_paths:
        id = fu.path2id(m_path)
        splits = list(find_all(id, "_seed"))
        if len(splits) == 1:
            pass
        elif len(splits) == 1 and "floor" not in id and "-" not in id:
            splits = []
        # if "seed" not in id:
        #     continue
        # print(splits)
        if len(splits) > 0:
            name_index = splits[-1]
            name = id[:name_index]
            name = re.sub(r'_seed\d+', '', name)
        else:
            name = id
        m = fu.load_json(m_path)
        if name not in reward_dict:
            reward_total_dict[name] = []
            reward_dict[name] = []
            # energy_dict[name] = []
            dist_dict[name] = []
        if "manipulation_ball" in folder_name:
            reward_dict[name].append((id, m['__reward__push']))
            reward_total_dict[name].append((id, m['reward']))
        else:
            reward_dict[name].append((id, m['reward']))
        # energy_dict[name].append((id, m['__reward__energy']))
    print(len(reward_dict))
    reward_mean_dict = {}
    for name, rewards in reward_dict.items():
        # if len(rewards) < 5:
        #     continue
        best_n_rewards = sorted(rewards, key=lambda x: x[1], reverse=True)[:num_seeds]
        
        reward_mean_dict[name] = np.mean([r[1] for r in best_n_rewards])
    
    # sort by mean reward
    sorted_reward_mean_dict = sorted(reward_mean_dict.items(), key=lambda x: x[1], reverse=True)

    # print(sorted_reward_mean_dict)
    # select top k agents
    # save_metadata = f"{folder_name}_top_{k}"
    # save_unimal_init = f"{folder_name}_top_{k}_unimal_init"
    # save_xml = f"{folder_name}_top_{k}_xml"
    # save_model = f"{folder_name}_top_{k}_model"
    
    # os.makedirs(fu.get_subfolder(save_xml), exist_ok=True)
    # os.makedirs(fu.get_subfolder(save_model), exist_ok=True)
    
    # print(save_metadata)
    # print(save_unimal_init)
    original_len = len(sorted_reward_mean_dict)

    print("=" * 50)
    print(f"Total {len(sorted_reward_mean_dict)}/{original_len} agents")
    print("=" * 50)

    name_list = []
    reward_list = []
    for name, rewards in sorted_reward_mean_dict[:k]:
        origin_id = reward_dict[name][0][0]
        if len(list(find_all(origin_id, "_seed"))) <= 0:
            name_index = len(origin_id)
        else:
            name_index = list(find_all(origin_id, "_seed"))[-1]
        id = origin_id[:name_index]
        best_n_rewards = sorted(reward_dict[name], key=lambda x: x[1], reverse=True)[:num_seeds]
        for i in range(len(reward_dict[name])):
            print(reward_dict[name][i][0], reward_dict[name][i][1])
            name_list.append(reward_dict[name][i][0])
            
        # print(reward_dict[name][best_n_rewards_indices[0]][0], reward_dict[name][best_n_rewards_indices[0]][1])
        
        # os.system(f"cp {fu.id2path(id, 'cluster40_final_xmls/xml')} {fu.get_subfolder(save_xml)}")
        # os.system(f"cp {fu.id2path(reward_dict[name][best_n_rewards_indices[0]][0], folder_name.replace('metadata', 'models'))} {fu.get_subfolder(save_model)}")
        reward_list.extend([r[1] for r in best_n_rewards])
        print()
    
    print(reward_list)
    
    return sorted_reward_mean_dict[:k], reward_dict


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def calculate_reward_mean_std(folder_name, name_list):
    metadata_paths = fu.get_files(fu.get_subfolder(folder_name), ".*json")
    metadatas = [fu.load_json(m) for m in metadata_paths]
    if len(name_list) <= 1:
        name_list = [m["id"].split("_seed")[0] for m in metadatas]
    print(f"Total name_list: {len(name_list)}")
    print("=====================================")
    reward_mean = defaultdict(list)

    for m in metadatas:
        if "_seed" not in m["id"]:
            continue
        name_index = list(find_all(m["id"], "_seed"))[-1]
        name = m["id"][:name_index]
        if name not in name_list:
            continue
        reward_mean[name].append(m['reward'])
    
    for name, rewards in reward_mean.items():
        print(f"[{name}] mean: {np.mean(rewards)}, std: {np.std(rewards)}, total: {len(rewards)}")
    
    print("")
    # Total median/mean/std
    all_rewards = reduce(lambda x, y: x + y, reward_mean.values())
    sorted_rewards = sorted(all_rewards, reverse=True)
    print(f"[Total] median: {np.median(sorted_rewards)} / mean: {np.mean(sorted_rewards)} / std: {np.std(sorted_rewards)}")

    print("")
    # Top 10 mean and std
    # k = 10
    # sorted_name_list = sorted(reward_mean.keys(), key=lambda x: np.mean(reward_mean[x]), reverse=True)
    # # print top k name
    # print(f"Top {k} designs for {cfg.ENV_TYPE}: {sorted_name_list[:k]}")
    # top_k_rewards = reduce(lambda x, y: x + y, [reward_mean[name] for name in sorted_name_list[:k]])
    # print(f"[Top {k}] median: {np.median(top_k_rewards)} / mean: {np.mean(top_k_rewards)} / std: {np.std(top_k_rewards)}")

    print("=====================================")
    return reward_mean
 

def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    # Unclear why this happens, very rare
    if cfg.OUT_DIR == "/tmp":
        exu.handle_exception("", "ERROR TMP")

    if args.select_top_k_agent_env:
        select_top_k_agent_env(args.top_k)
    if args.mutate_top_k_morphs:
        mutate_loki_final_morphs(args.proc_id)
        print("Node ID: {}, Proc ID: {} finished.".format(cfg.NODE_ID, args.proc_id))
    elif args.calculate_reward:
        calculate_reward_mean_std(f"metadata_{cfg.ENV_TYPE}", cfg.FINAL_LIST.split(" "))
    elif cfg.INIT_EVAL:
        init_population(args.proc_id)
    elif cfg.EVO.IS_EVO_TASK:
        eval_loki_final_morphs(args.proc_id, num_seeds=5)
        print("Node ID: {}, Proc ID: {} finished.".format(cfg.NODE_ID, args.proc_id))


if __name__ == "__main__":
    main()
