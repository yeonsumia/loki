import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import shutil
import random
import networkx as nx
import tarfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from metamorph.config import cfg
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
from metamorph.utils import file as fu
from metamorph.utils import model as mu
from metamorph.utils import optimizer as ou
from metamorph.utils.meter import TrainMeter, PerClassStatTracker, AgentMeter
from metamorph.envs.modules.agent import create_agent_xml

import sys
sys.path.append("..")
from vae.train import NUM_LAYERS, N_TOKENS, D_TOKEN, D_DEPTH, H_DIM, N_HEAD, FACTOR
from vae.model import Model_VAE
from vae.data import VectorDataset

from tools.pkl_2_vec_new import convert_all_pkl_to_vec
from tools.util import BINARY_TOKEN, CONTINUOUS_TOKEN, CATEGORY_TOKEN, TOTAL_CATEGORY, THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS

from derl.derl.envs.morphology import SymmetricUnimal
from derl.derl.utils import xml as xu
from derl.derl.utils import similarity as simu

from derl.derl.config import cfg as derl_cfg

# from torch.utils import tensorboard
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
import wandb
from .buffer import Buffer
from .envs import get_ob_rms, make_vec_envs, make_env, set_ob_rms
from .inherit_weight import restore_from_checkpoint, restore_from_checkpoints_model_soup
from .model import ActorCritic, Agent

class LOKI:
    def __init__(self, args, print_model=True):
        self.device = torch.device(cfg.DEVICE)

        self.num_unimals = cfg.LOKI.NUM_WALKER
        print(f"num_unimals: {self.num_unimals}")

        # load vae model
        vae_model_path = os.path.join("../vae/checkpoints", args.vae_path, "model.pt")
        self.vae_model = Model_VAE(NUM_LAYERS, N_TOKENS, D_TOKEN, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR).to(self.device)
        self.vae_model.load_state_dict(torch.load(vae_model_path, weights_only=True, map_location=self.device))

        # freeze vae model
        for param in self.vae_model.parameters():
            param.requires_grad = False
        
        self.tmp_samples_dir = os.path.join(cfg.ENV.WALKER_DIR, "tmp_samples")
        os.makedirs(self.tmp_samples_dir, exist_ok=True)

        self.tmp_samples_xml_dir = os.path.join(self.tmp_samples_dir, "xml")
        self.tmp_samples_pkl_dir = os.path.join(self.tmp_samples_dir, "pkl")
        os.makedirs(self.tmp_samples_xml_dir, exist_ok=True)
        os.makedirs(self.tmp_samples_pkl_dir, exist_ok=True)

        # print current cluster label
        print(f"Cluster label: {cfg.LOKI.CLUSTER_LABEL} / {cfg.LOKI.NUM_CLUSTERS}")

        # load cluster unimals
        self.cluster_tar_path = f"../data/latent_cluster{cfg.LOKI.NUM_CLUSTERS}_{cfg.LOKI.CLUSTER_LABEL}.tar"
        self.cluster_unimals_pkl_strings = []
        self.cluster_unimals_xml_strings = []
        with tarfile.open(self.cluster_tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    member.name = os.path.basename(member.name)
                    if member.name.endswith(".pkl"):
                        self.cluster_unimals_pkl_strings.append(tar.extractfile(member).read())
                    if member.name.endswith(".xml"):
                        self.cluster_unimals_xml_strings.append(tar.extractfile(member).read())

        print(f"Cluster unimals pkl: {len(self.cluster_unimals_pkl_strings)}")
        print(f"Cluster unimals xml: {len(self.cluster_unimals_xml_strings)}")

        self.mutation_ops = [
            "grow_limb",
            "joint_angle",
            "limb_params",
            "dof",
            "density",
            "delete_limb",
            "gear",
            "none"
        ]

        self.current_cluster_unimal_indices = torch.zeros(3 * self.num_unimals, dtype=torch.long, device=self.device)

        derl_cfg.OUT_DIR = self.tmp_samples_dir
        if cfg.LOKI.INIT_DIR != "":
            cluster_init_xml_path = os.path.join(cfg.LOKI.INIT_DIR, "xml_step", str(cfg.LOKI.RESUME_ITER))
            cluster_init_pkl_path = os.path.join(cfg.LOKI.INIT_DIR, "unimal_pkl_recons_sample", str(cfg.LOKI.RESUME_ITER))
            self.cluster_init_unimals_pkl_strings = []
            self.cluster_init_unimals_xml_strings = []
            for i in range(self.num_unimals):
                pkl_file = os.path.join(cluster_init_pkl_path, f"{i}.pkl")
                xml_file = os.path.join(cluster_init_xml_path, f"{i}.xml")
                # self.cluster_final_unimals_pkl_strings.append(fu.load_pickle(pkl_file))
                # get xml string
                with open(xml_file, "rb") as f:
                    xml_string = f.read()

                self.cluster_init_unimals_xml_strings.append(xml_string)

                # get pkl string
                with open(pkl_file, "rb") as f:
                    pkl_string = f.read()
                self.cluster_init_unimals_pkl_strings.append(pkl_string)

            print(f"Cluster final unimals pkl: {len(self.cluster_init_unimals_pkl_strings)}")
            print(f"Cluster final unimals xml: {len(self.cluster_init_unimals_xml_strings)}")

            self.initialize_kmeans_cluster_xml_from_dir()
        else:
            self.initialize_kmeans_cluster_xml()
        self.save_distance(cur_iter=0)
        # self.get_best_k_samples([str(i) for i in range(cfg.LOKI.NUM_WALKER)], [0 for i in range(cfg.LOKI.NUM_WALKER)], cur_iter=0, num_samples=cfg.LOKI.NUM_WALKER * 8, initialize=True)
        # Create vectorized envs
        # if cfg.PPO.CHECKPOINT_PATH and cfg.LOKI.INIT_DIR == "":
        #     # load final population
        #     self.update_agents_directory("final", 1218, remove_prev=False)

        self.envs = make_vec_envs()
        self.file_prefix = cfg.ENV_NAME

        derl_cfg.OUT_DIR = os.path.join(cfg.ENV.WALKER_DIR, "xml_step")

        self.actor_critic = globals()[cfg.MODEL.ACTOR_CRITIC](
            self.envs.observation_space, self.envs.action_space
        )

        # Used while using train_ppo.py
        if cfg.PPO.CHECKPOINT_PATH:
            if cfg.PPO.MODEL_SOUP:
                ob_rms = restore_from_checkpoints_model_soup(self.actor_critic)
            else:
                ob_rms = restore_from_checkpoint(self.actor_critic)
            set_ob_rms(self.envs, ob_rms)

            if cfg.LOKI.INIT_DIR == "":
                cluster_final_xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml_step", str(cfg.LOKI.RESUME_ITER))
                cluster_final_pkl_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_pkl_recons_sample", str(cfg.LOKI.RESUME_ITER))
                self.cluster_final_unimals_pkl_strings = []
                self.cluster_final_unimals_xml_strings = []
                for i in range(self.num_unimals):
                    pkl_file = os.path.join(cluster_final_pkl_path, f"{i}.pkl")
                    xml_file = os.path.join(cluster_final_xml_path, f"{i}.xml")
                    # self.cluster_final_unimals_pkl_strings.append(fu.load_pickle(pkl_file))
                    # get xml string
                    with open(xml_file, "rb") as f:
                        xml_string = f.read()

                    self.cluster_final_unimals_xml_strings.append(xml_string)

                    # get pkl string
                    with open(pkl_file, "rb") as f:
                        pkl_string = f.read()
                    self.cluster_final_unimals_pkl_strings.append(pkl_string)

                print(f"Cluster final unimals pkl: {len(self.cluster_final_unimals_pkl_strings)}")
                print(f"Cluster final unimals xml: {len(self.cluster_final_unimals_xml_strings)}")

        if print_model:
            print(self.actor_critic)
            print("Num params: {}".format(mu.num_params(self.actor_critic)))

        self.actor_critic.to(self.device)
        self.agent = Agent(self.actor_critic)

        # Setup experience buffer
        # print(self.envs.action_space)
        self.buffer = Buffer(self.envs.observation_space, self.envs.action_space.shape)
        # Optimizer for both actor and critic
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS
        )

        self.train_meter = TrainMeter()
        self.reward_tracker = PerClassStatTracker()
        # self.writer = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tensorboard"), flush_secs=30)
        # Get the param name for log_std term, can vary depending on arch
        for name, param in self.actor_critic.state_dict().items():
            if "log_std" in name:
                self.log_std_param = name
                break

        # for name, weight in self.actor_critic.named_parameters():
        #     print(name, weight.requires_grad)

        self.fps = 0


    def train(self):
        self.save_sampled_agent_seq(0)
        obs = self.envs.reset()
        self.buffer.to(self.device)
        self.start = time.time()
        prev_iter = 0
        worst_agent = []

        self.worst_agents_log = defaultdict(list)
        start_iter = cfg.LOKI.RESUME_ITER+1
        for cur_iter in range(start_iter, cfg.PPO.MAX_ITERS):
            if cfg.LOKI.UPDATE_WALKER and cur_iter >= cfg.LOKI.DROP_WARMUP and cur_iter % cfg.LOKI.DROP_FREQ == 0:
            # if cur_iter == 0 or (cur_iter // cfg.PPO.AGENT_UPDATE_FREQ) % 2 == 1:
                # find worst agent
                self.envs.log()
                self._log_stats(cur_iter) # before
                worst_agent, worst_rewards = self.train_meter.get_n_worst_agents(cfg.LOKI.NUM_DROP_WALKER)
                print(f"{cfg.LOKI.NUM_DROP_WALKER} worst agents: {worst_agent}")
                if worst_agent is None or len(worst_agent) == 0:
                    print("skip update")
                else:
                    self.update_agents_directory(cur_iter, prev_iter)
                    # Get good samples
                    xml_list, pkl_list = self.get_best_k_cluster_samples(worst_agent, worst_rewards, cur_iter)
                    # Caculate diversity metric
                    self.save_distance(cur_iter)

                    # print(f"Updating agents: {update_walker_indice}")
                    for i, walker_id in enumerate(worst_agent):
                        walker_idx = cfg.ENV.WALKERS.index(walker_id)
                        # Update agent design
                        new_xml = xml_list[i]
                        if new_xml is None:
                            print(f"[Step {self.env_steps_done(cur_iter)}] Skip updating walker {walker_id}")
                            continue
                        
                        data = (walker_idx, new_xml, cur_iter)
                        # self._log_stats(cur_iter) # before
                        self.envs.update_one_unimal(data)
                        tmp_obs = self.envs.reset_one_unimal(walker_idx)
                        print(f"[Step {self.env_steps_done(cur_iter)}] Update Worst Agent: {walker_id}")
                        self.worst_agents_log[cur_iter].append(walker_id)
                        for i in range(cfg.PPO.NUM_ENVS):
                            zeros = torch.zeros_like(obs['edges'][i], device=self.device)
                            # check the tensor is not zero tensor
                            check = torch.from_numpy(tmp_obs['edges'][i]).to(self.device)
                            if torch.all(torch.eq(check, zeros)):
                                continue
                            print(f"    Update env {i}")
                            for key in obs.keys():
                                obs[key][i] = torch.from_numpy(tmp_obs[key][i]).to(self.device)

                prev_iter = cur_iter
                # self._log_stats(cur_iter)
                with open(os.path.join(cfg.ENV.WALKER_DIR, 'dropped_agents.json'), 'w') as f:
                    json.dump(self.worst_agents_log, f, indent=4)
    
            print("!! ", cur_iter, cfg.PPO.MAX_ITERS, flush=True) # print all strings from buffer at this point
            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break

            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr)

            for step in range(cfg.PPO.TIMESTEPS):
                # Sample actions
                val, act, logp = self.agent.act(obs)
                # print(f"act: {act.shape}")
                next_obs, reward, done, infos = self.envs.step(act)

                self.train_meter.add_ep_info(infos)

                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device,
                )
                timeouts = torch.tensor(
                    [[0.0] if "timeout" in info.keys() else [1.0] for info in infos],
                    dtype=torch.float32,
                    device=self.device,
                )

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts)
                obs = next_obs

            next_val = self.agent.get_value(obs)
            self.buffer.compute_returns(next_val)
            self.train_on_batch(cur_iter)
            self.save_sampled_agent_seq(cur_iter)

            self.train_meter.update_mean()
            env_steps_done = self.env_steps_done(cur_iter)
            if len(self.train_meter.mean_ep_rews["reward"]):
                cur_rew = self.train_meter.mean_ep_rews["reward"][-1]
                # self.writer.add_scalar(
                #     'Reward', cur_rew, env_steps_done
                # )
                wandb.log({"Reward": cur_rew}, step=env_steps_done)
            if (
                cur_iter > 0
                and cur_iter % cfg.LOG_PERIOD == 0
                and cfg.LOG_PERIOD > 0
            ):
                self._log_stats(cur_iter)
                self.save_model()

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self, cur_iter):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5) # shape: [PPO.TIMESTEPS, PPO.NUM_ENVS, 1]

        for _ in range(cfg.PPO.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)

            for batch in batch_sampler:
                # Reshape to do in a single forward pass for all steps
                val, _, logp, ent = self.actor_critic(batch["obs"], batch["act"])
                clip_ratio = cfg.PPO.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])
                approx_kl = (batch["logp_old"] - logp).mean().item()

                if approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01:
                    return

                surr1 = ratio * batch["adv"]

                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                surr2 *= batch["adv"]

                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.PPO.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(
                        -clip_ratio, clip_ratio
                    )
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                self.optimizer.zero_grad()

                loss = val_loss * cfg.PPO.VALUE_COEF
                loss += pi_loss
                loss += -ent * cfg.PPO.ENTROPY_COEF
                loss.backward()

                # Log training stats
                norm = nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM
                )
                self.train_meter.add_train_stat("grad_norm", norm.item())

                log_std = (
                    self.actor_critic.state_dict()[self.log_std_param].cpu().numpy()[0]
                )
                std = np.mean(np.exp(log_std))
                self.train_meter.add_train_stat("std", float(std))

                self.train_meter.add_train_stat("approx_kl", approx_kl)
                self.train_meter.add_train_stat("pi_loss", pi_loss.item())
                self.train_meter.add_train_stat("val_loss", val_loss.item())
                self.train_meter.add_train_stat("ratio", ratio.mean().item())
                self.train_meter.add_train_stat("surr1", surr1.mean().item())
                self.train_meter.add_train_stat("surr2", surr2.mean().item())

                self.optimizer.step()

        # Save weight histogram
        # if cfg.SAVE_HIST_WEIGHTS:
        #     for name, weight in self.actor_critic.named_parameters():
        #         self.writer.add_histogram(name, weight, cur_iter)
        #         wandb.log({f"histogram/{name}": wandb.Histogram(weight.data.cpu().numpy())}, step=self.env_steps_done(cur_iter))
        #         try:
        #             self.writer.add_histogram(f"{name}.grad", weight.grad, cur_iter)
        #             wandb.log({f"histogram/{name}.grad": wandb.Histogram(weight.grad.data.cpu().numpy())}, step=self.env_steps_done(cur_iter))

        #         except NotImplementedError:
        #             # If layer does not have .grad move on
        #             continue

    def initialize_kmeans_cluster_xml(self, cur_iter=0):

        # latent_path = os.path.join(cfg.ENV.WALKER_DIR, "latents", str(cur_iter))
        xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml_step", str(cur_iter))
        pkl_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_pkl_recons_sample", str(cur_iter))

        # os.makedirs(latent_path, exist_ok=True)
        os.makedirs(xml_path, exist_ok=True)
        os.makedirs(pkl_path, exist_ok=True)

        # initialize walker name list
        cfg.ENV.WALKERS = []

        with torch.no_grad():
            # randomly select a sample from the cluster
            # unimal_indices = torch.randint(0, len(self.cluster_unimals_pkl_path), (self.num_unimals,))
            unimal_indices = torch.randperm(len(self.cluster_unimals_pkl_strings))[:self.num_unimals * 3]
            self.current_cluster_unimal_indices = unimal_indices
            
            # mutate the selected unimals
            for i in range(self.num_unimals * 3):
                walker_name = str(i)
                unimal_pkl_string = self.cluster_unimals_pkl_strings[unimal_indices[i]]
                unimal_xml_string = self.cluster_unimals_xml_strings[unimal_indices[i]]
                unimal = SymmetricUnimal(id_=walker_name, init_pkl=unimal_pkl_string, init_xml=unimal_xml_string)

                # save xml/pkl to current iteration directory
                unimal.save_random(self.tmp_samples_xml_dir, self.tmp_samples_pkl_dir)

                cfg.ENV.WALKERS.append(walker_name)

        G = simu.create_graph_from_uids(
            None, cfg.ENV.WALKERS, "geom_orientation", graph_type="species"
        )

        cc = list(nx.connected_components(G))

        # check if there is same topology morph
        unimals_to_remove = []
        unimals_to_keep = []
        for same_unimals in cc:
            if len(same_unimals) == 1:
                unimals_to_keep.extend(list(same_unimals))
                continue
            remove_unimals = list(same_unimals)
            unimals_to_keep.append(remove_unimals[0])
            remove_unimals = remove_unimals[1:]
            unimals_to_remove.extend(remove_unimals)

        assert len(unimals_to_keep) >= self.num_unimals
        unimals_to_keep, unimals_to_remove = unimals_to_keep[:self.num_unimals], unimals_to_remove + unimals_to_keep[self.num_unimals:]
        
        print(f"Get {len(unimals_to_keep)} unique unimals")
        print(f"Remove {len(unimals_to_remove)} unimals")

        # remove unimals
        for unimal in unimals_to_remove:
            xml_file = os.path.join(self.tmp_samples_xml_dir, f"{unimal}.xml")
            pkl_file = os.path.join(self.tmp_samples_pkl_dir, f"{unimal}.pkl")
            os.remove(xml_file)
            os.remove(pkl_file)

        # update walker list
        cfg.ENV.WALKERS = []
        for i in range(self.num_unimals):
            cfg.ENV.WALKERS.append(str(i))

        # save unique {self.num_unimals} unimals to xml_path, pkl_path
        for i, unimal in enumerate(unimals_to_keep):
            xml_file = os.path.join(self.tmp_samples_xml_dir, f"{unimal}.xml")
            pkl_file = os.path.join(self.tmp_samples_pkl_dir, f"{unimal}.pkl")
            shutil.copy(xml_file, os.path.join(xml_path, f"{i}.xml"))
            shutil.copy(pkl_file, os.path.join(pkl_path, f"{i}.pkl"))

        # remove all xml files in tmp_samples_dir
        for file in os.listdir(self.tmp_samples_xml_dir):
            if file.endswith(".xml"):
                os.remove(os.path.join(self.tmp_samples_xml_dir, file))
        for file in os.listdir(self.tmp_samples_pkl_dir):
            if file.endswith(".pkl"):
                os.remove(os.path.join(self.tmp_samples_pkl_dir, file))

        print(f"Finish initializing {self.num_unimals} unique walkers")

        return


    def initialize_kmeans_cluster_xml_from_dir(self, cur_iter=0):

        # latent_path = os.path.join(cfg.ENV.WALKER_DIR, "latents", str(cur_iter))
        xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml_step", str(cur_iter))
        pkl_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_pkl_recons_sample", str(cur_iter))

        # os.makedirs(latent_path, exist_ok=True)
        os.makedirs(xml_path, exist_ok=True)
        os.makedirs(pkl_path, exist_ok=True)

        # initialize walker name list
        cfg.ENV.WALKERS = []
        for i in range(self.num_unimals):
            cfg.ENV.WALKERS.append(str(i))
        
        # initialize walkers from cfg.LOKI.INIT_DIR
        for i in range(self.num_unimals):
            walker_name = str(i)
            unimal_pkl_string = self.cluster_init_unimals_pkl_strings[i]
            unimal_xml_string = self.cluster_init_unimals_xml_strings[i]
            unimal = SymmetricUnimal(id_=walker_name, init_pkl=unimal_pkl_string, init_xml=unimal_xml_string)
            # save xml/pkl to current iteration directory
            unimal.save_random(xml_path, pkl_path)

        # remove all xml files in tmp_samples_dir
        print(f"Finish initializing {self.num_unimals} walkers from dir")

        return

    def get_best_k_cluster_samples(self, walker_indice, worst_rewards, cur_iter, num_samples=-1, initialize=False):

        if num_samples == -1:
            num_samples = cfg.LOKI.SAMPLE_SIZE

        xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml_step", str(cur_iter))
        pkl_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_pkl_recons_sample", str(cur_iter))

        walker_indice = [str(idx) for idx in walker_indice]
        # update_walker_indice = walker_indice.copy()
        # get cluster samples
        print(f"Get {num_samples} cluster samples from {len(self.cluster_unimals_pkl_strings)} unimals")
        # unimal_samples_indice = torch.randint(0, len(self.cluster_unimals_pkl_path), (num_samples,))
        unimal_samples_indice = torch.randperm(len(self.cluster_unimals_pkl_strings))[:num_samples]
        unimal_samples_pkl_strings = [self.cluster_unimals_pkl_strings[idx] for idx in unimal_samples_indice]
        unimal_samples_xml_strings = [self.cluster_unimals_xml_strings[idx] for idx in unimal_samples_indice]

        # mutate each sample and save into tmp folder
        for i, (unimal_pkl_str, unimal_xml_str) in enumerate(zip(unimal_samples_pkl_strings, unimal_samples_xml_strings)):
            unimal = SymmetricUnimal(id_=str(i), init_pkl=unimal_pkl_str, init_xml=unimal_xml_str)
            if cfg.LOKI.MUTATE_SAMPLE:
                # randomly choose from self.mutation_op
                op = random.choice(self.mutation_ops)
                while (unimal.num_limbs <= 5 and op == "delete_limb") or (unimal.num_limbs == 11 and op == "grow_limb"):
                    op = random.choice(self.mutation_ops)
                unimal.mutate(op)
            unimal.save_random(self.tmp_samples_xml_dir, self.tmp_samples_pkl_dir)

            # move xml to xml_path temporarily
            shutil.copyfile(os.path.join(self.tmp_samples_xml_dir, "{}.xml".format(i)), os.path.join(xml_path, f"tmp_{i}.xml"))
        
        sampled_ids = [str(id_) for id_ in range(num_samples)]
        walker_rewards = self.eval_random_sample(sampled_ids, num_samples=num_samples, initialize=initialize)

        # find best samples
        k = len(walker_indice)
        walker_rewards = np.array(walker_rewards)
        best_walker_indice = np.argsort(walker_rewards)[::-1]

        xml_list = []
        pkl_list = []
        best_k_walker_indice = best_walker_indice[:k]
        for i, (worst_idx, new_idx) in enumerate(zip(walker_indice, best_k_walker_indice)):
            # if cfg.LOKI.INIT_DIR != "" and worst_rewards[i] >= walker_rewards[new_idx]:
            #     print(f"Skip {worst_idx} -> tmp_{sampled_ids[new_idx]} (Reward: {walker_rewards[new_idx]})")
            #     xml_list.append(None)
            #     pkl_list.append(None)
            #     continue
            # print best walker index & reward
            print(f"Replace {worst_idx} -> tmp_{sampled_ids[new_idx]} (Reward: {walker_rewards[new_idx]})")
            worst_idx = str(worst_idx)
            new_idx = str(new_idx)
            # copy sample xml to xml_path
            new_xml, new_pkl = self.replace_worst_to_new(xml_path, pkl_path, worst_idx, new_idx)
            # create new xmls
            xml_list.append(new_xml)
            # save pkl
            pkl_list.append(new_pkl)
        
        print(f"Return best {len(xml_list)} samples")
        return xml_list, pkl_list


    def replace_worst_to_new(self, xml_path, pkl_path, worst_str, new_str):
        # copy sample xml to xml_path
        walker_xml_save_path = os.path.join(xml_path, "{}.xml".format(worst_str))
        shutil.copyfile(os.path.join(self.tmp_samples_xml_dir, "{}.xml".format(new_str)), walker_xml_save_path)

        # copy sample pkl to pkl_path
        walker_pkl_save_path = os.path.join(pkl_path, "{}.pkl".format(worst_str))
        shutil.copyfile(os.path.join(self.tmp_samples_pkl_dir, "{}.pkl".format(new_str)), walker_pkl_save_path)

        return create_agent_xml(walker_xml_save_path), fu.load_pickle(walker_pkl_save_path)
    

    def update_agents_directory(self, cur_iter, prev_iter, remove_prev=True):
        # latent_path = os.path.join(cfg.ENV.WALKER_DIR, "latents", str(cur_iter))
        xml_path = os.path.join(cfg.ENV.WALKER_DIR, "xml_step", str(cur_iter))
        pkl_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_pkl_recons_sample", str(cur_iter))
        
        # os.makedirs(latent_path, exist_ok=True)
        os.makedirs(xml_path, exist_ok=True)
        os.makedirs(pkl_path, exist_ok=True)

        # copy from initial path (cur_iter = 0)
        # os.system(f"cp {os.path.join(cfg.ENV.WALKER_DIR, 'latents', str(prev_iter), '*.pt')} {os.path.join(cfg.ENV.WALKER_DIR, 'latents', str(cur_iter))}")
        os.system(f"cp {os.path.join(cfg.ENV.WALKER_DIR, 'xml_step', str(prev_iter), '*.xml')} {os.path.join(cfg.ENV.WALKER_DIR, 'xml_step', str(cur_iter))}")
        os.system(f"cp {os.path.join(cfg.ENV.WALKER_DIR, 'unimal_pkl_recons_sample', str(prev_iter), '*.pkl')} {os.path.join(cfg.ENV.WALKER_DIR, 'unimal_pkl_recons_sample', str(cur_iter))}")

        # remove previous xml/pkl/pt directories
        if remove_prev and prev_iter > 0:
            os.system(f"rm -r {os.path.join(cfg.ENV.WALKER_DIR, 'xml_step', str(prev_iter))}")
            os.system(f"rm -r {os.path.join(cfg.ENV.WALKER_DIR, 'unimal_pkl_recons_sample', str(prev_iter))}")
            # os.system(f"rm -r {os.path.join(cfg.ENV.WALKER_DIR, 'latents', str(prev_iter))}")

    def save_model(self, path=None):
        if not path:
            path = os.path.join(cfg.OUT_DIR, self.file_prefix + ".pt")
        torch.save([self.actor_critic, get_ob_rms(self.envs)], path)

    def _log_stats(self, cur_iter):
        self._log_fps(cur_iter)
        self.train_meter.log_stats()

    def _log_fps(self, cur_iter, log=True):
        env_steps = self.env_steps_done(cur_iter)
        end = time.time()
        self.fps = int(env_steps / (end - self.start))
        if log:
            print(
                "Updates {}, num timesteps {}, FPS {}".format(
                    cur_iter, env_steps, self.fps
                )
            )

    def env_steps_done(self, cur_iter):
        return (cur_iter + 1) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS

    def save_rewards(self, path=None, hparams=None):
        if not path:
            file_name = "{}_results.json".format(self.file_prefix)
            path = os.path.join(cfg.OUT_DIR, file_name)

        self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
        stats = self.train_meter.get_stats()
        stats["fps"] = self.fps
        fu.save_json(stats, path)

        # Save hparams when sweeping
        # if hparams:
        #     # Remove hparams which are of type list as tensorboard complains
        #     # on saving it's not a supported type.
        #     hparams_to_save = {
        #         k: v for k, v in hparams.items() if not isinstance(v, list)
        #     }
        #     final_env_reward = np.mean(stats["__env__"]["reward"]["reward"][-100:])
        #     self.writer.add_hparams(hparams_to_save, {"reward": final_env_reward})

        # self.writer.close()

    def save_distance(self, cur_iter):
        pkl_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_pkl_recons_sample", str(cur_iter))
        vec_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_vec_recons_sample", str(cur_iter))
        if not os.path.exists(vec_path):
            os.makedirs(vec_path)

        # pkl to vec
        convert_all_pkl_to_vec(pkl_path, vec_path)

        test_dataset = VectorDataset(directory=vec_path, input_dim=D_TOKEN)

        test_data_file_list = test_dataset.file_list
        print("test data file list", len(test_dataset))
        
        test_vector = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))], dim=0).to(self.device)
        test_mask = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))], dim=0).to(self.device)

        # test_data = torch.cat((test_vector, test_mask), dim=-1)
        # print("test data shape", test_vector.shape)
        # print("test mask shape", test_mask.shape)

        test_vector = test_vector.masked_fill(test_mask == 0, 0)

        data_con, data_cat, data_binary, data_depth = test_vector.float().split([CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, 1], dim=-1)
        # data_num, data_depth = latent_z.float().split([D_TOKEN - 1, 1], dim=-1)
        # mask_num, mask_depth = mask.split([D_TOKEN - 1, 1], dim=-1)
        
        num_classes_list = [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS]

        one_hot_x_cat = torch.cat([one_hot(data_cat[:, :, i].long(), num_classes=num_classes_list[i]) for i in range(CATEGORY_TOKEN)], dim=-1)

        data_num = torch.cat([data_con, one_hot_x_cat, data_binary], dim=-1)

        _, mu_z, std_z = self.vae_model.VAE(data_num, data_depth)
        z = self.vae_model.VAE.reparameterize(mu_z, std_z)

        distances = self.get_distance_nearest_neighbors(z, k=self.num_unimals-1)
        distances = distances.mean(dim=-1)

        # save distances
        distance_path = os.path.join(cfg.ENV.WALKER_DIR, "unimal_distance", str(cur_iter))
        os.makedirs(distance_path, exist_ok=True)

        # save distance vector
        distance_vector_path = os.path.join(distance_path, "distance_vector.pt")
        torch.save(distances, distance_vector_path)

        # log distance vector
        env_steps_done = self.env_steps_done(cur_iter)
        wandb.log({"mean_distance": torch.mean(distances).item(), "median_distance": torch.median(distances).item()}, step=env_steps_done)

    def get_distance_nearest_neighbors(self, X, k=3):
        """
        Get the distance to the k nearest neighbors for each point in X.
        Input:
            X: [batch_size, num_features] tensor
        Output:
            distances: [batch_size, k] tensor
        """
        X = X.view(1, X.size(0), -1)
        print("X shape", X.shape)
        # Compute the pairwise distance matrix
        D = torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist').squeeze(0)
        # print("D shape", D.shape)

        # Get the indices of the k nearest neighbors
        _, indices = torch.topk(D, k=k+1, largest=False, sorted=True)
        print("indices shape", indices.shape)
        # Get the distance to the k-th nearest neighbor
        distances = [torch.index_select(D[i], 0, indices[i, :])[1:] for i in range(D.size(0))]
        
        distances = torch.stack(distances, dim=0)

        return distances

    def save_video_orig(self, save_dir):
        env = make_vec_envs(training=False, norm_rew=False, save_video=True,)
        set_ob_rms(env, get_ob_rms(self.envs))

        env = VecVideoRecorder(
            env,
            save_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=cfg.PPO.VIDEO_LENGTH,
            file_prefix=self.file_prefix,
        )
        obs = env.reset()

        for _ in range(cfg.PPO.VIDEO_LENGTH + 1):
            _, act, _ = self.agent.act(obs)
            obs, _, _, _ = env.step(act)

        env.close()
        # remove annoying meta file created by monitor
        os.remove(os.path.join(save_dir, "{}_video.meta.json".format(self.file_prefix)))

    def save_video(self, save_dir, n_videos=15):
        all_videos = []
        print("Saving videos")
        for video_idx in range(n_videos):
            env = make_vec_envs(training=False, norm_rew=False, save_video=True, video_idx=video_idx)
            set_ob_rms(env, get_ob_rms(self.envs))
            
            file_prefix = f'{self.file_prefix}_{video_idx}_csr'
            print(f"Saving video {video_idx} with file prefix {file_prefix}")

            env = VecVideoRecorder(
                env,
                save_dir,
                record_video_trigger=lambda x: x == 0,
                video_length=cfg.PPO.VIDEO_LENGTH,
                file_prefix=file_prefix,
            )
            obs = env.reset()

            for _ in range(cfg.PPO.VIDEO_LENGTH + 1):
                _, act, _ = self.agent.act(obs)
                obs, _, _, _ = env.step(act)
                
            # wandb log video
            all_videos.append(wandb.Video(os.path.join(save_dir, f"{file_prefix}_video.mp4"), fps=4))
            
            env.close()
            # remove annoying meta file created by monitor
            os.remove(os.path.join(save_dir, "{}_video.meta.json".format(file_prefix)))
        wandb.log({"videos": all_videos})
    
    def eval_random_sample(self, sampled_ids, num_samples, initialize=False):
        self.actor_critic.eval()
        
        env = make_vec_envs(training=False, norm_rew=False, num_env=num_samples, tmp_sample=True, sampled_ids=sampled_ids)
        set_ob_rms(env, get_ob_rms(self.envs))

        id_to_idx = {id_: idx for idx, id_ in enumerate(sampled_ids)}
        
        reward = [[] for _ in range(num_samples)]
        obs = env.reset()
        with torch.no_grad():
            for _ in range(200): #  if cfg.LOKI.INIT_DIR == "" else 1000): # max_episode_length
                _, act, _ = self.agent.act(obs, num_samples=num_samples)
                obs, rew, done, infos = env.step(act)
                for info in infos:
                    if "episode" in info.keys():
                        reward[id_to_idx[info["name"]]].append(info["episode"]["r"])

        env.close()

        # print reward length
        # for i, r in enumerate(reward):
        #     print(f"Reward for {sampled_ids[i]}: {len(r)}")

        median_reward = [np.median(r) for r in reward]

        return median_reward

    def eval_random_sample_full(self, sampled_ids, num_samples, initialize=False):
        self.actor_critic.eval()
        
        env = make_vec_envs(training=False, norm_rew=False, num_env=num_samples, tmp_sample_full=True, sampled_ids=sampled_ids)
        set_ob_rms(env, get_ob_rms(self.envs))

        id_to_idx = {id_: idx for idx, id_ in enumerate(sampled_ids)}
        
        reward = [[] for _ in range(num_samples)]
        obs = env.reset()
        with torch.no_grad():
            for _ in range(1000): # max_episode_length
                _, act, _ = self.agent.act(obs, num_samples=num_samples)
                obs, rew, done, infos = env.step(act)
                for info in infos:
                    if "episode" in info.keys():
                        reward[id_to_idx[info["name"]]].append(info["episode"]["r"])

        env.close()

        # print reward length
        # for i, r in enumerate(reward):
        #     print(f"Reward for {sampled_ids[i]}: {len(r)}")

        median_reward = [np.median(r) for r in reward]

        return median_reward
    
    def save_sampled_agent_seq(self, cur_iter):
        num_agents = len(cfg.ENV.WALKERS)

        if num_agents <= 1:
            return

        if cfg.ENV.TASK_SAMPLING == "uniform_random_strategy":
            ep_lens = [1000] * num_agents
        elif cfg.ENV.TASK_SAMPLING == "balanced_replay_buffer":
            # For a first couple of iterations do uniform sampling to ensure
            # we have good estimate of ep_lens
            if cur_iter < 30:
                ep_lens = [1000] * num_agents
                ep_reward = [1000] * num_agents
            else:
                if cfg.TASK_SAMPLING.AVG_TYPE == "ema":
                    ep_lens = [
                        np.mean(self.train_meter.agent_meters[agent].ep_len_ema)
                        for agent in cfg.ENV.WALKERS
                    ]
                elif cfg.TASK_SAMPLING.AVG_TYPE == "moving_window":
                    ep_lens = [
                        np.mean(self.train_meter.agent_meters[agent].ep_len)
                        for agent in cfg.ENV.WALKERS
                    ]
                elif cfg.TASK_SAMPLING.AVG_TYPE == "reward":
                    ep_reward = [
                        np.mean(self.train_meter.agent_meters[agent].ep_rew["reward"])
                        for agent in cfg.ENV.WALKERS
                    ]
        if cfg.TASK_SAMPLING.AVG_TYPE == "reward":
            # Inverse proportion to reward
            probs = [1000.0 / r if r > 1 else 2000.0 for r in ep_reward]
        else:
            probs = [1000.0 / l if l > 0 else 1000.0 for l in ep_lens]
        probs = np.power(probs, cfg.TASK_SAMPLING.PROB_ALPHA)
        probs = [p / sum(probs) for p in probs]

        # Estimate approx number of episodes each subproc env can rollout
        avg_ep_len = np.mean([
            np.mean(self.train_meter.agent_meters[agent].ep_len)
            for agent in cfg.ENV.WALKERS
        ])
        # In the start the arrays will be empty
        if np.isnan(avg_ep_len):
            avg_ep_len = 100
        ep_per_env = cfg.PPO.TIMESTEPS / avg_ep_len
        # Task list size (multiply by 8 as padding)
        size = int(ep_per_env * cfg.PPO.NUM_ENVS * 50)
        task_list = np.random.choice(range(0, num_agents), size=size, p=probs)
        task_list = [int(_) for _ in task_list]
        path = os.path.join(cfg.OUT_DIR, "sampling.json")
        fu.save_json(task_list, path)