"""A wrapper env that handles multiple tasks from different envs."""

import os
import random
import torch

import gym
import numpy as np
from gym import spaces
from gym import utils

try:
    from metamorph.config import cfg
    from metamorph.envs.modules.agent import create_agent_xml
    from metamorph.envs.tasks.unimal import UnimalEnv
    from metamorph.utils import file as fu
    from metamorph.utils import spaces as spu
except:
    import sys
    sys.path.append("..")
    from metamorph.metamorph.config import cfg
    from metamorph.metamorph.envs.modules.agent import create_agent_xml
    from metamorph.metamorph.envs.tasks.unimal import UnimalEnv
    from metamorph.metamorph.utils import file as fu
    from metamorph.metamorph.utils import spaces as spu



class MultiEnvWrapper(utils.EzPickle):
    def __init__(self, env, env_idx):
        # Identify the idx of the env within N subproc envs
        self.multi_env_idx = env_idx
        self._env = env
        self._active_unimal_idx = None
        self._unimal_seq = None
        self._unimal_seq_idx = -1
        self.num_steps = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        if self.num_steps == 0:
            self._unimal_seq = self._get_unimal_seq()
            self._unimal_seq_idx = -1

        self._unimal_seq_idx += 1
        if self._unimal_seq_idx >= len(self._unimal_seq):
            self._unimal_seq_idx = 0
        try:
            self._active_unimal_idx = self._unimal_seq[self._unimal_seq_idx]
        except:
            print("ERROR: ", len(self._unimal_seq), self._unimal_seq_idx)
            print(self._unimal_seq)
            exit(0)
        try:
            unimal_id = cfg.ENV.WALKERS[self._active_unimal_idx]
            # print(f"unimal_id: {unimal_id}, active_unimal_idx: {self._active_unimal_idx}, unimal_seq_idx: {self._unimal_seq_idx}")
        except:
            print("ERROR: ", self._active_unimal_idx, len(cfg.ENV.WALKERS), len(self._unimal_seq))
        self._env.update(unimal_id, self._active_unimal_idx)
        obs = self._env.reset() # keys: proprioceptive, edges, obs_padding_mask, act_padding_mask, latent_morph
        # print(obs.keys())
        # print(f"obs propreceptive shape: {obs['proprioceptive'].shape}, edges shape: {obs['edges'].shape}, obs_padding_mask shape: {obs['obs_padding_mask'].shape}, act_padding_mask shape: {obs['act_padding_mask'].shape}, latent_morph shape: {obs['latent_morph'].shape}")
        return obs

    def reset_one_unimal(self, data):
        updated_unimal_id = data
        # _unimal_seq is changing every cfg.PPO.TIMESTEPS
        # print(f"updated_unimal_id: {updated_unimal_id}, unimal_id: {unimal_id} curr_unimal_id: {self.unimal_id}")
        if str(updated_unimal_id) == str(self.unimal_id):
            self._env.update_xml(updated_unimal_id)
            obs = self._env.reset()
            # print(f"obs propreceptive shape: {obs['proprioceptive'].shape}, edges shape: {obs['edges'].shape}, obs_padding_mask shape: {obs['obs_padding_mask'].shape}, act_padding_mask shape: {obs['act_padding_mask'].shape}, latent_morph shape: {obs['latent_morph'].shape}")
            return obs
        else:
            # no need to reset
            obs = self._env.get_dummy_obs()
            # print(f"dummy_obs propreceptive shape: {obs['proprioceptive'].shape}, edges shape: {obs['edges'].shape}, obs_padding_mask shape: {obs['obs_padding_mask'].shape}, act_padding_mask shape: {obs['act_padding_mask'].shape}, latent_morph shape: {obs['latent_morph'].shape}")
            return obs

    def step(self, action):
        self.num_steps += 1
        if self.num_steps % cfg.PPO.TIMESTEPS == 0:
            self._unimal_seq = self._get_unimal_seq()
            self._unimal_seq_idx = -1

        return self._env.step(action)
    
    def update_unimal(self, data):
        all_xml_str, cur_iter = data
        # Update whole unimal xmls which each environment is using
        for idx in list(set(self._unimal_seq)):
            self._env.update_one_unimal(idx, all_xml_str[idx], cur_iter)

    def update_one_unimal(self, data):
        idx, xml_str, cur_iter = data
        self._env.update_one_unimal(idx, xml_str, cur_iter)

    def close(self):
        self._env.close()

    def _get_unimal_seq(self):
        path = os.path.join(cfg.OUT_DIR, "sampling.json")
        env_seq = fu.load_json(path)
        chunks = fu.chunkify(env_seq, cfg.PPO.NUM_ENVS)
        return chunks[self.multi_env_idx]


class MultiUnimalNodeCentricObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        self.limb_obs_size = (
            self.observation_space["proprioceptive"].shape[0]
            // self.metadata["num_limbs"]
        )

        self._create_padding_mask()

        delta_obs = {
            "proprioceptive": (self.limb_obs_size * cfg.MODEL.MAX_LIMBS,),
            "edges": (2 * cfg.MODEL.MAX_JOINTS,),
            "obs_padding_mask": (self.obs_padding_mask.size,),
            "obs_padding_cm_mask": (self.obs_padding_cm_mask.size,),
            "act_padding_mask": (self.act_padding_mask.size,),
        }
        self.observation_space = spu.update_obs_space(env, delta_obs)

    def _create_padding_mask(self):
        num_limbs = self.metadata["num_limbs"]
        self.num_limb_pads = self.max_limbs - num_limbs

        # Create src_key_padding_mask for transformer
        obs_padding_mask = [False] * num_limbs + [True] * self.num_limb_pads

        self.obs_padding_mask = np.asarray(obs_padding_mask)
        # Create src_key_padding_mask for cross-modal transformer
        obs_padding_mask = [False] * (num_limbs + 1) + [True] * self.num_limb_pads
        self.obs_padding_cm_mask = np.asarray(obs_padding_mask)

        num_joints = self.metadata["num_joints"]
        self.num_joint_pads = self.max_joints - num_joints

        act_mask = self.metadata["joint_mask_for_node_graph"]
        act_padding_mask = [not _ for _ in act_mask] + [True] * self.num_limb_pads * 2

        if cfg.MIRROR_DATA_AUG:
            self.metadata["act_m_to_o"] = (
                self.metadata["m_to_o"] + [self.max_limbs - 1] * self.num_limb_pads
            )

        self.act_padding_mask = np.asarray(act_padding_mask)

        # Add to metadata, as used in MultiWalkerAction
        self.metadata["act_padding_mask"] = self.act_padding_mask

    def observation(self, obs):
        proprioceptive = obs["proprioceptive"]
        padding = [0.0] * (self.limb_obs_size * self.num_limb_pads)
        
        obs["proprioceptive"] = np.concatenate([proprioceptive, padding]).ravel()
        obs["obs_padding_mask"] = self.obs_padding_mask
        obs["obs_padding_cm_mask"] = self.obs_padding_cm_mask
        obs["act_padding_mask"] = self.act_padding_mask

        edges = obs["edges"]
        # Pad with edges which connect non-exsisten padded observations
        padding = [self.max_limbs - 1] * 2 * self.num_joint_pads
        obs["edges"] = np.concatenate([edges, padding]).ravel()
            
        return obs

    def _create_binary_encoding(self):
        agent_idx = self.metadata["agent_idx"]
        binary_encoding = bin(agent_idx)[2:].zfill(self.limb_obs_size)
        return [int(_) for _ in binary_encoding]

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._create_padding_mask()
        return self.observation(observation)
    
    def get_dummy_obs(self, *args, **kwargs):
        dummy_obs = dict()
        dummy_obs["proprioceptive"] = np.zeros((self.limb_obs_size * cfg.MODEL.MAX_LIMBS,))
        dummy_obs["edges"] = np.zeros((2 * cfg.MODEL.MAX_JOINTS,))
        dummy_padding_mask = np.asarray([False] * self.max_limbs)
        dummy_act_padding_mask = np.asarray([False] * self.max_limbs * 2)

        dummy_obs["obs_padding_mask"] = dummy_padding_mask
        dummy_obs["act_padding_mask"] = dummy_act_padding_mask
        if cfg.ENV.TYPE != "ft":
            dummy_obs["hfield"] = np.zeros((1410,))

        return dummy_obs


class MultiUnimalNodeCentricAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._update_action_space()
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

    def _update_action_space(self):
        num_joints = self.metadata["num_joints"]
        num_pads = self.max_limbs * 2 - num_joints
        low, high = self.action_space.low, self.action_space.high
        low = np.concatenate([low, [-1] * num_pads]).astype(np.float32)
        high = np.concatenate([high, [-1] * num_pads]).astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        if (cfg.MIRROR_DATA_AUG and self.metadata["mirrored"]):
            action = action.reshape(-1, 2)
            action = action[self.metadata["act_m_to_o"], :].reshape(-1)

        new_action = action[~self.metadata["act_padding_mask"]]
        return new_action
