from collections import defaultdict
from collections import deque

import numpy as np

from metamorph.config import cfg


class AgentMeter:
    def __init__(self, name):
        self.name = name
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []

        self.ep_rew = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos = deque(maxlen=10)
        self.ep_vel = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_count = 0
        self.ep_len = deque(maxlen=10)
        self.ep_len_ema = -1

    def add_ep_info(self, infos):
        for info in infos:
            # if "episode" in info.keys() and info["name"] == self.name:
            #     print(info["name"], info["episode"]["r"], info["episode"]["l"])
            if info["name"] != self.name:
                continue
            if "episode" in info.keys():
                self.ep_rew["reward"].append(info["episode"]["r"])
                self.ep_count += 1
                self.ep_len.append(info["episode"]["l"])
                if self.ep_count == 10:
                    self.ep_len_ema = np.mean(self.ep_len)
                elif self.ep_count >= 10:
                    alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                    self.ep_len_ema = (
                        alpha * self.ep_len[-1] + (1 - alpha) * self.ep_len_ema
                    )

                for rew_type, rew_ in info["episode"].items():
                    if "__reward__" in rew_type:
                        self.ep_rew[rew_type].append(rew_)

                if "x_pos" in info:
                    self.ep_pos.append(info["x_pos"])
                if "x_vel" in info:
                    self.ep_vel.append(info["x_vel"])
                if "metric" in info:
                    self.ep_metric.append(info["metric"])

    def update_mean(self):
        if len(self.ep_rew["reward"]) == 0:
            return False

        for rew_type, rews_ in self.ep_rew.items():
            self.mean_ep_rews[rew_type].append(round(np.mean(rews_), 2))

        self.mean_pos.append(round(np.mean(self.ep_pos), 2))
        self.mean_vel.append(round(np.mean(self.ep_vel), 2))
        self.mean_metric.append(round(np.mean(self.ep_metric), 2))
        self.mean_ep_len.append(round(np.mean(self.ep_len), 2))
        return True

    def log_stats(self, max_name_len):
        if len(self.ep_rew["reward"]) == 0:
            return
        ep_rew = self.ep_rew["reward"]
        print(
            "Agent {:>{size}}: mean/median reward {:>4.0f}/{:<4.0f}, "
            "min/max reward {:>4.0f}/{:<4.0f}, "
            "#Ep: {:>7.0f}, avg/ema Ep len: {:>4.0f}/{:>4.0f}".format(
                self.name,
                np.mean(ep_rew),
                np.median(ep_rew),
                np.min(ep_rew),
                np.max(ep_rew),
                self.ep_count,
                np.mean(self.ep_len),
                self.ep_len_ema,
                size=max_name_len
            )
        )

    def initialize(self):
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []

        self.ep_rew = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos = deque(maxlen=10)
        self.ep_vel = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_count = 0
        self.ep_len = deque(maxlen=10)
        self.ep_len_ema = -1


class TrainMeter:
    def __init__(self):
        self.agents = cfg.ENV.WALKERS
        self.max_name_len = max([len(a) for a in self.agents])
        self.agent_meters = {agent: AgentMeter(agent) for agent in self.agents}

        # Env stats
        self.train_stats = defaultdict(list)
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []
        self.worst_rews = []
        self.worst_rews_ptr = 0

        self.update_agent_idx = 0

    def add_train_stat(self, stat_type, stat_value):
        self.train_stats[stat_type].append(stat_value)

    def add_ep_info(self, infos):
        for _, agent_meter in self.agent_meters.items():
            agent_meter.add_ep_info(infos)

    def update_mean(self):
        for _, agent_meter in self.agent_meters.items():
            success = agent_meter.update_mean()
            if not success:
                return

        metrics = ["mean_pos", "mean_vel", "mean_metric", "mean_ep_len"]
        for metric in metrics:
            metric_list = []
            for _, agent_meter in self.agent_meters.items():
                metric_list.append(getattr(agent_meter, metric)[-1])

            getattr(self, metric).append(round(np.mean(metric_list), 2))

        rew_types = self.agent_meters[self.agents[0]].mean_ep_rews.keys()

        for rew_type in rew_types:
            rew_list = []
            for _, agent_meter in self.agent_meters.items():
                rew_list.append(agent_meter.mean_ep_rews[rew_type][-1])

            self.mean_ep_rews[rew_type].append(round(np.mean(rew_list), 2))

    def log_stats(self):
        for _, agent_meter in self.agent_meters.items():
            agent_meter.log_stats(self.max_name_len)

        if len(self.mean_ep_rews["reward"]) > 0:
            print("Agent {:>{size}}: mean/------ reward {:>4.0f}, ".format(
                    "__env__",
                    self.mean_ep_rews["reward"][-1],
                    size=self.max_name_len
                )
            )

    def get_stats(self):
        stats = {}
        for agent, agent_meter in self.agent_meters.items():
            stats[agent] = {
                "reward": agent_meter.mean_ep_rews,
                "pos": agent_meter.mean_pos,
                "vel": agent_meter.mean_vel,
                "metric": agent_meter.mean_metric,
                "ep_len": agent_meter.mean_ep_len,
            }

        stats["__env__"] = {
                "reward": self.mean_ep_rews,
                "pos": self.mean_pos,
                "vel": self.mean_vel,
                "metric": self.mean_metric,
                "ep_len": self.mean_ep_len,
        }
        stats["__env__"].update(dict(self.train_stats))
        return stats

    def get_worst_agent(self):
        worst_agent = None
        worst_reward = 1e6
        for agent, agent_meter in self.agent_meters.items():
            if len(agent_meter.mean_ep_rews["reward"]) == 0:
                continue
            if agent_meter.mean_ep_rews["reward"][-1] < worst_reward:
                worst_agent = agent
                worst_reward = agent_meter.mean_ep_rews["reward"][-1]

        # added condition
        if worst_reward < 0 or worst_reward < self.mean_ep_rews["reward"][-1] * 0.2:
            # initialize worst agent_meter
            self.agent_meters[worst_agent].initialize()
            return worst_agent
        else:
            return None
        
    def get_n_worst_agents(self, n):

        n_worst_agents = sorted(self.agent_meters.values(), key=lambda x: x.mean_ep_rews["reward"][-1] if len(x.mean_ep_rews["reward"]) > 0 else 1e6)[:n]

        worst_agent_indice = [agent_meter.name for agent_meter in n_worst_agents]
        worst_agent_rewards = [agent_meter.mean_ep_rews["reward"][-1] if len(agent_meter.mean_ep_rews["reward"]) > 0 else 1e6 for agent_meter in n_worst_agents]
        
        for agent_meter in n_worst_agents:
            agent_meter.initialize()

        return worst_agent_indice, worst_agent_rewards
        
    def get_n_best_agents(self, n):
            
            n_best_agents = sorted(self.agent_meters.values(), key=lambda x: x.mean_ep_rews["reward"][-1] if len(x.mean_ep_rews["reward"]) > 0 else -1e6, reverse=True)[:n]
    
            best_agent_indice = [agent_meter.name for agent_meter in n_best_agents]
            best_agent_rewards = [agent_meter.mean_ep_rews["reward"][-1] if len(agent_meter.mean_ep_rews["reward"]) > 0 else -1e6 for agent_meter in n_best_agents]
    
            for agent_meter in n_best_agents:
                agent_meter.initialize()
    
            return best_agent_indice, best_agent_rewards
        
    def get_shortest_agent(self):
        short_agent = None
        short_ep_len = None
        mean_ep_len = np.mean([
                        np.mean(self.agent_meters[agent].ep_len)
                        for agent in self.agents if len(self.agent_meters[agent].ep_len) > 0
                    ])
        
        print(f"mean_ep_len: {mean_ep_len}")

        i = self.update_agent_idx
        while i < len(self.agents):
            agent = self.agents[i]
            agent_meter = self.agent_meters[agent]
            if len(agent_meter.ep_len) == 0:
                i += 1
                if i == len(self.agents):
                    i = 0
                continue
            print(f"agent: {agent}, mean_ep_len: {np.mean(agent_meter.ep_len)}")
            if np.mean(agent_meter.ep_len) < mean_ep_len:
                short_agent = agent
                short_ep_len = np.mean(agent_meter.ep_len)
                i += 1
                break
            i += 1
            if i == len(self.agents):
                i = 0
        
        self.update_agent_idx = i if i < len(self.agents) else 0
        # initialize agent_meter
        self.agent_meters[short_agent].initialize()

        return short_agent, short_ep_len


class PerClassStatTracker:
    def __init__(self, buffer_size=64, min_count=8):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, labels, rewards):
        labels = np.array(labels)
        rewards = np.array(rewards)
        unique = np.unique(labels)
        advantages = np.empty_like(rewards)
        for label in unique:
            label_rewards = rewards[labels == label]
            if label not in self.stats:
                self.stats[label] = deque(maxlen=self.buffer_size)
            self.stats[label].extend(label_rewards)

            if len(self.stats[label]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[label])
                std = np.std(self.stats[label]) + 1e-6
            advantages[labels == label] = (label_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }