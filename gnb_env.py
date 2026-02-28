"""
Improved GNBGymEnv:
- Path-loss + log-normal shadowing SINR model (simple)
- Throughput estimation via Shannon approximation
- Configurable reward weights: throughput, fairness (Jain), movement energy cost
- History stacking of observations for temporal context
- Action supports discrete movement (DQN) or continuous vector (PPO)
"""
import math
import random
from typing import Tuple, Dict, Any, Optional

import gym
import numpy as np
from gym import spaces

def path_loss(d, freq_ghz=3.5):
    # Free-space path loss in dB (approx)
    if d <= 1e-3:
        d = 1e-3
    return 20 * math.log10(d) + 20 * math.log10(freq_ghz) + 32.44

def shannon_capacity(bw_hz, sinr_linear):
    # capacity in bits/s
    return bw_hz * math.log2(1.0 + max(0.0, sinr_linear))

def jain_fairness(x):
    x = np.array(x, dtype=np.float32)
    if np.sum(x) == 0:
        return 0.0
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2) + 1e-9)

class GNBGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
        grid_size: Tuple[int, int] = (50, 50),
        n_ues: int = 8,
        max_steps: int = 200,
        obstacle_prob: float = 0.03,
        los_distance_threshold: float = 10.0,
        movement_cost: float = 0.01,
        bw_hz: float = 10e6,  # 10 MHz
        tx_power_dbm: float = 30.0,
        noise_figure_db: float = 7.0,
        history_len: int = 4,
        continuous_action: bool = False,
        ue_speed: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.grid_w, self.grid_h = grid_size
        self.n_ues = n_ues
        self.max_steps = max_steps
        self.obstacle_prob = obstacle_prob
        self.los_distance_threshold = los_distance_threshold
        self.movement_cost = movement_cost
        self.bw_hz = bw_hz
        self.tx_power_dbm = tx_power_dbm
        self.noise_figure_db = noise_figure_db
        self.history_len = history_len
        self.continuous_action = continuous_action
        self.ue_speed = ue_speed

        # Actions:
        # - discrete: 0 stay, 1 up, 2 down, 3 left, 4 right
        # - continuous: 2D vector in [-1,1] scaled to step size
        if self.continuous_action:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)

        # Base observation: gNB(x,y), UEs(x,y), last step's per-UE throughput
        base_dim = 2 + 2 * self.n_ues + self.n_ues
        # history stacking along first dim -> final shape base_dim * history_len
        obs_low = np.full((base_dim * self.history_len,), 0.0, dtype=np.float32)
        obs_high = np.full((base_dim * self.history_len,), max(self.grid_w, self.grid_h) * 1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.seed(seed)
        self._rng = np.random.RandomState(self._seed)
        self.reset()

    def seed(self, seed=None):
        self._seed = seed if seed is not None else np.random.randint(0, 2 ** 31 - 1)
        random.seed(self._seed)
        np.random.seed(self._seed)

    def reset(self):
        self.step_count = 0
        self.gnb_pos = np.array([self.grid_w // 2, self.grid_h // 2], dtype=np.float32)
        # random UEs
        self.ues = self._rng.uniform(low=0, high=[self.grid_w - 1, self.grid_h - 1], size=(self.n_ues, 2)).astype(np.float32)
        # obstacles grid (not used in pathloss here, reserved for future)
        self.obstacles = self._rng.rand(self.grid_w, self.grid_h) < self.obstacle_prob
        self.last_throughputs = np.zeros(self.n_ues, dtype=np.float32)
        # history buffer for state stacking
        self._history = [self._base_obs()] * self.history_len
        return self._get_obs()

    def _base_obs(self):
        # gNB xy, UEs flattened xy, last throughputs
        return np.concatenate(([self.gnb_pos[0], self.gnb_pos[1]], self.ues.flatten(), self.last_throughputs)).astype(np.float32)

    def _get_obs(self):
        stacked = np.concatenate(self._history[-self.history_len :])
        return stacked.astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} invalid"
        old_pos = self.gnb_pos.copy()
        if self.continuous_action:
            # action in [-1,1]^2 -> scale by 1 cell per step
            move = np.clip(action, -1.0, 1.0)
            self.gnb_pos += move  # can be fractional
        else:
            if action == 1:
                self.gnb_pos[1] = min(self.grid_h - 1, self.gnb_pos[1] + 1)
            elif action == 2:
                self.gnb_pos[1] = max(0, self.gnb_pos[1] - 1)
            elif action == 3:
                self.gnb_pos[0] = max(0, self.gnb_pos[0] - 1)
            elif action == 4:
                self.gnb_pos[0] = min(self.grid_w - 1, self.gnb_pos[0] + 1)
            # 0 => stay

        # move UEs with simple random direction and speed
        angles = self._rng.uniform(0, 2 * math.pi, size=(self.n_ues,))
        deltas = np.stack([np.cos(angles), np.sin(angles)], axis=1) * self.ue_speed
        self.ues += deltas
        self.ues[:, 0] = np.clip(self.ues[:, 0], 0, self.grid_w - 1)
        self.ues[:, 1] = np.clip(self.ues[:, 1], 0, self.grid_h - 1)

        # compute SINR and throughput per UE
        throughputs = []
        sinrs = []
        for ue in self.ues:
            d = np.linalg.norm(ue - self.gnb_pos)
            pl_db = path_loss(max(0.1, d))
            shadow_db = self._rng.normal(0.0, 3.0)  # 3 dB shadow std
            rx_dbm = self.tx_power_dbm - pl_db + shadow_db
            noise_dbm = -174.0 + 10.0 * math.log10(self.bw_hz) + self.noise_figure_db
            sinr_db = rx_dbm - noise_dbm
            sinr_linear = 10 ** (sinr_db / 10.0)
            cap = shannon_capacity(self.bw_hz, sinr_linear)
            throughputs.append(cap)
            sinrs.append(sinr_linear)
        self.last_throughputs = np.array(throughputs, dtype=np.float32)

        # reward composition (weights can be tuned or exposed)
        # throughput_sum (bits/s), fairness (Jain), movement cost (distance)
        throughput_sum = float(np.sum(self.last_throughputs))
        fairness = jain_fairness(self.last_throughputs)
        movement_energy = self.movement_cost * float(np.linalg.norm(self.gnb_pos - old_pos))
        # normalize throughput to a reasonable scale (e.g., Mbps)
        throughput_mbps = throughput_sum / 1e6

        # weights (could be parameters)
        alpha = 1.0  # throughput weight
        beta = 50.0  # fairness weight
        gamma = 1.0  # movement cost multiplier
        reward = alpha * throughput_mbps + beta * fairness - gamma * movement_energy

        self.step_count += 1
        done = self.step_count >= self.max_steps

        # update history
        self._history.append(self._base_obs())
        if len(self._history) > self.history_len:
            self._history = self._history[-self.history_len :]

        info = {
            "throughputs": self.last_throughputs.copy(),
            "throughput_sum": throughput_sum,
            "throughput_mbps": throughput_mbps,
            "fairness": fairness,
            "movement_energy": movement_energy,
            "gnb_pos": self.gnb_pos.copy(),
        }
        return self._get_obs(), float(reward), bool(done), info

    def render(self, mode="human"):
        # Optional simple render; prefer visualization via callbacks
        print(f"Step {self.step_count} GNB {self.gnb_pos} Throughput sum={np.sum(self.last_throughputs)/1e6:.2f} Mbps Fairness={jain_fairness(self.last_throughputs):.3f}")

    def close(self):
        pass
