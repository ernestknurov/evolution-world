import numpy as np  # type: ignore
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore
from typing import Any, Dict, List, Sequence

from stable_baselines3.common.vec_env import VecEnv  # type: ignore

from src.evolution_world.envs.multi_agent import MultiAgentGridWorld

class MultiAgentVecEnv(VecEnv):
    """Option B implementation: treat each agent in a single MultiAgentGridWorld
    as one vector-environment slot (shared policy / shared weights).

    Notes:
    - A single underlying world is stepped once per "step_wait" with the joint
      actions collected from all alive agents.
    - Dead agents keep a zeroed observation (alive=0) and yield reward=0 after
      the death step; their slot is not marked done individually (credit assignment
      stays stable). The entire vector env resets only when ALL agents are dead
      (global episode end).
    - This avoids per-agent resets (which would require partial world resets / respawn logic).
    - PPO will see constant num_envs = initial agent count.
    - Add an "alive" scalar (0/1) to observation for masking if desired downstream.
    """

    def __init__(self, core: MultiAgentGridWorld):
        self.core = core
        self.agent_ids: List[int] = sorted(core.agents.keys())
        self.num_agents = len(self.agent_ids)

        # Single-agent observation & action spaces (augment with alive flag)
        single = core.observation_space
        self._single_obs_space = spaces.Dict({
            **single.spaces,
            "alive": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        })
        # IMPORTANT: observation_space must represent ONE env slot (one agent)
        observation_space = self._single_obs_space
        action_space = core.action_space  # per-agent action space

        super().__init__(self.num_agents, observation_space, action_space)

        # Buffers
        self._actions = None
        self._alive_mask = np.ones((self.num_agents,), dtype=np.int8)
        # Cache last observations per agent (for dead agents persistence)
        self._last_obs_per_agent: Dict[int, Dict[str, Any]] = {}

    # ---- VecEnv API ----
    def reset(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        obs_dict, _ = self.core.reset()
        self.agent_ids = sorted(self.core.agents.keys())
        self.num_agents = len(self.agent_ids)
        self._alive_mask = np.ones((self.num_agents,), dtype=np.int8)
        self._last_obs_per_agent = {}
        return self._build_stacked_obs(obs_dict)

    def step_async(self, actions: np.ndarray) -> None:  # type: ignore[override]
        self._actions = actions

    def step_wait(self):  # type: ignore[override]
        assert self._actions is not None, "step_async must be called before step_wait"
        # Build actions for alive agents only
        action_ids = {
            aid: int(self._actions[i])
            for i, aid in enumerate(self.agent_ids)
            if self._alive_mask[i] == 1
        }
        obs_dict, reward_dict, terminated_dict, _, info = self.core.step(action_ids)

        # Rewards buffer
        rewards = np.zeros((self.num_envs,), dtype=np.float32)

        # Update alive mask & rewards
        for i, aid in enumerate(self.agent_ids):
            if self._alive_mask[i] == 1:
                rewards[i] = reward_dict.get(aid, 0.0)
                if terminated_dict.get(aid, False):
                    self._alive_mask[i] = 0

        # Episode end when all agents dead
        all_dead = self._alive_mask.sum() == 0
        dones = np.array([all_dead] * self.num_envs, dtype=bool)

        # Build stacked observations (persist last for dead agents, zero vision/self/action_mask)
        stacked_obs = self._build_stacked_obs(obs_dict)

        # Per-slot infos (SB3 expects list length num_envs)
        infos: List[Dict[str, Any]] = []
        for i, aid in enumerate(self.agent_ids):
            slot_info: Dict[str, Any] = {
                "agent_id": aid,
                "alive": int(self._alive_mask[i]),
            }
            if i == 0:  # include global info once
                slot_info["global"] = info
            infos.append(slot_info)

        return stacked_obs, rewards, dones, infos

    def close(self) -> None:  # type: ignore[override]
        return

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None):  # type: ignore[override]
        if indices is None:
            indices = range(self.num_envs)
        # Single underlying core env; return same attribute for each requested index if exists
        if hasattr(self.core, attr_name):
            value = getattr(self.core, attr_name)
            return [value for _ in indices]
        raise AttributeError(f"Attribute {attr_name} not found in core environment")

    def set_attr(self, attr_name: str, value, indices: Sequence[int] | None = None):  # type: ignore[override]
        if indices is None:
            indices = range(self.num_envs)
        # Set on core once
        setattr(self.core, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: Sequence[int] | None = None, **method_kwargs):  # type: ignore[override]
        if indices is None:
            indices = range(self.num_envs)
        if hasattr(self.core, method_name):
            method = getattr(self.core, method_name)
            result = method(*method_args, **method_kwargs)
            # Return result replicated to match number of indices
            return [result for _ in indices]
        raise AttributeError(f"Method {method_name} not found in core environment")

    def env_is_wrapped(self, wrapper_class, indices: Sequence[int] | None = None):  # type: ignore[override]
        if indices is None:
            indices = range(self.num_envs)
        # No wrappers applied per-slot (single core world)
        return [False for _ in indices]

    def render(self, mode: str = "human"):
        # Optional: integrate pygame render if needed. For now, return None to satisfy interface.
        return None

    def seed(self, seed: int | None = None):  # type: ignore[override]
        # Delegate to core
        self.core.reset(seed=seed)

    # ---- Helpers ----
    def _build_stacked_obs(self, obs_dict: Dict[int, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        # Returns dict where each value has leading dimension = num_envs (agents)
        vision_list, self_list, mask_list, alive_list = [], [], [], []
        for i, aid in enumerate(self.agent_ids):
            if self._alive_mask[i] == 1:
                o = obs_dict[aid]
                self._last_obs_per_agent[aid] = o
            else:
                o = self._last_obs_per_agent.get(aid)
                if o is None:
                    o = {
                        "vision": np.zeros(self.core.observation_space["vision"].shape, dtype=np.int8),
                        "self": np.zeros(self.core.observation_space["self"].shape, dtype=np.float32),
                        "action_mask": np.zeros(self.core.observation_space["action_mask"].shape, dtype=np.int8),
                    }
                    self._last_obs_per_agent[aid] = o
            vision_list.append(o["vision"])
            self_list.append(o["self"])
            mask_list.append(o["action_mask"])
            alive_list.append(np.array([self._alive_mask[i]], dtype=np.int8))
        return {
            "vision": np.stack(vision_list, axis=0),
            "self": np.stack(self_list, axis=0),
            "action_mask": np.stack(mask_list, axis=0),
            "alive": np.stack(alive_list, axis=0),
        }
