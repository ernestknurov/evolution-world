import numpy as np # type: ignore

from stable_baselines3 import PPO # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy # type: ignore
from stable_baselines3.common.vec_env import VecMonitor # type: ignore
 
from src.evolution_world.utils import LogFactory
from src.evolution_world.envs.grid_world import GridWorld
from src.evolution_world.envs.multi_agent import MultiAgentGridWorld
from src.evolution_world.envs.wrappers.multi_agent_vec_env import MultiAgentVecEnv
from src.evolution_world.training.configs.config import EnvConfig, TrainingConfig

logger = LogFactory.get_logger(__name__)

class RLAgent:
    def __init__(self, model_path: str | None = None):
        if isinstance(model_path, str):
            logger.info(f"Loading model from {model_path}")
            self.load_model(model_path)
        else:
            self.model = None

    def train(self, cfg: TrainingConfig, env_cfg: EnvConfig, callback=None):
        logger.info(f"Training model for {cfg.total_timesteps} timesteps on environment MultiAgentGridWorld")
        # Option B: single core world, agents become vectorized slots
        core = MultiAgentGridWorld(cfg=env_cfg, seed=None)
        base_vec = MultiAgentVecEnv(core)
        self.vec_env = VecMonitor(base_vec, filename=None)
        
        policy_kwargs = {
            "net_arch": cfg.net_arch,
        }
        if not self.model:
            logger.info("Initializing new RL model")
            self.model = PPO(
                "MultiInputPolicy",
                self.vec_env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                    # === rollout / batch settings ===
                n_steps=512,          # 512×num_agents samples per update
                batch_size=256,
                n_epochs=10,

                # === optimization ===
                learning_rate=3e-4,
                clip_range=0.2,
                gamma=0.99,
                gae_lambda=0.95,

                # === losses / regularization ===
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5                
            )
        else:
            logger.info("Using loaded model, setting environment for continued training")
            self.model.set_env(self.vec_env)

        # Debug: Print model architecture
        logger.debug("Model architecture:")
        logger.debug(self.model.policy)
        
        # Debug: Count parameters
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        logger.debug(f"Total policy parameters: {total_params:,}")

        self.model.learn(total_timesteps=cfg.total_timesteps, progress_bar=True, callback=callback)

    def load_model(self, model_path: str, **kwargs):
        self.model = PPO.load(model_path, policy="MultiInputPolicy", **kwargs)

    def save_model(self, model_path: str, **kwargs):
        logger.info(f"Saving model to {model_path}")
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        self.model.save(model_path, **kwargs)

    def evaluate(self, num_episodes: int=100, **kwargs) -> tuple:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if self.vec_env is None:
            raise ValueError("Vectorized environment is not initialized.")
        logger.debug(f"Evaluating model on {num_episodes} episodes")
        mean_reward, std_reward = evaluate_policy(self.model, self.vec_env, n_eval_episodes=num_episodes, **kwargs)
        logger.debug(f"Mean reward: {mean_reward} +/- {std_reward}")
        return mean_reward, std_reward

    def predict(self, state: dict, **kwargs) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        
        action, _ = self.model.predict(state, **kwargs)
        return action
    
    def act(self, observation: dict, **kwargs) -> int:
        return int(self.predict(observation, **kwargs))