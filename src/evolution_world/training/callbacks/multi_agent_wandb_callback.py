import wandb
import numpy as np # type: ignore
from stable_baselines3.common.callbacks import BaseCallback # type: ignore

class MultiAgentWandbCallback(BaseCallback):
    """
    Custom callback for logging metrics to Weights & Biases during training.
    """
    def __init__(self, num_envs: int, verbose=0):
        super(MultiAgentWandbCallback, self).__init__(verbose)
        self.num_envs = num_envs
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energy = []

        self.step_rewards = []
        self.current_episode_energy = []

        # Episode statistics across all environments
        self.all_episode_ended = [False] * self.num_envs
        self.all_episode_rewards = [0.0] * self.num_envs
        self.all_episode_lengths = [0] * self.num_envs

        # Once in episode metrics. Collected when episode ends to see what is the final value
        self.illegal_action_count = []
        self.collision_count = []
        
        # Loss tracking
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.explained_variances = []

        self.printed = False
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout to log training losses."""
        # Access the model's logger to get loss information
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if hasattr(self.model.logger, 'name_to_value'):
                logger_dict = self.model.logger.name_to_value
                
                loss_metrics = {}
                current_step = self.num_timesteps
                
                # Extract loss values if available
                if 'train/policy_gradient_loss' in logger_dict:
                    policy_loss = float(logger_dict['train/policy_gradient_loss'])
                    self.policy_losses.append(policy_loss)
                    # loss_metrics['loss/policy_loss'] = policy_loss
                
                if 'train/value_loss' in logger_dict:
                    value_loss = float(logger_dict['train/value_loss'])
                    self.value_losses.append(value_loss)
                    # loss_metrics['loss/value_loss'] = value_loss
                
                if 'train/entropy_loss' in logger_dict:
                    entropy_loss = float(logger_dict['train/entropy_loss'])
                    self.entropy_losses.append(entropy_loss)
                    loss_metrics['loss/entropy_loss'] = entropy_loss
                
                if 'train/loss' in logger_dict:
                    total_loss = float(logger_dict['train/loss'])
                    self.total_losses.append(total_loss)
                    # loss_metrics['loss/total_loss'] = total_loss
                
                if 'train/explained_variance' in logger_dict:
                    explained_var = float(logger_dict['train/explained_variance'])
                    self.explained_variances.append(explained_var)
                    # loss_metrics['loss/explained_variance'] = explained_var
                
                # Add rolling averages for smoother visualization
                if len(self.policy_losses) >= 10:
                    loss_metrics['loss/policy_loss_ma10'] = np.mean(self.policy_losses[-10:])
                if len(self.value_losses) >= 10:
                    loss_metrics['loss/value_loss_ma10'] = np.mean(self.value_losses[-10:])
                if len(self.total_losses) >= 10:
                    loss_metrics['loss/total_loss_ma10'] = np.mean(self.total_losses[-10:])
                if len(self.explained_variances) >= 10:
                    loss_metrics['loss/explained_variance_ma10'] = np.mean(self.explained_variances[-10:])
                
                # Log to wandb if we have any loss metrics
                if loss_metrics:
                    wandb.log(loss_metrics) # type: ignore
                    
                if self.verbose > 1 and loss_metrics:
                    print(f"Step {current_step} - Losses: {loss_metrics}")

    def _on_step(self) -> bool:
        # Track step-level rewards for all environments
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        
        # Aggregating across all environments
        if isinstance(rewards, (list, np.ndarray)):
            # Log mean reward across all environments
            step_reward = np.mean(rewards)
        else:
            step_reward = float(rewards)
            
        self.step_rewards.append(step_reward)
        # Then in the logging section:
        wandb.log({ # type: ignore
            "training/mean_step_reward_1000": np.mean(self.step_rewards[-1000:]) if len(self.step_rewards) >= 1000 else np.mean(self.step_rewards)
        })

        # if not self.printed:
        #     print(infos)
        #     self.printed = True
        if infos:
            # We have several envs running in parallel, so we need to aggregate their metrics
            # Get global info once and check if it exists
            global_info = infos[0].get('global', {}) if infos else {}
            agent_states = global_info.get('agent_states', {})
            
            if agent_states:
                # Use list comprehension with fewer dictionary lookups
                alive_energies = [
                    agent_states[info['agent_id']]['energy_level']
                    for info in infos 
                    if info.get('agent_id') in agent_states and info.get('alive', 0) == 1
                ]
                
                if alive_energies:
                    self.current_episode_energy.append(np.mean(alive_energies))
            
            current_collision_count = int(global_info.get('collision_count', 0))
            current_illegal_actions_count = int(global_info.get('illegal_action_count', 0))

            # Process all environments
            for env_idx, info in enumerate(infos):
                # Check for episode completion
                if 'episode' in info:
                    episode_info = info['episode']
                    self.all_episode_rewards[env_idx] = float(episode_info['r'])
                    self.all_episode_lengths[env_idx] = int(episode_info['l'])
                    self.all_episode_ended[env_idx] = True
                    # print(self.all_episode_ended)
        
        # Log episode completion
        if all(self.all_episode_ended):
            self.episode_rewards.append(np.mean(self.all_episode_rewards))
            self.episode_lengths.append(np.mean(self.all_episode_lengths))
            self.illegal_action_count.append(current_illegal_actions_count)
            self.collision_count.append(current_collision_count)
            self.episode_energy.append(np.mean(self.current_episode_energy))
            
            # Reset current episode tracking
            self.all_episode_ended = [False] * self.num_envs
            self.all_episode_rewards = [0.0] * self.num_envs
            self.all_episode_lengths = [0] * self.num_envs

            episode_metrics = {}
            
            # Calculate rolling averages
            if len(self.episode_rewards) >= 100:
                episode_metrics["episode/mean_reward_100"] = np.mean(self.episode_rewards[-100:])
            
            if len(self.episode_lengths) >= 100:
                episode_metrics["episode/mean_length_100"] = np.mean(self.episode_lengths[-100:])

            if len(self.episode_energy) >= 100:
                episode_metrics["episode/mean_energy_100"] = np.mean(self.episode_energy[-100:])

            if len(self.illegal_action_count) >= 100:
                episode_metrics["episode/mean_illegal_actions_100"] = np.mean(self.illegal_action_count[-100:])

            if len(self.collision_count) >= 100:
                episode_metrics["episode/mean_collisions_100"] = np.mean(self.collision_count[-100:])

            # Log episode metrics to wandb
            wandb.log(episode_metrics) # type: ignore
            
        return True