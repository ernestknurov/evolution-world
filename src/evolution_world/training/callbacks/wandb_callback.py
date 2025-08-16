import wandb
import numpy as np # type: ignore
from stable_baselines3.common.callbacks import BaseCallback # type: ignore

class WandbCallback(BaseCallback):
    """
    Custom callback for logging metrics to Weights & Biases during training.
    """
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energy = []

        self.episode_count = 0
        self.step_rewards = []
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.current_episode_energy = []

        # self.num_food = []
        # self.num_water = []
        self.energy = []
        # once in episode metrics
        self.illegal_action_count = []
        self.lifespan = []
        
        # Loss tracking
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.explained_variances = []
    
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
        self.current_episode_rewards.append(step_reward)
        self.current_episode_steps += 1
        # Then in the logging section:
        wandb.log({ # type: ignore
            "training/mean_step_reward_1000": np.mean(self.step_rewards[-1000:]) if len(self.step_rewards) >= 1000 else np.mean(self.step_rewards)
        })

        episode_ended = False
        
        if infos:
            # Iwe have several envs running in parallel, so we need to aggregate their metrics
            # all_num_food = []
            # all_num_water = []
            all_energy = []
            all_illegal_actions_count = []
            all_lifespan = []
            
            # Process all environments
            for env_idx, info in enumerate(infos):
                if isinstance(info, dict):
                    # Collect metrics from this environment
                    # if 'num_food' in info:
                    #     all_num_food.append(float(info['num_food']))
                    # if 'num_water' in info:
                    #     all_num_water.append(float(info['num_water']))
                    if 'illegal_action_count' in info:
                        all_illegal_actions_count.append(int(info['illegal_action_count']))
                    if 'agent_state' in info: 
                        if 'energy_level' in info['agent_state']:
                            all_energy.append(float(info['agent_state']['energy_level']))
                        if 'lifespan' in info['agent_state']:
                            all_lifespan.append(int(info['agent_state']['lifespan']))
                        
                    # Check for episode completion
                    if 'episode' in info:
                        episode_info = info['episode']
                        episode_reward = float(episode_info['r'])
                        episode_length = int(episode_info['l'])
                        episode_ended = True
        
            # Now aggregate and log metrics from all environments
            env_metrics = {}
            
            # if all_num_food:
            #     self.num_food.extend(all_num_food)
            #     if len(self.num_food) > 100:
            #         env_metrics['env/mean_num_food_100'] = np.mean(self.num_food[-100:])
            # if all_num_water:
            #     self.num_water.extend(all_num_water)
            #     if len(self.num_water) > 100:
            #         env_metrics['env/mean_num_water_100'] = np.mean(self.num_water[-100:])
            if all_energy:
                self.current_episode_energy.extend(all_energy)
            
            # Log the aggregated metrics
            # if env_metrics:
            #     wandb.log(env_metrics) # type: ignore
        
        # Log episode completion
        if episode_ended:
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_energy.append(np.mean(self.current_episode_energy))
            self.illegal_action_count.append(np.mean(all_illegal_actions_count))
            self.lifespan.append(np.mean(all_lifespan))
            
            # Reset current episode tracking
            self.current_episode_rewards = []
            self.current_episode_steps = 0

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

            if len(self.lifespan) >= 100:
                episode_metrics["episode/mean_lifespan_100"] = np.mean(self.lifespan[-100:])
            
            # Log episode metrics to wandb
            wandb.log(episode_metrics) # type: ignore
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count}: Reward={episode_reward:.3f}, Length={episode_length}")
            
        return True