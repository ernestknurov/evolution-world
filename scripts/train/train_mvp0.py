import wandb
from dataclasses import asdict

from src.evolution_world.utils import LogFactory
from src.evolution_world.agents.rl_agent import RLAgent
from src.evolution_world.training.callbacks import WandbCallback
from src.evolution_world.training.configs.config import TrainingConfig, WandbConfig

logger = LogFactory.get_logger(__name__)

def main():
    logger.info("Starting Evolution World training script...")
    
    rl_agent = RLAgent()
    training_config = TrainingConfig()
    logger.info(f"Training configuration: {training_config}")
    wandb_config = WandbConfig()
    logger.info(f"Wandb configuration: {wandb_config}")

    wandb.init( # type: ignore
            project=wandb_config.project_name,
            name=wandb_config.run_name,
            config=asdict(training_config),
            tags=["rl", "evolution", "training"]
        )
    
    if training_config.from_loaded_model:
        logger.info(f"Loading model from {training_config.load_path}")
        rl_agent.load_model(training_config.load_path)
    else:
        logger.info("No pre-trained model specified, starting fresh training.")
        
    logger.info("Starting training...")
    rl_agent.train(cfg=training_config, callback=WandbCallback())

    rl_agent.save_model(training_config.save_path)
    logger.info(f"Model saved to {training_config.save_path}")

    logger.info("Evaluating the trained model...")
    rl_agent.evaluate(1, deterministic=True, render=False)

    logger.info("Training and evaluation completed successfully.")

if __name__ == "__main__":
    main()