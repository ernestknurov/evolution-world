# Changelog

All notable changes to Evolution World will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-16 - MVP0 Release

### Added
- **Core Survival Simulation**: Grid-based 2D world where AI agents learn to survive by managing food, water, and energy resources
- **Reinforcement Learning Agent**: PPO-based agent using Stable Baselines3 that learns optimal survival strategies
- **Multi-resource Management**: Agents must balance food, water, energy, and lifespan to survive
- **Dynamic Environment**: 
  - Food and water resources that regenerate over time
  - Vision-based navigation (5x5 local view)
  - Realistic survival constraints with starvation and dehydration mechanics
- **Interactive Demo System**: 
  - Real-time Pygame visualization
  - Switch between human and AI control modes
  - Visual display of agent state (energy, food, water levels)
- **Training Infrastructure**:
  - Wandb integration for experiment tracking
  - Configurable environment and training parameters
  - Model saving and loading capabilities
- **Agent Actions**: Movement (UP, DOWN, LEFT, RIGHT), REST, EAT, DRINK
- **Survival Mechanics**:
  - Energy depletion leads to death
  - Starvation (food=0) increases energy drain
  - Dehydration (water=0) critically increases energy drain
  - Lifespan countdown system
- **Reward System**: 
  - Positive rewards for survival and resource consumption
  - Penalties for starvation, dehydration, and death
  - Illegal action penalties
- **Configuration System**: Dataclass-based configuration for easy parameter tuning
- **Logging**: Comprehensive logging with Loguru
- **Project Structure**: Clean separation of concerns with agents, environments, training, and utilities

### Technical Features
- **Environment**: Gymnasium-compatible interface
- **Multi-environment Training**: Vectorized environments for faster training
- **Observation Space**: Multi-input observation with vision, agent state, and action masks
- **Action Masking**: Legal action constraints based on environment state
- **Evaluation System**: Model evaluation with deterministic rollouts

### Documentation
- Comprehensive README with installation and usage instructions
- Project documentation with detailed game mechanics
- Development log tracking progress and experiments
- Code examples and quick start guide

### Performance
- Agents successfully learn to:
  - Prioritize water over food (dehydration is more critical)
  - Balance exploration vs exploitation for resource gathering
  - Manage energy efficiently through strategic resting
  - Survive entire lifespan under optimal conditions

### Project Stats
- **Training Time**: Successfully trained agents in ~500K timesteps
- **Survival Performance**: Trained agents achieve mean rewards of 23+ in evaluation
- **Environment Complexity**: 30x25 grid with dynamic resource spawning
- **Agent Capabilities**: 2-radius vision, 7 possible actions, multi-resource state management