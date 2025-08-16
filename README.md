# Evolution World

A reinforcement learning survival simulation where agents learn to manage food, water, and energy resources in a grid world environment.

## 🎯 Overview

Evolution World is a survival simulation game where AI agents must learn to balance critical resources to stay alive as long as possible. Agents navigate a grid world, searching for food and water while managing their energy levels through strategic decision-making.

![alt text](assets/images/mvp0_demo.png)

## 🌟 Features

### Agent Survival Mechanics
- **Multi-resource management**: Agents must balance food, water, energy, and lifespan
- **Complex state interactions**: Starvation and dehydration affect energy levels
- **Strategic decision-making**: Choose between movement, resting, eating, and drinking
- **Vision-based navigation**: Agents observe their local environment to make decisions

### Environment
- **Grid-based world**: 2D grid with walls, food, water, and empty spaces
- **Dynamic resource spawning**: Food and water regenerate over time
- **Realistic survival constraints**: Energy depletion leads to death
- **Pygame visualization**: Real-time rendering of the simulation

### AI Training
- **Reinforcement Learning**: Uses Stable Baselines3 for training intelligent agents
- **Gymnasium integration**: Standard RL environment interface
- **Wandb tracking**: Comprehensive experiment tracking and visualization
- **Configurable parameters**: Easily adjust game mechanics and training settings

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- uv (recommended) or pip for package management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ernestknurov/evolution-world.git
cd evolution-world
```

2. Install dependencies:
```bash
uv sync
```

### Running the Simulation

**Train a new agent:**
```bash
python scripts/train/train_mvp0.py
```

**Play with a trained agent:**
```bash
python scripts/demo/play_simulation.py
```


## 🎮 Game Mechanics

### Agent State
- **Energy (0-100)**: Required for all actions, depletes over time
- **Food Level (0-10)**: Prevents starvation, affects energy regeneration
- **Water Level (0-10)**: Prevents dehydration, critical for survival
- **Lifespan**: Fixed maximum lifetime, decreases each turn

### Actions
- **Movement**: UP, DOWN, LEFT, RIGHT (costs energy)
- **REST**: Stay in place, recover small amount of energy
- **EAT**: Consume food at current location (if available)
- **DRINK**: Consume water at current location (if available)

### Survival Rules
- **Death conditions**: Energy ≤ 0 or lifespan ≤ 0
- **Starvation**: When food level = 0, energy drains faster
- **Dehydration**: When water level = 0, energy drains even faster (more critical)
- **Resource regeneration**: Food and water spawn randomly over time

## 📁 Project Structure

```
evolution-world/
├── src/evolution_world/          # Core package
│   ├── agents/                   # Agent implementations
│   │   ├── rl_agent.py          # Reinforcement learning agent
│   │   └── random_agent.py      # Random baseline agent
│   ├── envs/                    # Environment implementations
│   │   └── grid_world.py        # Main grid world environment
│   ├── training/                # Training utilities
│   │   ├── callbacks/           # Training callbacks
│   │   └── configs/             # Configuration classes
│   └── utils/                   # Utility functions
├── scripts/                     # Executable scripts
│   ├── train/                   # Training scripts
│   └── demo/                    # Demo and visualization scripts
├── notebooks/                   # Jupyter notebooks for exploration
├── models/                      # Saved model files
└── docs/                        # Documentation
```

## 🔧 Configuration

The project uses dataclass-based configuration for easy customization:

- **Environment settings**: Grid size, resource spawn rates, agent starting state
- **Training parameters**: Learning rate, batch size, training steps
- **Reward structure**: Penalties and rewards for different actions
- **Agent capabilities**: Vision radius, energy costs, resource gains

## 📊 Monitoring & Visualization

- **Wandb integration**: Automatic experiment tracking and metric visualization
- **Real-time rendering**: Watch agents learn in real-time with Pygame
- **Training metrics**: Episode rewards, survival time, resource efficiency
- **Agent behavior analysis**: Track decision patterns and survival strategies

## 🛠️ Development

**Install development dependencies:**
```bash
uv sync --group dev
```

**Project uses:**
- **Gymnasium**: Standard RL environment interface
- **Stable Baselines3**: State-of-the-art RL algorithms
- **Pygame**: Real-time visualization and rendering
- **Wandb**: Experiment tracking and visualization
- **NumPy**: Efficient numerical computations
- **Loguru**: Comprehensive logging

## 📈 Performance

Agents typically learn to:
1. Prioritize water over food (dehydration is more critical)
2. Balance exploration vs exploitation for resource gathering
3. Manage energy efficiently through strategic resting
4. Develop spatial memory for resource locations



## 📝 Documentation

- `docs/project_docs.md` - Detailed game mechanics and agent behavior
