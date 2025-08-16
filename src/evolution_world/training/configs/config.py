# src/evolution_world/envs/configs.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ScreenConfig:
    width: int = 1200
    height: int = 1000
    cell_size: int = 40
    title: str = "Evolution World"

@dataclass
class MapConfig:
    width: int = 30
    height: int = 25

@dataclass
class ResourceConfig:
    rate: float  # probability per tick of regeneration
    max_per_tile: int

@dataclass
class AgentState:
    food_level: float
    water_level: float
    energy_level: float
    lifespan: int
    position: tuple[int, int]

@dataclass
class AgentConfig:
    start_energy_level: float = 50
    min_energy_level: float = 0
    max_energy_level: float = 100

    energy_drain_default: float = 0.1  # Energy drain for living
    energy_drain_move: float = 1.0
    energy_drain_food_starvation: float = 3.0
    energy_drain_water_starvation: float = 10.0 # lack of water drains energy faster
    energy_boost_food: float = 10.0
    energy_boost_water: float = 0.2 # Water doesn't give you much energy, it just prevents starvation
    energy_boost_rest: float = 0.5

    start_water_level: float = 5
    min_water_level: float = 0
    max_water_level: float = 10

    water_drain_default: float = 0.1
    water_boost_drink: float = 1.0

    start_food_level: float = 5
    min_food_level: float = 0
    max_food_level: float = 10

    food_drain_default: float = 0.05 # Food drain for living
    food_boost_eat: float = 1.0 # Food boost for eating

    lifespan: int = 500  # ticks
    start_location: tuple[int, int] = (5, 5)  # Starting position in the grid

    vision_radius: int = 2

@dataclass
class RewardConfig:
    alive_tick: float = 0.1
    consume_food: float = 0.8
    consume_water: float = 1.0
    food_starvation_penalty: float = -2.0
    water_starvation_penalty: float = -4.0
    illegal_action_penalty: float = -0.1
    death_penalty: float = -20.0

@dataclass
class EnvConfig:
    map: MapConfig = field(default_factory=MapConfig)
    resources: Dict[int, ResourceConfig] = field(
        default_factory=lambda: {
            1: ResourceConfig(rate=0.025, max_per_tile=1),  # FOOD
            2: ResourceConfig(rate=0.05, max_per_tile=1),  # WATER
        }
    )
    agent: AgentConfig = field(default_factory=AgentConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

@dataclass
class TrainingConfig:
    total_timesteps: int = 500_000
    num_envs: int = 4
    net_arch: Dict[str, list[int]] = field(default_factory=lambda: {
        "pi": [64, 64],
        "vf": [64, 64]
    })
    from_loaded_model: bool = False
    load_path: str = "models/trained_model"
    save_path: str = "models/trained_model.zip"

@dataclass
class WandbConfig:
    project_name: str = "evolution_world"
    run_name: str = "pilot_run10"