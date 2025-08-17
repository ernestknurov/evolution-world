import pygame # type: ignore
from pygame.locals import Color # type: ignore
import numpy as np # type: ignore
import gymnasium as gym # type: ignore
from dataclasses import asdict
from typing import Tuple, Dict, Any
from pathlib import Path

from src.evolution_world.training.configs.enums import Entity, Action
from src.evolution_world.training.configs.config import EnvConfig, ScreenConfig, AssetPaths, AgentState


class GridWorld(gym.Env):
    def __init__(self, cfg: EnvConfig, render_mode: str | None = None, seed: int | None = None):
        self.cfg = cfg
        self.seed = seed
        self.render_mode = render_mode
        self.grid_size = (cfg.map.height, cfg.map.width)
        self.padding = cfg.agent.vision_radius
        self.terrain_size = (self.grid_size[0] + self.padding * 2, 
                             self.grid_size[1] + self.padding * 2)
        self.n_actions = 7
        self.n_entities = 4  # Empty, FOOD, WATER, WALL
        self.action_space = gym.spaces.Discrete(self.n_actions)  # up, down, left, right, rest
        self.observation_space = gym.spaces.Dict({
            'vision': gym.spaces.Box(low=0, high=1, shape=(self.cfg.agent.vision_radius * 2 + 1, 
                                                           self.cfg.agent.vision_radius * 2 + 1, 
                                                           self.n_entities + 1), dtype=np.int8),
            'self': gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),  # energy, lifespan
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=np.int8),
        })
        self.illegal_action_count = 0
        self.action_to_vector = {
            Action.UP: np.array([-1, 0]),    # Up
            Action.DOWN: np.array([1, 0]),   # Down
            Action.LEFT: np.array([0, -1]),  # Left
            Action.RIGHT: np.array([0, 1]),  # Right
            Action.REST: np.array([0, 0]),   # Rest
            Action.EAT: np.array([0, 0]),    # Eat (no movement)
            Action.DRINK: np.array([0, 0])   # Drink (no movement)
        }
        self.reset()

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Note: agents and some objects (food, water) on terrain can overlay. Agents with agent cannot overlay. Objects with objects cannot overlay.
        # 1. Terrain
        # 1.1 walls
        self.terrain = np.full(self.terrain_size, Entity.WALL.value, dtype=np.int32)  # Initialize with WALL

        # 1.2 empty tiles
        self.terrain[
            self.padding: self.terrain_size[0] - self.padding,
            self.padding: self.terrain_size[1] - self.padding] = Entity.EMPTY.value  # keeping only padding as walls
        # 1.3 food
        food_count = int(self.grid_size[0] * self.grid_size[1] * self.cfg.resources[1].max_per_tile * self.cfg.resources[1].rate)
        food_positions = np.random.choice(np.flatnonzero(self.terrain == Entity.EMPTY.value), size=food_count, replace=False)
        self.terrain.flat[food_positions] = Entity.FOOD.value
        # 1.4 water
        water_count = int(self.grid_size[0] * self.grid_size[1] * self.cfg.resources[2].max_per_tile * self.cfg.resources[2].rate)
        water_positions = np.random.choice(np.flatnonzero(self.terrain == Entity.EMPTY.value), size=water_count, replace=False)
        self.terrain.flat[water_positions] = Entity.WATER.value

        # 2. Agents
        self.agent = AgentState(
            energy_level=self.cfg.agent.start_energy_level,
            water_level=self.cfg.agent.start_water_level,
            food_level=self.cfg.agent.start_food_level,
            lifespan=self.cfg.agent.lifespan,
            position=self.cfg.agent.start_location
        )
        self.agent_grid = np.zeros_like(self.terrain)
        self.agent_grid[self.agent.position] = 1

        self.illegal_action_count = 0

        # 2. Collecting information
        info = {
            'terrain': self.terrain,
            'agent_grid': self.agent_grid,
            'agent_state': asdict(self.agent),
        }
        return self.get_observation(), info

    def get_observation(self) -> Dict[str, Any]:
        # 1. Vision
        # The vision is a kxk egocentric window of tiles around the agent,
        x, y = self.agent.position
        terrain_vision = self.terrain[
            x - self.cfg.agent.vision_radius:x + self.cfg.agent.vision_radius + 1,
            y - self.cfg.agent.vision_radius:y + self.cfg.agent.vision_radius + 1
        ]
        agent_vision = self.agent_grid[
            x - self.cfg.agent.vision_radius:x + self.cfg.agent.vision_radius + 1,
            y - self.cfg.agent.vision_radius:y + self.cfg.agent.vision_radius + 1
        ]
        stratified_terrain_vision = np.eye(self.n_entities, dtype=np.int8)[terrain_vision]
        stratified_agent_vision = agent_vision[..., np.newaxis].astype(np.int8)  # Add a new axis for stacking
        vision = np.concatenate((stratified_terrain_vision, stratified_agent_vision), axis=-1)  # Concatenate terrain and agent vision

        # 2. Agent state
        # The agent state is a vector of its resources (energy, food, water, etc.)
        # Normalized to [0, 1] range.
        # For example, if the agent has 50 energy out of 100 max energy,
        # it will be 0.5.
        norm_agent_energy_level = self.agent.energy_level / self.cfg.agent.max_energy_level
        norm_agent_food_level = self.agent.food_level / self.cfg.agent.max_food_level
        norm_agent_water_level = self.agent.water_level / self.cfg.agent.max_water_level
        norm_agent_lifespan = self.agent.lifespan / self.cfg.agent.lifespan
        agent_state = np.array([norm_agent_energy_level, norm_agent_food_level, 
                                norm_agent_water_level, norm_agent_lifespan], dtype=np.float32)

        # 3. Legal actions
        # The legal actions are determined by the agent's position and the terrain.
        # For example, if the agent is at the edge of the grid, it cannot move
        # outside the grid, and if there is food or water in the adjacent tiles, it
        # can eat or drink.
        # The action mask is a binary vector of length n_actions, where 1 means the
        # action is legal and 0 means the action is illegal.
        legal_actions = self.get_legal_actions()
        legal_action_mask = np.array([legal_actions[Action(i)] for i in range(self.n_actions)], dtype=np.int8)

        return {
            'vision': vision,
            'self': agent_state,
            'action_mask': legal_action_mask,
        }

    def get_legal_actions(self) -> Dict[Action, bool]:
        legal_actions = {
            Action.UP: self.agent.position[0] > self.padding,  # Up
            Action.DOWN: self.agent.position[0] < self.grid_size[0] - 1 + self.padding,  # Down
            Action.LEFT: self.agent.position[1] > self.padding,  # Left
            Action.RIGHT: self.agent.position[1] < self.grid_size[1] - 1 + self.padding,  # Right
            Action.EAT: self.terrain[self.agent.position] == 2,  # Can eat if there's food
            Action.DRINK: self.terrain[self.agent.position] == 3,  # Can drink if there's water
            Action.REST: True,  # Rest action is always legal
        }
        return legal_actions
    
    def move_agent(self, action: Action):
        # x, y from array, so x is from top to bottom, y - from left to right
        # x - horizontal (from left to right), y - vertical (from bottom to top)
        self.agent_grid[self.agent.position] = 0  
        dx, dy = self.action_to_vector[action]
        self.agent.position = self.agent.position[0] + dx, self.agent.position[1] + dy
        self.agent_grid[self.agent.position] = 1  

    def clip(self, value: float, min_value: float, max_value: float) -> float:
        """Clip the value to the range [min_value, max_value]."""
        return max(min(value, max_value), min_value)
    
    def step(self, action_id: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0
        info = {}

        action = Action(action_id)
        legal_actions = self.get_legal_actions()

        if not legal_actions[action]:
            reward += self.cfg.rewards.illegal_action_penalty
            self.illegal_action_count += 1
        else:
            # State update
            if action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                self.move_agent(action)
                self.agent.energy_level -= self.cfg.agent.energy_drain_move
            elif action == Action.REST:
                self.agent.energy_level +=  self.cfg.agent.energy_boost_rest
            elif action == Action.EAT:
                self.agent.energy_level += self.cfg.agent.energy_boost_food
                self.agent.food_level += self.cfg.agent.food_boost_eat
                reward += self.cfg.rewards.consume_food
                self.terrain[self.agent.position] = Entity.EMPTY.value
            elif action == Action.DRINK:
                self.agent.energy_level += self.cfg.agent.energy_boost_water
                self.agent.water_level += self.cfg.agent.water_boost_drink
                reward += self.cfg.rewards.consume_water
                self.terrain[self.agent.position] = Entity.EMPTY.value


        # Default processes
        self.agent.lifespan -= 1
        self.agent.energy_level -= self.cfg.agent.energy_drain_default
        self.agent.food_level -= self.cfg.agent.food_drain_default
        self.agent.water_level -= self.cfg.agent.water_drain_default

        self.agent.energy_level = self.clip(self.agent.energy_level, self.cfg.agent.min_energy_level, self.cfg.agent.max_energy_level)
        self.agent.water_level = self.clip(self.agent.water_level, self.cfg.agent.min_water_level, self.cfg.agent.max_water_level)
        self.agent.food_level = self.clip(self.agent.food_level, self.cfg.agent.min_food_level, self.cfg.agent.max_food_level)

        if self.agent.food_level <= 0:
            self.agent.energy_level -= self.cfg.agent.energy_drain_food_starvation
            reward += self.cfg.rewards.food_starvation_penalty
        if self.agent.water_level <= 0:
            self.agent.energy_level -= self.cfg.agent.energy_drain_water_starvation
            reward += self.cfg.rewards.water_starvation_penalty

        # Resources regeneration
        resource_id_to_entity_id = {1: 2, 2: 3}  # FOOD -> 2, WATER -> 3
        for resource_id, resource_config in self.cfg.resources.items():
            if np.random.rand() < resource_config.rate:
                empty_tiles = np.flatnonzero(self.terrain == Entity.EMPTY.value)
                if empty_tiles.size > 0:
                    random_tile = np.random.choice(empty_tiles)
                    self.terrain.flat[random_tile] = resource_id_to_entity_id[resource_id]
            
        # 3. Termination conditions
        if self.agent.energy_level <= 0:
            reward += self.cfg.rewards.death_penalty
            terminated = True
        elif self.agent.lifespan <= 0:
            terminated = True
        else:
            reward += self.cfg.rewards.alive_tick

        # 4. collecting useful information
        info = {
            'terrain': self.terrain,
            'agent_grid': self.agent_grid,
            'agent_state': asdict(self.agent),
            'num_food': np.sum(self.terrain == Entity.FOOD.value) / self.grid_size[0] / self.grid_size[1],
            'num_water': np.sum(self.terrain == Entity.WATER.value) / self.grid_size[0] / self.grid_size[1],
            'illegal_action_count': self.illegal_action_count,
        }
        
        return self.get_observation(), reward, terminated, truncated, info
    
    def render(self, screen, screen_config: ScreenConfig=ScreenConfig(), asset_paths: AssetPaths=AssetPaths()):
        cell_size = screen_config.cell_size
        if self.render_mode == 'human':
            # 1. Render the grid
            screen.fill(Color("gray"))
            # 1.1 Fill the space for the inside of the grid
            pygame.draw.rect(screen, Color("azure3"), 
                             (self.padding * cell_size + 2, self.padding * cell_size + 2, 
                              self.grid_size[1] * cell_size - 4, self.grid_size[0] * cell_size - 4))
            
            # 2. Objects on the grid.
            # in pygame (0,0) is the top left corner, same as in numpy
            # but the directions of x,y are swapped
            for x in range(self.padding, self.grid_size[0] + self.padding):
                for y in range(self.padding, self.grid_size[1] + self.padding):
                    # 2. Render the resources
                    if self.terrain[x, y] == 2:
                        self.render_object(screen, asset_paths.food, (x, y), cell_size)
                    elif self.terrain[x, y] == 3:
                        self.render_object(screen, asset_paths.water, (x, y), cell_size)
                    # 3. Render the agents
                    if self.agent_grid[x, y] == 1:
                        self.render_state(screen, screen_config)
                        self.render_object(screen, asset_paths.agent, (x, y), cell_size)

            # 1.2 Draw the grid lines
            for x in range(self.padding * cell_size, (self.grid_size[1] + self.padding) * cell_size, cell_size):
                pygame.draw.line(screen, Color("azure4"), (x, self.padding * cell_size), (x, (self.grid_size[0] + self.padding) * cell_size), 2)
            for y in range(self.padding * cell_size, (self.grid_size[0] + self.padding) * cell_size, cell_size):
                pygame.draw.line(screen, Color("azure4"), (self.padding * cell_size, y), ((self.grid_size[1] + self.padding)* cell_size, y), 2)

            # 1.3 Border rendering
            pygame.draw.rect(screen, Color("black"), 
                             (self.padding * cell_size, self.padding * cell_size, 
                              self.grid_size[1] * cell_size, self.grid_size[0] * cell_size), 4)         

        else:
            raise NotImplementedError("Only 'human' render mode is implemented.")
    def load_image(self, path: str | Path, size: tuple[int, int]) -> pygame.Surface:
        """Load an image and scale it to the specified size."""
        image = pygame.image.load(str(path))
        return pygame.transform.scale(image, size)

    def render_object(self, screen: pygame.Surface, image_path: str | Path, position: tuple[int, int], cell_size: int):
        x, y = position
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        image = self.load_image(image_path, (cell_size, cell_size))
        screen.blit(image, rect)
        screen.blit(image, rect)
        
    def render_state(self, screen, screen_config: ScreenConfig):
        """Render the agent's state (energy, food, water) on the screen."""
        cell_size = screen_config.cell_size
        
        # Energy bar (replaces energy text)
        energy_ratio = self.agent.energy_level / self.cfg.agent.max_energy_level
        bar_height = max(3, cell_size // 10)
        bar_margin = 1
        bar_width_full = cell_size - bar_margin * 2
        filled_width = int(bar_width_full * energy_ratio)

        bar_x = self.agent.position[1] * cell_size + bar_margin
        bar_y = self.agent.position[0] * cell_size + bar_margin
        pygame.draw.rect(screen, Color("gray20"), (bar_x, bar_y, bar_width_full, bar_height))
        pygame.draw.rect(screen, Color("green"), (bar_x, bar_y, filled_width, bar_height))
        pygame.draw.rect(screen, Color("black"), (bar_x, bar_y, bar_width_full, bar_height), 1)
        
        # Food bar (below energy)
        food_ratio = self.agent.food_level / self.cfg.agent.max_food_level
        food_bar_y = bar_y + bar_height + 1
        food_filled_width = int(bar_width_full * food_ratio)
        pygame.draw.rect(screen, Color("gray20"), (bar_x, food_bar_y, bar_width_full, bar_height))
        pygame.draw.rect(screen, Color("red3"), (bar_x, food_bar_y, food_filled_width, bar_height))
        pygame.draw.rect(screen, Color("black"), (bar_x, food_bar_y, bar_width_full, bar_height), 1)

        # Water bar (below food)
        water_ratio = self.agent.water_level / self.cfg.agent.max_water_level
        water_bar_y = food_bar_y + bar_height + 1
        water_filled_width = int(bar_width_full * water_ratio)
        pygame.draw.rect(screen, Color("gray20"), (bar_x, water_bar_y, bar_width_full, bar_height))
        pygame.draw.rect(screen, Color("dodgerblue3"), (bar_x, water_bar_y, water_filled_width, bar_height))
        pygame.draw.rect(screen, Color("black"), (bar_x, water_bar_y, bar_width_full, bar_height), 1)
        