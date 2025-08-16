import pygame # type: ignore
import numpy as np # type: ignore
import gymnasium as gym # type: ignore
from dataclasses import asdict
from typing import Tuple, Dict, Any

from src.evolution_world.training.configs.enums import Entity, Action
from src.evolution_world.training.configs.config import EnvConfig, ScreenConfig, AgentState

CELL_SIZE = ScreenConfig.cell_size
WHITE = (255, 255, 255)
LIGHT_GRAY = (180, 180, 180)  # Darker grid lines for better visibility
GRAY = (150, 170, 190)          
GREEN = (0, 150, 0)           # Bright green for energy - high contrast
RED = (200, 50, 50)           # Keep red as is
BLUE = (0, 100, 200)          # Bright blue for lifespan - high contrast  
BLACK = (0, 0, 0)             # Pure black for maximum contrast

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
    
    def render(self, screen):
        if self.render_mode == 'human':
            # 1. Render the grid
            screen.fill(WHITE)
            for x in range(0, self.grid_size[1] * CELL_SIZE, CELL_SIZE):
                pygame.draw.line(screen, LIGHT_GRAY, (x, 0), (x, self.grid_size[0] * CELL_SIZE))
            for y in range(0, self.grid_size[0] * CELL_SIZE, CELL_SIZE):
                pygame.draw.line(screen, LIGHT_GRAY, (0, y), (self.grid_size[1] * CELL_SIZE, y))

            # in pygame (0,0) is the top left corner, same as in numpy
            # but the directions of x,y are swapped
            for x in range(self.padding, self.grid_size[0] + self.padding):
                for y in range(self.padding, self.grid_size[1] + self.padding):
                    # 2. Render the agents
                    if self.agent_grid[x, y] == 1:
                        agent_x, agent_y = x - self.padding, y - self.padding
                        pygame.draw.circle(screen, GRAY, 
                                            (agent_y * CELL_SIZE + CELL_SIZE // 2, 
                                            agent_x * CELL_SIZE + CELL_SIZE // 2), 
                                            CELL_SIZE // 2)
                        self.render_state(screen)
                    # 3. Render the resources
                    if self.terrain[x, y] == 2:
                        food_x, food_y = x - self.padding, y - self.padding
                        pygame.draw.circle(screen, GREEN, 
                                           (food_y * CELL_SIZE + CELL_SIZE // 2, 
                                            food_x * CELL_SIZE + CELL_SIZE // 2), 
                                           CELL_SIZE // 4)
                    elif self.terrain[x, y] == 3:
                        water_x, water_y = x - self.padding, y - self.padding
                        pygame.draw.circle(screen, BLUE, 
                                           (water_y * CELL_SIZE + CELL_SIZE // 2, 
                                            water_x * CELL_SIZE + CELL_SIZE // 2), 
                                           CELL_SIZE // 4)
                    

        else:
            raise NotImplementedError("Only 'human' render mode is implemented.")
        
    def render_state(self, screen):

        font = pygame.font.Font(None, 16)  # Smaller font for compact display
        
        # Calculate center position of the agent's cell
        cell_center_x = self.agent.position[1] * CELL_SIZE + CELL_SIZE // 2 - self.padding * CELL_SIZE
        cell_center_y = self.agent.position[0] * CELL_SIZE + CELL_SIZE // 2 - self.padding * CELL_SIZE
        
        # Display energy centered above cell center
        energy_text = font.render(f'E:{self.agent.energy_level:.1f}', True, GREEN)
        energy_rect = energy_text.get_rect()
        energy_rect.centerx = cell_center_x
        energy_rect.centery = cell_center_y - 10  # 16 pixels above center
        screen.blit(energy_text, energy_rect)
        
        # Display food level above energy
        food_text = font.render(f'F:{self.agent.food_level:.1f}', True, RED)
        food_rect = food_text.get_rect()
        food_rect.centerx = cell_center_x
        food_rect.centery = cell_center_y + 0   # 8 pixels above center
        screen.blit(food_text, food_rect)
        
        # Display water level below cell center
        water_text = font.render(f'W:{self.agent.water_level:.1f}', True, BLUE)
        water_rect = water_text.get_rect()
        water_rect.centerx = cell_center_x
        water_rect.centery = cell_center_y + 10  # 8 pixels below center
        screen.blit(water_text, water_rect)
        
        # # Display lifespan centered below water level
        # lifespan_text = font.render(f'L:{self.agent.lifespan}', True, BLACK)
        # lifespan_rect = lifespan_text.get_rect()
        # lifespan_rect.centerx = cell_center_x
        # lifespan_rect.centery = cell_center_y + 10  # 16 pixels below center
        # screen.blit(lifespan_text, lifespan_rect)