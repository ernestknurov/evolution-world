import pygame # type: ignore
from pygame.locals import Color # type: ignore
import numpy as np # type: ignore
import gymnasium as gym # type: ignore
from dataclasses import asdict
from typing import Tuple, Dict, Any
from pathlib import Path

from src.evolution_world.training.configs.enums import Entity, Action
from src.evolution_world.training.configs.config import EnvConfig, ScreenConfig, AssetPaths, AgentState


class MultiAgentGridWorld(gym.Env):
    def __init__(self, cfg: EnvConfig, render_mode: str | None = None, seed: int | None = None):
        self.cfg = cfg
        self.seed = seed
        self.render_mode = render_mode
        self.grid_size = (cfg.map.height, cfg.map.width)
        self.padding = cfg.agent.vision_radius # should be equal to the padding in screen config
        self.terrain_size = (self.grid_size[0] + self.padding * 2, 
                             self.grid_size[1] + self.padding * 2)
        self.num_agents = cfg.num_agents
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
        self._image_cache = {}
        self._font = None
        self._screen_config = None
        self.reset()

    def reset(self, seed: int | None = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
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
        self.agents = {}
        for agent_id in range(self.num_agents):
            self.agents[agent_id] = AgentState(
                energy_level=self.cfg.agent.start_energy_level,
                water_level=self.cfg.agent.start_water_level,
                food_level=self.cfg.agent.start_food_level,
                lifespan=self.cfg.agent.lifespan,
                position=tuple(np.random.randint(self.padding, self.grid_size[0] + self.padding, size=2).tolist())
            )
        self.agent_grid = np.zeros_like(self.terrain)
        for agent_id, agent in self.agents.items():
            self.agent_grid[agent.position] = agent_id + 1  # +1 to avoid zero (empty) in agent grid

        self.illegal_action_count = 0
        self.collisions_count = 0

        # 2. Collecting information
        info = {
            'terrain': self.terrain,
            'agent_grid': self.agent_grid,
            'agents_states': {agent_id: asdict(self.agents[agent_id]) for agent_id in self.agents},
        }
        return self.get_observation(), info

    def get_observation(self) -> Dict[int, Dict[str, Any]]:
        observations = {}
        for agent_id, agent in self.agents.items():
            # 1. Vision
            # The vision is a kxk egocentric window of tiles around the agent,
            x, y = agent.position
            terrain_vision = self.terrain[
                x - self.cfg.agent.vision_radius:x + self.cfg.agent.vision_radius + 1,
                y - self.cfg.agent.vision_radius:y + self.cfg.agent.vision_radius + 1
            ]
            agent_vision = self.agent_grid[
                x - self.cfg.agent.vision_radius:x + self.cfg.agent.vision_radius + 1,
                y - self.cfg.agent.vision_radius:y + self.cfg.agent.vision_radius + 1
            ]
            stratified_terrain_vision = np.eye(self.n_entities, dtype=np.int8)[terrain_vision]
            # split the agent vision into 2 layers: one for the agent position and one for the other agents
            agent_vision_self = (agent_vision == agent_id + 1).astype(np.uint8)
            agent_vision_others = ((agent_vision != 0) & (agent_vision != agent_id + 1)).astype(np.uint8)
            stratified_agent_vision = np.stack((agent_vision_others, agent_vision_self), axis=-1)
            vision = np.concatenate((stratified_terrain_vision, stratified_agent_vision), axis=-1)  # Concatenate terrain and agent vision
            
            # 2. Agent state
            norm_agent_energy_level = agent.energy_level / self.cfg.agent.max_energy_level
            norm_agent_food_level = agent.food_level / self.cfg.agent.max_food_level
            norm_agent_water_level = agent.water_level / self.cfg.agent.max_water_level
            norm_agent_lifespan = agent.lifespan / self.cfg.agent.lifespan
            agent_state = np.array([norm_agent_energy_level, norm_agent_food_level, 
                                    norm_agent_water_level, norm_agent_lifespan], dtype=np.float32)

            # 3. Legal actions
            legal_actions = self.get_legal_actions(agent_id)
            legal_action_mask = np.array([legal_actions[Action(i)] for i in range(self.n_actions)], dtype=np.int8)

            observations[agent_id] = {
                'vision': vision,
                'self': agent_state,
                'action_mask': legal_action_mask,
            }
        return observations

    def get_legal_actions(self, agent_id: int) -> Dict[Action, bool]:
        legal_actions = {
            Action.UP: self.agents[agent_id].position[0] > self.padding,  # Up
            Action.DOWN: self.agents[agent_id].position[0] < self.grid_size[0] - 1 + self.padding,  # Down
            Action.LEFT: self.agents[agent_id].position[1] > self.padding,  # Left
            Action.RIGHT: self.agents[agent_id].position[1] < self.grid_size[1] - 1 + self.padding,  # Right
            Action.EAT: self.terrain[self.agents[agent_id].position] == 2,  # Can eat if there's food
            Action.DRINK: self.terrain[self.agents[agent_id].position] == 3,  # Can drink if there's water
            Action.REST: True,  # Rest action is always legal
        }
        return legal_actions
    
    def get_intended_position(self, agent_id: int, action: Action) -> Tuple[int, int]:
            """Get the intended position of the agent after taking the action."""
            if action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                return (self.agents[agent_id].position[0] + self.action_to_vector[action][0],
                        self.agents[agent_id].position[1] + self.action_to_vector[action][1])
            return self.agents[agent_id].position
    
    def move_agent(self, agent_id: int, action: Action):
        # x, y from array, so x is from top to bottom, y - from left to right
        # x - horizontal (from left to right), y - vertical (from bottom to top)
        self.agent_grid[self.agents[agent_id].position] = 0  
        dx, dy = self.action_to_vector[action]
        self.agents[agent_id].position = self.agents[agent_id].position[0] + dx, self.agents[agent_id].position[1] + dy
        self.agent_grid[self.agents[agent_id].position] = agent_id + 1  # +1 to avoid zero (empty) in agent grid  

    def clip(self, value: float, min_value: float, max_value: float) -> float:
        """Clip the value to the range [min_value, max_value]."""
        return max(min(value, max_value), min_value)
    
    def step(self, action_ids: Dict[int, int]) -> Tuple[Dict[int, Dict[str, Any]], 
                                                        Dict[int, float], 
                                                        Dict[int, bool], 
                                                        Dict[int, bool], 
                                                        Dict[str, Any]]:
        """ Perform a step in the environment.
        Args:
            actions_ids (Dict[int, int]): A dictionary mapping agent IDs to their actions.
        """
        terminated_dict = {}
        truncated_dict = {}
        reward_dict = {}
        info = {}

        actions: Dict[int, Action] = {agent_id: Action(action_id) for agent_id, action_id in action_ids.items()}

        
        # 1. Intent collection (pure read)
        intended_positions = {agent_id: self.get_intended_position(agent_id, action) for agent_id, action in actions.items()}
        
        # 2. resolution # find duplicates in intended positions values
        intended_positions_count = {pos: 0 for pos in intended_positions.values()}
        for pos in intended_positions.values():
            intended_positions_count[pos] += 1
        invalid_actions = {agent_id: actions[agent_id] for agent_id, pos in intended_positions.items() if intended_positions_count[pos] > 1}

        # 3. Action execution
        for agent_id, action in actions.items():

            terminated = False
            truncated = False
            reward = 0

            if not self.get_legal_actions(agent_id)[action]:
                reward += self.cfg.rewards.illegal_action_penalty
                self.illegal_action_count += 1
            elif agent_id in invalid_actions:
                self.collisions_count += 1
            else:
                # State update
                if action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                    self.move_agent(agent_id, action)
                    self.agents[agent_id].energy_level -= self.cfg.agent.energy_drain_move
                elif action == Action.REST:
                    self.agents[agent_id].energy_level += self.cfg.agent.energy_boost_rest
                elif action == Action.EAT:
                    self.agents[agent_id].energy_level += self.cfg.agent.energy_boost_food
                    self.agents[agent_id].food_level += self.cfg.agent.food_boost_eat
                    reward += self.cfg.rewards.consume_food
                    self.terrain[self.agents[agent_id].position] = Entity.EMPTY.value
                elif action == Action.DRINK:
                    self.agents[agent_id].energy_level += self.cfg.agent.energy_boost_water
                    self.agents[agent_id].water_level += self.cfg.agent.water_boost_drink
                    reward += self.cfg.rewards.consume_water
                    self.terrain[self.agents[agent_id].position] = Entity.EMPTY.value


            # Default processes
            self.agents[agent_id].lifespan -= 1
            self.agents[agent_id].energy_level -= self.cfg.agent.energy_drain_default
            self.agents[agent_id].food_level -= self.cfg.agent.food_drain_default
            self.agents[agent_id].water_level -= self.cfg.agent.water_drain_default

            self.agents[agent_id].energy_level = self.clip(self.agents[agent_id].energy_level, self.cfg.agent.min_energy_level, self.cfg.agent.max_energy_level)
            self.agents[agent_id].water_level = self.clip(self.agents[agent_id].water_level, self.cfg.agent.min_water_level, self.cfg.agent.max_water_level)
            self.agents[agent_id].food_level = self.clip(self.agents[agent_id].food_level, self.cfg.agent.min_food_level, self.cfg.agent.max_food_level)

            if self.agents[agent_id].food_level <= 0:
                self.agents[agent_id].energy_level -= self.cfg.agent.energy_drain_food_starvation
                reward += self.cfg.rewards.food_starvation_penalty
            if self.agents[agent_id].water_level <= 0:
                self.agents[agent_id].energy_level -= self.cfg.agent.energy_drain_water_starvation
                reward += self.cfg.rewards.water_starvation_penalty

            # 3. Termination conditions
            if self.agents[agent_id].energy_level <= 0:
                reward += self.cfg.rewards.death_penalty
                terminated = True
            elif self.agents[agent_id].lifespan <= 0:
                terminated = True
            else:
                reward += self.cfg.rewards.alive_tick

            reward_dict[agent_id] = reward
            terminated_dict[agent_id] = terminated
            truncated_dict[agent_id] = truncated

        # Resources regeneration
        resource_id_to_entity_id = {1: 2, 2: 3}  # FOOD -> 2, WATER -> 3
        for resource_id, resource_config in self.cfg.resources.items():
            if np.random.rand() < resource_config.rate:
                empty_tiles = np.flatnonzero(self.terrain == Entity.EMPTY.value)
                if empty_tiles.size > 0:
                    random_tile = np.random.choice(empty_tiles)
                    self.terrain.flat[random_tile] = resource_id_to_entity_id[resource_id]
            
        # 4. collecting useful information
        info = {
            'terrain': self.terrain,
            'agent_grid': self.agent_grid,
            'agent_states': {agent_id: asdict(self.agents[agent_id]) for agent_id in self.agents},
            'num_food': np.sum(self.terrain == Entity.FOOD.value) / self.grid_size[0] / self.grid_size[1],
            'num_water': np.sum(self.terrain == Entity.WATER.value) / self.grid_size[0] / self.grid_size[1],
            'illegal_action_count': self.illegal_action_count,
            'collisions_count': self.collisions_count,
        }
        
        return self.get_observation(), reward_dict, terminated_dict, truncated_dict, info
    
    def _load_and_cache_assets(self, screen_config: ScreenConfig, asset_paths: AssetPaths):
        """Load and cache all assets once"""
        if self._screen_config != screen_config:
            self._screen_config = screen_config
            cell_size = screen_config.cell_size
            
            # Cache scaled images
            self._image_cache = {
                'food': self._load_image(asset_paths.food, (cell_size, cell_size)),
                'water': self._load_image(asset_paths.water, (cell_size, cell_size)),
                'agent': self._load_image(asset_paths.agent, (cell_size, cell_size))
            }
            
            # Cache font
            self._font = pygame.font.Font(None, 16)
    
    def _load_image(self, path: str | Path, size: tuple[int, int]) -> pygame.Surface:
        """Load an image and scale it to the specified size."""
        image = pygame.image.load(str(path))
        return pygame.transform.scale(image, size)

    def render_object_cached(self, screen: pygame.Surface, image_key: str, position: tuple[int, int], cell_size: int):
        """Render using cached images"""
        x, y = position
        rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
        screen.blit(self._image_cache[image_key], rect)

    def render(self, screen, screen_config: ScreenConfig=ScreenConfig(), asset_paths: AssetPaths=AssetPaths()):
        cell_size = screen_config.cell_size
        self._load_and_cache_assets(screen_config, asset_paths)
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
                        self.render_object_cached(screen, "food", (x, y), cell_size)
                    elif self.terrain[x, y] == 3:
                        self.render_object_cached(screen, "water", (x, y), cell_size)
                    # 3. Render the agents
                    if (self.agent_grid[x, y] - 1) in self.agents.keys():
                        agent_id = self.agent_grid[x, y] - 1  # -1 to get the actual agent ID
                        self.render_state(screen, agent_id, screen_config)
                        self.render_object_cached(screen, "agent", (x, y), cell_size)

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
        
    # def load_image(self, path: str | Path, size: tuple[int, int]) -> pygame.Surface:
    #     """Load an image and scale it to the specified size."""
    #     image = pygame.image.load(str(path))
    #     return pygame.transform.scale(image, size)

    # def render_object(self, screen: pygame.Surface, image_path: str | Path, position: tuple[int, int], cell_size: int):
    #     x, y = position
    #     rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
    #     image = self.load_image(image_path, (cell_size, cell_size))
    #     screen.blit(image, rect)
        
    def render_state(self, screen, agent_id, screen_config: ScreenConfig):
        """Render the agent's state (energy, food, water) on the screen."""
        agent = self.agents[agent_id]
        cell_size = screen_config.cell_size
        
        # Energy bar (replaces energy text)
        energy_ratio = agent.energy_level / self.cfg.agent.max_energy_level
        bar_height = max(3, cell_size // 10)
        bar_margin = 1
        bar_width_full = cell_size - bar_margin * 2
        filled_width = int(bar_width_full * energy_ratio)

        bar_x = agent.position[1] * cell_size + bar_margin
        bar_y = agent.position[0] * cell_size + bar_margin
        pygame.draw.rect(screen, Color("gray20"), (bar_x, bar_y, bar_width_full, bar_height))
        pygame.draw.rect(screen, Color("green"), (bar_x, bar_y, filled_width, bar_height))
        pygame.draw.rect(screen, Color("black"), (bar_x, bar_y, bar_width_full, bar_height), 1)
        
        # Food bar (below energy)
        food_ratio = agent.food_level / self.cfg.agent.max_food_level
        food_bar_y = bar_y + bar_height + 1
        food_filled_width = int(bar_width_full * food_ratio)
        pygame.draw.rect(screen, Color("gray20"), (bar_x, food_bar_y, bar_width_full, bar_height))
        pygame.draw.rect(screen, Color("red3"), (bar_x, food_bar_y, food_filled_width, bar_height))
        pygame.draw.rect(screen, Color("black"), (bar_x, food_bar_y, bar_width_full, bar_height), 1)

        # Water bar (below food)
        water_ratio = agent.water_level / self.cfg.agent.max_water_level
        water_bar_y = food_bar_y + bar_height + 1
        water_filled_width = int(bar_width_full * water_ratio)
        pygame.draw.rect(screen, Color("gray20"), (bar_x, water_bar_y, bar_width_full, bar_height))
        pygame.draw.rect(screen, Color("dodgerblue3"), (bar_x, water_bar_y, water_filled_width, bar_height))
        pygame.draw.rect(screen, Color("black"), (bar_x, water_bar_y, bar_width_full, bar_height), 1)

        # // draw the agent id on the borrom right corner of the cell
        agent_id_text = str(agent_id)
        # font = pygame.font.Font(None, 16)
        text_surface = self._font.render(agent_id_text, True, Color("black"))
        
        cell_x = agent.position[1] * cell_size
        cell_y = agent.position[0] * cell_size
        margin = 1
        text_x = cell_x + cell_size - text_surface.get_width() - margin
        text_y = cell_y + cell_size - text_surface.get_height() - margin
        
        text_rect = text_surface.get_rect(topleft=(text_x, text_y))
        screen.blit(text_surface, text_rect)
        