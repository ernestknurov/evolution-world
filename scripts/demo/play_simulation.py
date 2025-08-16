import pygame # type: ignore
from pygame.locals import * # type: ignore

from src.evolution_world.envs.grid_world import GridWorld
from src.evolution_world.agents import RandomAgent, RLAgent
from src.evolution_world.training.configs.enums import Action
from src.evolution_world.training.configs.config import EnvConfig, ScreenConfig, TrainingConfig

WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def init_pygame() -> tuple[pygame.Surface, pygame.time.Clock, int]:
    pygame.init()
    screen_config = ScreenConfig()
    screen = pygame.display.set_mode((screen_config.width, screen_config.height))
    pygame.display.set_caption(screen_config.title)
    screen.fill(WHITE)
    clock = pygame.time.Clock()
    return screen, clock, screen_config.cell_size


if __name__ == "__main__":
    screen, clock, CELL_SIZE = init_pygame()
    env = GridWorld(cfg=EnvConfig(), render_mode='human', seed=42)
    training_cfg = TrainingConfig()
    # agent = RandomAgent()
    agent = RLAgent(training_cfg.load_path)

    obs, info = env.reset()
    done = False
    control_mode = 'human' 
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN: # type: ignore
                if event.key == K_h: # type: ignore
                    print("Switch to human control mode")
                    control_mode = 'human'
                elif event.key == K_a: # type: ignore
                    print("Switch to agent control mode")
                    control_mode = 'agent' 
                
                if control_mode == 'human':
                    # Agent takes action based on its policy
                    if event.key == K_ESCAPE: # type: ignore
                        done = True
                        continue
                    elif event.key == K_SPACE: # type: ignore
                        if done:
                            obs, info = env.reset()
                            done = False
                        action = agent.act(obs)
                        obs, reward, done, truncated, info = env.step(action)
                        print(f"Action: {Action(action)}, Reward: {reward}, position: {info['agent_state']['position']}")

        if control_mode == 'agent':
            if done:
                obs, info = env.reset()
                done = False
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {Action(action)}, Reward: {reward}, position: {info['agent_state']['position']}")
        # Render outside the event loop so it happens every frame
        env.render(screen)
        pygame.display.update()  # Update the display
        clock.tick(10)  # Limit to 10 FPS for demo purposes instead of using delay

    pygame.quit()
