import pygame
import math
import gym
from gym import spaces, logger
from random import seed
from random import randint
from gym.utils import seeding
import numpy as np


class BoxEnv(gym.Env):
    def __init__(self):
        # Environment size and name
        self.game_width = 500
        self.game_height = 500
        self.win = pygame.display.set_mode((self.game_width, self.game_height))
        pygame.display.set_caption("Environment")

        # Agent size and shape
        self.agent_color = (255, 0, 0)
        self.agent_size = 10
        self.agent_min_x_pos = 0
        self.agent_min_y_pos = 0
        self.agent_max_x_pos = self.game_width
        self.agent_max_y_pos = self.game_height

        # Goal size and shape
        self.goal_color = (255, 255, 255)
        self.goal_size = 20

        # Initial agent position
        self.agent_x = 0
        self.agent_y = 0

        # Initial goal position
        self.goal_x = 0
        self.goal_y = 0

        # Other variables
        self.vel = 5
        self.timestep = 0
        self.reward = 0
        self.done = False
        self.goal_distance = 0
        self.degree = 0


        # Observation and action space
        self.low = np.array(
            [0, -360], dtype=np.float32
        )
        self.high = np.array(
            [1000, 360], dtype=np.float32
        )

        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = 0

        # Generating new position variables for the agent
        self.agent_x = randint(100, self.game_width-100)
        self.agent_y = randint(100, self.game_height-100)

        # Generating new position variables for the goal
        self.goal_x = randint(100, self.game_width-100)
        self.goal_y = randint(100, self.game_height-100)

        self.goal_distance = math.hypot(self.goal_x - self.agent_x, self.goal_y - self.agent_y)
        self.degree = (180/math.pi)*math.atan2(self.goal_y - self.agent_y, self.goal_x - self.agent_x)

        state = (self.goal_distance, self.degree)
        return np.array(state)

    def step(self, action):
        self.done = False
        pre_dist = self.goal_distance = math.hypot(self.goal_x - self.agent_x, self.goal_y - self.agent_y)
        reward = 0

        if action == 0 and self.agent_x > self.vel + self.agent_size:
            self.agent_x -= self.vel

        elif action == 1 and self.agent_x < self.game_width - self.agent_size - self.vel:
            self.agent_x += self.vel

        elif action == 2 and self.agent_y > self.vel + self.agent_size:
            self.agent_y -= self.vel

        elif action == 3 and self.agent_y < self.game_height - self.agent_size - self.vel:
            self.agent_y += self.vel
        else:
            self.done = True
            reward -= 100

        # Setting terminal state reward
        if self.timestep == 200 and not self.done:  # We reached the max number of steps
            self.done = True
            reward -= 100

        else:
            self.goal_distance = math.hypot(self.goal_x - self.agent_x, self.goal_y - self.agent_y)
            self.degree = (180 / math.pi) * math.atan2(self.goal_y - self.agent_y, self.goal_x - self.agent_x)

            if self.goal_distance < pre_dist:  # We receive a positive reward for moving towards the goal
                reward += pre_dist - self.goal_distance
            else:
                reward -= 1

            # We we have reached the goal
            if self.goal_distance < self.goal_size:
                reward += 100
                self.done = True

        self.timestep += 1
        next_state = (self.goal_distance, self.degree)
        return np.array(next_state), reward, self.done

    def render(self):
        pygame.init()
        self.win.fill((0, 0, 0))

        # Draw the goal in the game
        pygame.draw.circle(self.win, (255, 255, 255), (self.goal_x, self.goal_y), self.goal_size)
        # Draw the agent in the game
        pygame.draw.circle(self.win, (255, 0, 0), (self.agent_x, self.agent_y), self.agent_size)

        pygame.display.update()

    def close(self):
        pygame.quit()
        quit()








# Wait for the player to press the mouse before moving on
"""
event_happen = False
while not event_happen:
    event = pygame.event.wait()  # Wait for event
    if event.type == pygame.MOUSEBUTTONDOWN:  # If the player clicked on the mouse
        mx, my = pygame.mouse.get_pos()  # Get the mouse position
        event_happen = True  # Stop the loop
"""

"""
game_done = False
reward = 0
reward_list = []
p_env = BoxEnv()
state = (p_env.agent_x, p_env.agent_y, p_env.goal_distance)

seed(1)

for episode in range(10):
    if episode > 0:
        print("Reward: {},   Game Done: {}".format(sum(reward_list), game_done))
        reward_list.clear()
        reward = 0
        game_done = False
        state = p_env.reset()

    while not game_done:
        p_env.render()
        event_happen = False
        print("Goal Distance: {},    Reward: {}".format(state[2], reward))

        while not event_happen:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    state, reward, game_done = p_env.step(0)
                    event_happen = True

                elif keys[pygame.K_RIGHT]:
                    state, reward, game_done = p_env.step(1)
                    event_happen = True

                elif keys[pygame.K_UP]:
                    state, reward, game_done = p_env.step(2)
                    event_happen = True

                elif keys[pygame.K_DOWN]:
                    state, reward, game_done = p_env.step(3)
                    event_happen = True
        reward_list.append(reward)


p_env.close()
"""

