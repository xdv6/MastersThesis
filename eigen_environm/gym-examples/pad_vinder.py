import gym
import gym_game
import pygame
import matplotlib.pyplot as plt


env = gym.make("GridWorld-v0", render_mode="rgb_array")
env.reset()
screen = env.render()

# 0: right, 1: down, 2: left and 3: up
action = 0
_, reward, done, _, _ = env.step(action)
screen2 = env.render()


action = 1
_, reward, done, _, _ = env.step(action)
screen3 = env.render()


plt.figure()
plt.axis('off')
plt.imshow(screen)
plt.show()


plt.figure()
plt.axis('off')
plt.imshow(screen2)
plt.show()


plt.figure()
plt.axis('off')
plt.imshow(screen3)
plt.show()

env.close()
