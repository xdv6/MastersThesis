import gym
import gym_game
import pygame
import matplotlib.pyplot as plt

env = gym.make("GridWorld-v0", render_mode="human")
env.reset()
screen = env.render()

# 0: right, 1: down, 2: left and 3: up  (down en up omgekeerd omdat je met linkshandig assenstelsel werkt)
action = 0
_, reward, done, _, _ = env.step(action)

while not done:
    _, reward, done, _, _ = env.step(action)
    env.render()
env.close()



