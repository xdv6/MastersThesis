import gym
import gym_game
import pygame
import matplotlib.pyplot as plt
"""
I: show example of gameplay in pygame window
"""

# env = gym.make("GridWorld-v0", render_mode="human")


# env.reset()
# screen = env.render()

# # 0: right, 1: down, 2: left and 3: up  (down en up omgekeerd omdat je met linkshandig assenstelsel werkt)
# action = 3
# _, reward, done, _, _ = env.step(action)

# while not done:
#     _, reward, done, _, _ = env.step(action)
#     env.render()
# env.close()


"""
II: show example to screen in the form of an image
"""

env = gym.make("GridWorld-v0", render_mode="rgb_array")

env.reset()
# screen = env.render()

reward = 0
while not reward == -1:
    _, reward, done, _, info = env.step(1)
    print(f"reward: {reward}")
    value = info["distance"]
    print(f"Manhattan distance: {value}")

print(f"reward: {reward}")
screen = env.render()
env.close()


plt.figure()
plt.axis('off')
plt.imshow(screen)
plt.show()