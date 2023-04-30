import sys
import os 
import gym

sys.path.append(os.path.abspath('../../Prototree-main'))

print(sys.path)

from util.parent import hello

hello()
# import gym_game



# """
# II: show example to screen in the form of an image
# """

# env = gym.make("GridWorld-v0", render_mode="rgb_array")

# env.reset()
# action = 3
# # _, reward, done, _, _ = env.step(0)
# # _, reward, done, _, _ = env.step(action)
# # _, reward, done, _, _ = env.step(action)
# # import ipdb; ipdb.set_trace()
# screen = env.render()