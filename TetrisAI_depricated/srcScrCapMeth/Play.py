from srcScrCapMeth.Environment import *

# Make gym environment
env = Environment()

# Play random
done = True
for step in range(5000):
    if done:
        state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print()
    env.render()

# Close device
env.close()
