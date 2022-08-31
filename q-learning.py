import numpy as np
import random
import gym
from time import sleep
from IPython.display import clear_output


# Creating env
env = gym.make("Taxi-v3").env
env.reset()

q_table = np.zeros([env.observation_space.n, env.action_space.n]) # 500*6
alpha = 0.1
gamma = 0.95
epsilon = 0.1
all_epochs = []
all_penalties = []

for i in range(1, 200001):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False
    #---#
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, info = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        if reward == -10:
            penalties += 1
        state = next_state
        epochs += 1
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

env.reset()
frames = []
state = env.s ; done = False
while not done: 
    state, reward, done, info = env.step(np.argmax(q_table[state]))
    #---#
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
    })
    #---#
    epochs += 1

def Frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(1)
        
Frames(frames)
