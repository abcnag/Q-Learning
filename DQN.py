import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

batch_size = 64
alpha = 0.7
gamma = 0.95
epsilon = 0.99
show_per = False


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv:
    size = 10
    return_image = True
    move_penalty = 1
    enemy_penalty = 300
    food_reward = 25
    observation_space = (size, size, 3)  
    action_space = 9
    player_n = 1 
    food_n = 2  
    enemy_n = 3  
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.size)
        self.food = Blob(self.size)
        while self.food == self.player:
            self.food = Blob(self.size)
        self.enemy = Blob(self.size)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.size)

        self.episode_step = 0

        if self.return_image:
            observation = np.array(self.get_image())
        # else:
            # observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        if self.return_image:
            new_observation = np.array(self.get_image())
        # else:
            # new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.enemy_penalty
        elif self.player == self.food:
            reward = self.food_reward
        else:
            reward = -self.move_penalty

        done = False
        if reward == self.food_reward or reward == -self.enemy_penalty or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300)) 
        cv2.imshow("image", np.array(img)) 
        cv2.waitKey(1)

    def get_image(self):
        env = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        env[self.food.x][self.food.y] = self.d[self.food_n] 
        env[self.enemy.x][self.enemy.y] = self.d[self.enemy_n] 
        env[self.player.x][self.player.y] = self.d[self.player_n] 
        img = Image.fromarray(env, 'RGB')
        return img

env = BlobEnv()
ep_rewards = [-200]

random.seed(1)
np.random.seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

        self.replay_memory = deque(maxlen=5000)

        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(8, (3, 3), input_shape=env.observation_space))  
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(8, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(32))

        model.add(Dense(env.action_space, activation='linear'))  
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < 1000:
            return

        minibatch = random.sample(self.replay_memory, batch_size)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                currnet_q = current_qs_list[index][action]
                new_q = (1-alpha) * currnet_q + alpha * (reward + gamma * max_future_q)
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=batch_size, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > 5:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

agent = DQNAgent()

for episode in tqdm(range(1, 20000 + 1), ascii=True, unit='episodes'):

    episode_reward = 0
    step = 1

    current_state = env.reset()

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space)

        new_state, reward, done = env.step(action)

        episode_reward += reward

        if show_per and not episode % 50:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    ep_rewards.append(episode_reward)
    if not episode % 50 or episode == 1:
        average_reward = sum(ep_rewards[-50:])/len(ep_rewards[-50:])
        min_reward = min(ep_rewards[-50:])
        max_reward = max(ep_rewards[-50:])

        if min_reward >= -200:
            agent.model.save(f'models/{-200}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    if epsilon > 0.001:
        epsilon *= 0.99975
        epsilon = max(0.001, epsilon)
