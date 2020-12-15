import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95  # importance of future rewards
EPISODES = 10000
SHOW_EVERY = 500

# Space Information
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

# splitting our Table into 20 chunks for our data range
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # OS = Observation Space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# init Epsilon
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2  # dividing to Integer and not Float
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Initializing Random Data
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# every combination of position and velocity is in q_table, tells what action is best to take
# print(q_table.shape)
ep_rewards = []  # list
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}  # dictionary


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0: render = True
    else: render = False
    discrete_state = get_discrete_state(env.reset())  # first Action
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # 0, 1, 2
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)  # position & velocity# _ = ignoring value
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render: env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])  # q-value gets backpropagated
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))

        # print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[SHOW_EVERY:])})

        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = "avg")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
        plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")

        plt.legend(location=4)
        plt.show()
env.close()
