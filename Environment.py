                #### TUTORIAL AND CODE https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0") #this environment has 3 available actions
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95 #measure of how important we find/value future actions/reward over current actions/reward
EPISODES = 10000
#print(env.observation_space.high) #values - [0.6  0.07]
#print(env.observation_space.low)  #values - [-1.2  -0.07]
#print(env.action_space.n) #number of actions the environment has

SHOW_EVERY = 500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


epsilon = 0.5 #this will be used as a random variable so the higher the value the more random
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 #the double // is to make sure it is an intenger and not a float

epsilon_decaying_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#print(q_table.shape)
#print(q_table)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:   #np.random.random creates a random float between 0 and 1
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n) #chooses a random action between 0 and our number of actions
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        #print(reward, new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.unwrapped.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decaying_value


    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        #work on our dictionary
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/SHOW_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
        #this print will give us the metrics
        np.save(f"qtables/{episode}-qtable.pny", q_table) #saving the qtable in the qtables directory

env.close()

#The x axis will always be the episodes
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], Label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], Label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], Label="max")
plt.legend(loc=4) #location of the legend for the graph
plt.show()
