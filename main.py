import numpy as np
from airline.Airline import Airline
from airline.RL_brain import QLearningTable

def update(env, RL):

    for episode in range(1000):
        RL.eligibility_trace *= 0
        observation = env.reset()


        while True:
            env.render()

            action, exploit_state = RL.choose_action(observation)
            observation_, r, done, info = env.step(action)

            RL.learn(observation, action, r, observation_, exploit_state)

            observation = observation_

            if done:
                break
        if episode % 50 == 0:
            print(RL.q_table)

if __name__ == '__main__':
    env = Airline({'capacity': 100}, np.linspace(300, 1100, 20), 30)
    RL = QLearningTable(np.linspace(300, 1100, 20))
    update(env, RL)