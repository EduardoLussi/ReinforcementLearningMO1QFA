import time
import json

import numpy as np

from QFA import QFA
from actor_critic import Agent

if __name__ == '__main__':
    env = QFA()
    agent = Agent()

    EPISODES = 50

    file_name = 'qfa.png'
    figure_file = 'plots/' + file_name

    best_score = 1
    score_history = []
    load_checkpoint = False

    time_file = time.strftime("%m%d-%H%M")

    if load_checkpoint:
        agent.load_models()
    
    episodes_history = []

    for i in range(EPISODES):
        episode_history = {}

        print(f"\n{'='*100}")
        print(f"EPISODE {i}")
        observation = env.reset()
        done = False
        score = 0
        j = 0
        while not done:
            print(f"{'='*10} STEP {'='*10}")
            action = agent.choose_action(observation)
            print(f"Action: {action*100}e-2")
            try:
                observation_, reward, done = env.step(action)
            except Exception as err:
                print(err)
                break
            print(f"Observation: {observation_[0]*100:.2f} {int(observation_[1]*1000)}")
            error = np.abs(observation_[0]-env._expected)
            print(f"Error:       {error*100:.2f}")
            print(f"Reward: {reward}")
            if done:
                print("DONE!\n")
            score += reward
            if not load_checkpoint:
                actor_loss, critic_loss = agent.learn(observation, reward, observation_, done)
            observation = observation_

            episode_history[j] = {
                'action': action,
                'observation': observation_.tolist(),
                'reward': reward,
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
            }
            j += 1

        episodes_history.append({i: episode_history})

        with open(f'./tests/{time_file}.json', 'w') as json_file:
            json.dump(episodes_history, json_file)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f"END EPISODE {i}: score {score:.1f}, avg_score {avg_score:.1f}")
