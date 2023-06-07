import numpy as np

from QFA import QFA
from actor_critic import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = QFA()
    agent = Agent()

    EPISODES = 20

    file_name = 'qfa.png'
    figure_file = 'plots/' + file_name

    best_score = 1
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(EPISODES):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            print(f"Action: {action}")
            print(f"Observation: {observation_}")
            print(f"Error: {np.abs(observation_[:-1]-env._expected)}")
            if done:
                print("DONE!\n")
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        print(f"Episode {i}, score {score:.1f}, avg_score {avg_score:.1f}")
        print(f"{'='*100}\n")
    
    if not load_checkpoint:
        x = [i+1 for i in range(EPISODES)]
        plot_learning_curve(x, score_history, figure_file)