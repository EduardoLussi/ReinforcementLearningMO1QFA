import numpy as np

from QFA import QFA
from actor_critic import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = QFA()
    agent = Agent()

    EPISODES = 100

    file_name = 'qfa.png'
    figure_file = 'plots/' + file_name

    best_score = 1
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(EPISODES):
        print(f"\n{'='*100}")
        print(f"EPISODE {i}")
        observation = env.reset()
        done = False
        score = 0
        while not done:
            print(f"{'='*10} STEP {'='*10}")
            action = agent.choose_action(observation)
            print(f"Action: {action*100}e+2")
            observation_, reward, done = env.step(action)
            print(f"Observation: {observation_[0]*100:.2f} {int(observation_[1])}")
            error = np.abs(observation_[0]-env._expected)
            print(f"Error:       {error*100:.2f}")
            print(f"Reward: {reward}")
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
        
        print(f"END EPISODE {i}: score {score:.1f}, avg_score {avg_score:.1f}")
    
    if not load_checkpoint:
        x = [i+1 for i in range(EPISODES)]
        plot_learning_curve(x, score_history, figure_file)