import math
import pickle
import time

from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import matplotlib.pyplot as np

# https://github.com/nuno-faria/tetris-ai

scores = [0]
epsilon_stop_episode = 0

# Run dqn with Tetris
def dqn(saveAs=time.strftime("%Y_%m_%d_%H_%M_%S.h5"), loadmodelsrc="", doTrain=True, episodes=2000, render_after=10000, agent=None):
    global scores, epsilon_stop_episode
    env = Tetris()
    #episodes = 2000 # standard=2000~, går sakte etter 1500
    #render_after = 1950
    epsilon_stop_episode = math.ceil(episodes * 0.75)
    max_steps = None
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 500
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    if not agent:
        if loadmodelsrc:
            agent = DQNAgent(env.get_state_size(),
                             n_neurons=n_neurons, activations=activations,
                             epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                             discount=discount, replay_start_size=replay_start_size)
            agent.model = agent.loadModel(loadmodelsrc)
        else:
            agent = DQNAgent(env.get_state_size(),
                             n_neurons=n_neurons, activations=activations,
                             epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                             discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = [0]

    for episode in tqdm(range(episodes+1)):
        current_state = env.reset()
        done = False
        steps = 0

        if (render_every and episode % render_every == 0) or episode > render_after:
            render = True
            #print("Mean score this", render_every, ": ", mean(scores))
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0 and doTrain:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

    #agent.model.save(str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + str(datetime.now().hour) + ".h5")

    if not loadmodelsrc:
        agent.model.save(saveAs)
        pickle.dump(agent, open(saveAs + ".pickle", "wb"))


if __name__ == "__main__":
    num = 1
    for i in range(num):
        start = time.time()
        dqn(saveAs="ABC.h5", episodes=2500, render_after=2450)
        end = time.time()
        maxi_score = max(scores[epsilon_stop_episode:])
        mean_after_epsilon = mean(scores[epsilon_stop_episode:])
        best_episode = 0
        time.sleep(1)
        print("Run:", i)
        print("Time:", end - start, "sec")
        print("Mean:", mean_after_epsilon)
        print("best score:", maxi_score)
        print("best episode:", scores.index(maxi_score))
        np.plot([x for x in range(len(scores))], scores)
        np.show()

        dqn(agent=pickle.load(open("ABC.h5.pickle", "rb")), doTrain=False, episodes=50, render_after=1)
        maxi_score = max(scores[epsilon_stop_episode:])
        mean_after_epsilon = mean(scores[epsilon_stop_episode:])
        print("Mean:", mean_after_epsilon)
        print("best score:", maxi_score)
