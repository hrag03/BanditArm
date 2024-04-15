"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#



class Visualization():

    @staticmethod
    def plot1(data, labels, title):
        plt.figure(figsize=(10, 6))
        for i in range(len(data)):
            plt.plot(data[i], label=labels[i])
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()

    @staticmethod
    def plot2(data1, data2, labels, title):
        plt.figure(figsize=(10, 6))
        plt.plot(data1, label=labels[0])
        plt.plot(data2, label=labels[1])
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.show()



#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p):
        self.p = p 
        self.p_estimate = 0
        self.N = 0

    def __repr__(self):
        return f'An Arm with {self.p} Win Rate'

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

    def experiment(self, BANDIT_REWARDS, NUM_TRIALS):
        bandits = [EpsilonGreedy(i) for i in BANDIT_REWARDS]
        cumulative_regret = 0
        cumulative_regret_history = {}
        learning_rate_history = []
        rewards_history = []
        bandit_choices = []
        regret_track = []
        for i in range(1, NUM_TRIALS + 1):
            epsilon = 1 / i
            random_prob = np.random.random()
            if random_prob < epsilon:
                chosen_bandit_index = np.random.choice(len(bandits))
            else:
                chosen_bandit_index = np.argmax([j.p_estimate for j in bandits])
            chosen_bandit = bandits[chosen_bandit_index]
            bandit_choices.append(chosen_bandit_index)
            reward = chosen_bandit.pull()
            chosen_bandit.update(reward)
            rewards_history.append(reward)


            regret = max([j.pull() for j in bandits]) - reward
            cumulative_regret += regret
            cumulative_regret_history[i] = cumulative_regret
            regret_track.append(regret)
            avg_reward = np.mean(rewards_history)
            learning_rate_history.append(avg_reward)

        cumulative_avg_reward = np.cumsum(rewards_history) / np.arange(1, NUM_TRIALS + 1)
        cumulative_avg_regret = np.cumsum(cumulative_regret) / np.arange(1, NUM_TRIALS + 1)

        estimated_avg_rewards = [bandit.p_estimate for bandit in bandits]
        self.estimated_avg_rewards = estimated_avg_rewards

        self.cumulative_reward = np.cumsum(rewards_history)
        self.cumulative_avg_reward = cumulative_avg_reward
        self.rewards = rewards_history

        self.cumulative_regret = cumulative_regret
        self.cumulative_avg_regret = cumulative_avg_regret
        self.cumulative_regret_history = cumulative_regret_history

        self.learning = learning_rate_history
        self.bandit_choices = bandit_choices
        self.best_bandit_index = np.argmax([j.p_estimate for j in bandits])

    def report(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(8, 12)
        ax1.plot(self.learning)
        ax2.plot(self.cumulative_reward)
        ax1.set_title('Learning Rate with epsilon 1/t')
        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Expected Reward')
        ax2.set_title('Accumulated Reward with epsilon 1/t')
        ax2.set_xlabel('Trials')
        ax2.set_ylabel('Cumulative Reward')
        logger.info(f'Total Cumulative Reward: {np.sum(self.rewards)}')
        logger.info(f'Total Cumulative Regret: {self.cumulative_regret}')
        logger.info(f'Average Rewards: {self.cumulative_avg_reward}')
        logger.info(f'Average Regrets: {self.cumulative_avg_regret}')
        plt.show()
        return

#--------------------------------------#

class ThompsonSampling(Bandit):

    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.mean = 0
        self.lambda_ = 1
        self.tau = 1
        self.trials = 0
        self.total_reward = 0

    def __repr__(self):
        return f"Thompson Sampling with mean {self.mean}"
        
    def pull(self):
        return (np.random.randn() / np.sqrt(self.tau)) + self.mean

    def sample(self):
        return (np.random.randn() / np.sqrt(self.lambda_)) + self.mean

    def update(self, reward):
        self.lambda_ += self.tau
        self.total_reward += reward
        self.mean = (self.tau * self.total_reward) / self.lambda_
        self.trials += 1

    def experiment(self, BANDIT_REWARDS, NUM_TRIALS):
        bandits = [ThompsonSampling(mean) for mean in BANDIT_REWARDS]
        rewards_history = []
        learning_rate_history = []
        bandit_choices = []
        cumulative_regret = 0
        cumulative_regret_history = {}

        for trial in range(1, NUM_TRIALS + 1):
            chosen_bandit_index = np.argmax([j.sample() for j in bandits])
            chosen_bandit = bandits[chosen_bandit_index]
            learning_rate_history.append(chosen_bandit.mean)
            bandit_choices.append(chosen_bandit_index)

            reward = chosen_bandit.pull()
            chosen_bandit.update(reward)
            rewards_history.append(reward)

            regret = max([j.sample() for j in bandits]) - reward
            cumulative_regret += regret
            cumulative_regret_history[trial] = cumulative_regret

        cumulative_average_reward = np.cumsum(rewards_history) / np.arange(1, NUM_TRIALS + 1)
        cumulative_average_regret = np.cumsum(cumulative_regret) / np.arange(1, NUM_TRIALS + 1)

        self.rewards = rewards_history
        self.cumulative_reward = np.cumsum(rewards_history)
        self.cumulative_average_reward = cumulative_average_reward
        self.cumulative_regret_history = cumulative_regret_history
        self.cumulative_average_regret = cumulative_average_regret
        self.cumulative_regret = cumulative_regret
        self.learning = learning_rate_history
        self.bandit_choices = bandit_choices
        self.best_bandit_index = np.argmax([j.sample() for j in bandits])

    def report(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(10, 10)
        ax1.plot(self.learning)
        ax2.plot(self.cumulative_reward)
        ax1.set_title('Learning Rate with Thompson Sampling')
        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Expected Reward')
        ax2.set_title('Accumulated Reward with Thompson Sampling')
        ax2.set_xlabel('Trials')
        ax2.set_ylabel('Cumulative Reward')
        logger.info(f'Total Cumulative Reward: {np.sum(self.rewards)}')
        logger.info(f'Total Cumulative Regret: {self.cumulative_regret}')
        logger.info(f'Average Rewards: {self.cumulative_average_reward}')
        logger.info(f'Average Regrets: {self.cumulative_average_regret}')
        plt.show()
        return

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def comparison(BANDIT_REWARDS, NUM_TRIALS):
    epsilon_greedy = EpsilonGreedy(1)
    epsilon_greedy.experiment(BANDIT_REWARDS, NUM_TRIALS)
    epsilon_cumulative_reward = epsilon_greedy.cumulative_reward

    thompson_sampling = ThompsonSampling(1)
    thompson_sampling.experiment(BANDIT_REWARDS, NUM_TRIALS)
    thompson_cumulative_reward = thompson_sampling.cumulative_reward

    epsilon_best_reward = BANDIT_REWARDS[epsilon_greedy.best_bandit_index]
    thompson_best_reward = BANDIT_REWARDS[thompson_sampling.best_bandit_index]

    if thompson_best_reward > epsilon_best_reward:
        print('The Thompson Sampling algorithm performed better as it discovered a more rewarding arm.')
    elif epsilon_best_reward > thompson_best_reward:
        print('The Epsilon Greedy algorithm performed better as it discovered a more rewarding arm.')
    else:
        print('Both algorithms found equally rewarding arms. Further analysis is required.')
        epsilon_rewards = epsilon_greedy.rewards_history
        thompson_rewards = thompson_sampling.rewards
        epsilon_cumulative_reward = np.cumsum(epsilon_rewards)
        thompson_cumulative_reward = np.cumsum(thompson_rewards)
        diff_ls = epsilon_cumulative_reward - thompson_cumulative_reward
        plt.plot(diff_ls)
        plt.title('Cumulative Reward Difference (Epsilon Greedy - Thompson Sampling)')
        plt.xlabel('Number of trials')
        plt.ylabel('Cumulative Reward Difference')
        plt.show()

        print('The T-test evaluates whether the mean rewards achieved through Epsilon Greedy are statistically equivalent to those attained through Thompson Sampling.')
        print(ttest_ind(epsilon_rewards, thompson_rewards, alternative='two-sided'))

def main():
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000
    import random as rn
    rn.seed(200)


    epsilon_greedy = EpsilonGreedy(1)
    epsilon_greedy.experiment(Bandit_Reward, NumberOfTrials)
    epsilon_greedy.report()
    thompson_sampling1 = ThompsonSampling(1)
    thompson_sampling1.experiment(Bandit_Reward, NumberOfTrials)
    thompson_sampling1.report()

    comparison(Bandit_Reward, NumberOfTrials)


    rewards_data = [epsilon_greedy.cumulative_avg_reward]
    labels = ['Epsilon Greedy']
    title = 'Epsilon Greedy Algorithm Performance'
    Visualization.plot1(rewards_data, labels, title)

    thompson_sampling = ThompsonSampling(1)
    thompson_sampling.experiment(Bandit_Reward, NumberOfTrials)
    thompson_cumulative_reward = thompson_sampling.cumulative_reward

    rewards_data.append(thompson_cumulative_reward)
    labels.append('Thompson Sampling')
    title = 'Comparison of Epsilon Greedy and Thompson Sampling'
    Visualization.plot2(rewards_data[0], rewards_data[1], labels, title)
    




if __name__=='__main__':
    main()
    # logger.debug("debug message")
    # logger.info("info message")
    # logger.warning("warning message")
    # logger.error("error message")
    # logger.critical("critical message")
