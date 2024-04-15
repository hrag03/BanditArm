# BanditArm

This Python script compares the performance of two Bandit Arm algorithms: Epsilon Greedy and Thompson Sampling. It evaluates their effectiveness in maximizing cumulative rewards in a scenario with multiple bandit arms, each with different reward probabilities.

## Description

The script consists of several classes and functions:

- **Bandit**: An abstract base class defining the interface for bandit algorithms.
- **EpsilonGreedy**: An implementation of the Epsilon Greedy algorithm, which balances exploration and exploitation by choosing between exploring new arms and exploiting the current best arm.
- **ThompsonSampling**: An implementation of the Thompson Sampling algorithm, which leverages Bayesian inference to make decisions by sampling from the posterior distribution of each arm's reward probability.
- **Visualization**: A helper class for plotting the cumulative rewards and learning rates of the algorithms.
- **comparison**: A function to compare the performance of Epsilon Greedy and Thompson Sampling algorithms.
- **main**: The main function orchestrating the execution of the experiments, reporting results, and visualizing the performance.

## Usage

1. **Setup**: Make sure you have Python installed on your system.
2. **Dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **Run**: Execute the script `main.py`.
4. **Results**: The script will output the total cumulative reward, cumulative regret, average rewards, and average regrets for each algorithm. Additionally, it will display plots showing the learning rate and accumulated reward over time for each algorithm.

## Customization

- Adjust the parameters `Bandit_Reward` and `NumberOfTrials` in the `main` function to customize the bandit rewards and the number of trials.
- You can modify the algorithms' parameters and experiment settings within their respective classes to explore different scenarios.

## Logging

The script utilizes Python's logging module to provide information about the execution process. By default, it logs messages at the DEBUG level, but you can adjust the logging level and format in the code.

## Notes

- The `comparison` function conducts a statistical test to evaluate whether the mean rewards obtained by Epsilon Greedy and Thompson Sampling are statistically equivalent.
- For any inquiries or issues, feel free to open an issue on this repository.

