# How Does User Behavior Evolve During Exploratory Visual Analysis?

## ForeCache Models

This repository contains implementations of various algorithms for FORECACHE USER STUDY, as well as scripts for experiments, data analysis, and visualization. Below is the folder structure and a brief description of each directory:

## A_Simplified_Implementation

This directory contains the simple implementation of all algorithms.
- `ActorCritic.py`: Implementation of the Actor-Critic algorithm.
- `NaiveModel.py`: Implementation of a naive caching model.
- `QLearning.py`: Implementation of the Q-learning algorithm.
- `SARSA.py`: Implementation of the SARSA algorithm.
- `Win-Stay-Loose-Shift.py`: Implementation of the Win-Stay-Loose-Shift algorithm.
- `environment2.py`: Environment setup for experiments.
- `misc.py`: Miscellaneous utility functions.
- `plotting.py`: Plotting functions for visualization.


### Experiments_Folder

This directory contains the results of experiments conducted using the implemented algorithms.

### Gold

Clean results of the experiments with charts generated for analysis.

- `ActorCritic.csv`: Results of the Actor-Critic algorithm.
- `Greedy.csv`: Results of the Greedy algorithm.
- `Naive.csv`: Results of the Naive caching model.
- `QLearn.csv`: Results of the Q-learning algorithm.
- `Reinforce.csv`: Results of the Reinforce algorithm.
- `SARSA.csv`: Results of the SARSA algorithm.
- `WSLS.csv`: Results of the Win-Stay-Loose-Shift algorithm.
- `experiments-master-with-reinforce.csv`: Additional experiment data for visualization

### Results

Other rounds of experiments, where new experiment data is stored.
- `experiments-master_new.csv`: Additional experiment data.

### Experiment Visualization

Files related to experiment result visualization using Tableau- Paper plots

### ForeCache_Notebooks

Jupyter notebooks for data analysis and exploration.

### Coallate_Experiment_Results

Notebooks for combining results from various algorithms.

- `Accuracy_per_state.ipynb`: Notebook for calculating accuracy per state.
- `Actor-Critic-Correlations.ipynb`: Notebook for analyzing correlations in Actor-Critic results.
- `Merge-Experiments.ipynb`: Notebook for merging experiment data.


### Exploratory-N-Grams

Additional exploration of the dataset using N-Grams.

### Probability_Distribution

Statistical analysis of datasets.

- `Probability_Distribution-Annotated-Subtask.ipynb`: Notebook for annotated subtask analysis of probability distribution.
- `Probability_Distribution-Context.ipynb`: Notebook for context analysis of probability distribution.
- `Probability_Distribution.ipynb`: Notebook for general probability distribution analysis.
- `Probability_Reward.ipynb`: Notebook for reward analysis.

### Statistical_Tests

Notebooks for conducting various statistical tests.

- `ManKendall-Bandits.ipynb`: Notebook for Mann-Kendall test on bandit data.
- `ManKendall-trend.ipynb`: Notebook for Mann-Kendall trend analysis.
- `ROI-Subtask-Wincoxon-Signed-Rank-Probability_Distribution.ipynb`: Notebook for ROI subtask analysis using Wilcoxon signed-rank test.
- `Regions-Aggregate-Per-State.ipynb`: Notebook for region aggregation per state analysis.
- `Regions-Subtask-Wincoxon-Signed-Rank-Probability_Distribution.ipynb`: Notebook for region subtask analysis using Wilcoxon signed-rank test.
- `Wincoxon-Signed-Rank-Probability_Distribution.ipynb`: Notebook for Wilcoxon signed-rank test on probability distribution.

## Tableau

Tableau workbooks for Log and Distribution visualization.

## How to Run Algorithms

In ForeCache Models/

```bash
python AlgorithmName.py
```

Example:

To run the Actor-Critic algorithm, execute the following command in your terminal:

```bash
python ActorCritic.py
```
