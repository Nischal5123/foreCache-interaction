📦ForeCache_Models
 ┣ 📂A_Simplified_Implementation: Implementation of all algorithms: RUN python AlgorithmName.py
 ┃ ┣ 📜ActorCritic.py
 ┃ ┣ 📜NaiveModel.py
 ┃ ┣ 📜QLearning.py
 ┃ ┣ 📜SARSA.py
 ┃ ┣ 📜Win-Stay-Loose-Shift.py
 ┃ ┣ 📜environment2.py
 ┃ ┣ 📜misc.py
 ┃ ┗ 📜plotting.py
 ┣ 📂Experiments_Folder
 ┃ ┣ 📂Gold : Clean results of the experiments- Charts are produced from here
 ┃ ┃ ┣ 📜ActorCritic.csv
 ┃ ┃ ┣ 📜Greedy.csv
 ┃ ┃ ┣ 📜Naive.csv
 ┃ ┃ ┣ 📜QLearn.csv
 ┃ ┃ ┣ 📜Reinforce.csv
 ┃ ┃ ┣ 📜SARSA.csv
 ┃ ┃ ┣ 📜WSLS.csv
 ┃ ┃ ┗ 📜experiments-master-with-reinforce.csv
 ┃ ┣ 📂Results - Other rounds of experiments- Where new experiment data is dumped
 ┃ ┃ ┣ 📜ActorCritic.csv
 ┃ ┃ ┣ 📜Greedy.csv
 ┃ ┃ ┣ 📜Momentum.csv
 ┃ ┃ ┣ 📜Naive.csv
 ┃ ┃ ┣ 📜QLearn.csv
 ┃ ┃ ┣ 📜Random.csv
 ┃ ┃ ┣ 📜Reinforce.csv
 ┃ ┃ ┣ 📜SARSA.csv
 ┃ ┃ ┣ 📜Tamer_Q_Learning.csv
 ┃ ┃ ┣ 📜WSLS.csv
 ┃ ┃ ┗ 📜experiments-master_new.csv
 ┃ ┣ 📜.~Experiment_Visualization__79112.twbr
 ┃ ┣ 📜Experiment_Visualization.twb
 ┃ ┣ 📜Greedy.csv
 ┃ ┣ 📜HyperparameterPerState.csv
 ┃ ┣ 📜Momentum.csv
 ┃ ┣ 📜Naive.csv
 ┃ ┣ 📜experiments-master-original.csv
 ┃ ┣ 📜experiments-master.csv
 ┃ ┣ 📜~Experiment_Visualization__1016.twbr
 ┃ ┣ 📜~Experiment_Visualization__2216.twbr
 ┃ ┣ 📜~Experiment_Visualization__22640.twbr
 ┃ ┣ 📜~Experiment_Visualization__24008.twbr
 ┃ ┗ 📜~Experiment_Visualization__30052.twbr
 ┣ 📂ForeCache_Notebooks
 ┃ ┣ 📂Coallate_Experiment_Results : To combine all Algorithms results
 ┃ ┃ ┣ 📜Accuracy_per_state.ipynb
 ┃ ┃ ┣ 📜Actor-Critic-Correlations.ipynb
 ┃ ┃ ┣ 📜Merge-Experiments.ipynb
 ┃ ┃ ┗ 📜State_Accuracy_Fix.ipynb
 ┃ ┣ 📂Exploratory-N-Grams : Additional Exploration of the Dataset with NGrams
 ┃ ┃ ┣ 📜N-Gram-Context.ipynb
 ┃ ┃ ┣ 📜N-Gram-Longitudinal-Probabilistic.ipynb
 ┃ ┃ ┣ 📜N-Gram-Longitudinal.ipynb
 ┃ ┃ ┣ 📜N-Gram-Tuning.ipynb
 ┃ ┃ ┗ 📜N-Gram.ipynb
 ┃ ┣ 📂Probability_Distribution : Statistical Analysis of Datasets
 ┃ ┃ ┣ 📜Probability_Distribution-Annotated-Subtask.ipynb
 ┃ ┃ ┣ 📜Probability_Distribution-Context.ipynb
 ┃ ┃ ┣ 📜Probability_Distribution.ipynb
 ┃ ┃ ┗ 📜Probability_Reward.ipynb
 ┃ ┗ 📂Statistical_Tests : Statistical Tests- Includes ManKendall and other exploratory tests
 ┃ ┃ ┣ 📜ManKendall-Bandits.ipynb
 ┃ ┃ ┣ 📜ManKendall-trend.ipynb
 ┃ ┃ ┣ 📜ROI-Subtask-Wincoxon-Signed-Rank-Probability_Distribution.ipynb
 ┃ ┃ ┣ 📜Regions-Aggregate-Per-State.ipynb
 ┃ ┃ ┣ 📜Regions-Subtask-Wincoxon-Signed-Rank-Probability_Distribution.ipynb
 ┃ ┃ ┗ 📜Wincoxon-Signed-Rank-Probability_Distribution.ipynb
 ┣ 📂Tableau : Tableau workbooks for visulization 
 ┃ ┣ 📜2D-Region-Locations.twb
 ┃ ┣ 📜Actor-Critic-Hyperparamters-Threshold.twb
 ┃ ┣ 📜Actor_Critic_Experiments.twb
 ┃ ┣ 📜Aggregate-ROI-ManWhitneyResults.twb
 ┃ ┣ 📜Aggregate-Region-ManWhitneyResults.twb
 ┃ ┣ 📜Annotated-Subtasks.twb
 ┃ ┣ 📜Location-3D.twb
 ┃ ┣ 📜Location.twb
 ┃ ┣ 📜ManKendall.twb
 ┃ ┣ 📜ManWhitneyResults.twb
 ┃ ┣ 📜ROI-ManWhitneyResults.twb
 ┃ ┣ 📜Region-ManWhitneyResults.twb
 ┃ ┣ 📜State_Action_Loose_Probability.twb
 ┃ ┣ 📜Tableu-nonstationary.twb
 ┃ ┣ 📜Tableu-results.twb
 ┃ ┣ 📜~2D-Region-Locations__15868.twbr
 ┃ ┣ 📜~2D-Region-Locations__15872.twbr
 ┃ ┣ 📜~2D-Region-Locations__17844.twbr
 ┃ ┣ 📜~2D-Region-Locations__18836.twbr
 ┃ ┣ 📜~2D-Region-Locations__2060.twbr
 ┃ ┣ 📜~2D-Region-Locations__21912.twbr
 ┃ ┣ 📜~2D-Region-Locations__22000.twbr
 ┃ ┣ 📜~2D-Region-Locations__22860.twbr
 ┃ ┣ 📜~2D-Region-Locations__23348.twbr
 ┃ ┣ 📜~2D-Region-Locations__23680.twbr
 ┃ ┣ 📜~2D-Region-Locations__25436.twbr
 ┃ ┣ 📜~2D-Region-Locations__26504.twbr
 ┃ ┣ 📜~2D-Region-Locations__2716.twbr
 ┃ ┣ 📜~2D-Region-Locations__5644.twbr
 ┃ ┣ 📜~2D-Region-Locations__6576.twbr
 ┃ ┣ 📜~2D-Region-Locations__6592.twbr
 ┃ ┣ 📜~2D-Region-Locations__9288.twbr
 ┃ ┣ 📜~Actor-Critic-Hyperparamters-Threshold__12648.twbr
 ┃ ┣ 📜~Aggregate-ROI-ManWhitneyResults__28136.twbr
 ┃ ┣ 📜~Aggregate-Region-ManWhitneyResults__12776.twbr
 ┃ ┣ 📜~Aggregate-Region-ManWhitneyResults__17996.twbr
 ┃ ┣ 📜~Aggregate-Region-ManWhitneyResults__23340.twbr
 ┃ ┣ 📜~Aggregate-Region-ManWhitneyResults__2912.twbr
 ┃ ┣ 📜~Aggregate-Region-ManWhitneyResults__30532.twbr
 ┃ ┣ 📜~Annotated-Subtasks__332.twbr
 ┃ ┣ 📜~ManWhitneyResults__16892.twbr
 ┃ ┣ 📜~ROI-ManWhitneyResults__14924.twbr
 ┃ ┣ 📜~ROI-ManWhitneyResults__18904.twbr
 ┃ ┣ 📜~ROI-ManWhitneyResults__29924.twbr
 ┃ ┣ 📜~Region-ManWhitneyResults__29792.twbr
 ┃ ┣ 📜~Region-ManWhitneyResults__5572.twbr
 ┃ ┣ 📜~State_Action_Loose_Probability__8116.twbr
 ┃ ┗ 📜~Tableu-nonstationary__34204.twbr
 ┣ 📂data
 ┃ ┣ 📂NDSI-2D : All users processed interaction logs: Use Rainbow extension for easier exploration
 ┃ ┃ ┣ 📜Annotated-Probability Distribution -2D.twb
 ┃ ┃ ┣ 📜Probability Distribution.twb
 ┃ ┃ ┣ 📜Sensemaking-region.twb
 ┃ ┃ ┣ 📜U_1.csv
 ┃ ┃ ┣ 📜U_10.csv
 ┃ ┃ ┣ 📜U_11.csv
 ┃ ┃ ┣ 📜U_12.csv
 ┃ ┃ ┣ 📜U_13.csv
 ┃ ┃ ┣ 📜U_14.csv
 ┃ ┃ ┣ 📜U_15.csv
 ┃ ┃ ┣ 📜U_16.csv
 ┃ ┃ ┣ 📜U_17.csv
 ┃ ┃ ┣ 📜U_18.csv
 ┃ ┃ ┣ 📜U_19.csv
 ┃ ┃ ┣ 📜U_2.csv
 ┃ ┃ ┣ 📜U_20.csv
 ┃ ┃ ┣ 📜U_3.csv
 ┃ ┃ ┣ 📜U_4.csv
 ┃ ┃ ┣ 📜U_5.csv
 ┃ ┃ ┣ 📜U_6.csv
 ┃ ┃ ┣ 📜U_7.csv
 ┃ ┃ ┣ 📜U_8.csv
 ┃ ┃ ┣ 📜U_9.csv
 ┗ 📜sampled-hyperparameters-config.json
