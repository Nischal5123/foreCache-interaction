import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentAnalyzer:
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task
        self.data_folder = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
        self.result_files = [f for f in os.listdir(self.data_folder) if f.endswith('predictions_data.csv')]
        self.results = []

    def load_results(self):
        for file in self.result_files:
            df = pd.read_csv(os.path.join(self.data_folder, file))
            df['algorithm'] = os.path.splitext(file)[0][0]  # Add algorithm name to the dataframe
            self.results.append(df)

    def plot_algorithm_distributions(self):
        all_data = []

        for df in self.results:

            alg = df['algorithm'].iloc[0]
            for _, row in df.iterrows():
                user = row['user']
                threshold = row['threshold']
                y_pred = row['y_pred']
                y_true = row['y_true']
                all_data.append([user, alg, threshold, y_pred, y_true])

        df_all = pd.DataFrame(all_data, columns=['user', 'algorithm', 'threshold', 'y_pred', 'y_true'])
        df_all.to_csv(f"Experiments_Folder/VizRec/{self.dataset}/{self.task}/data/all_data.csv", index=False)

        pred_values = ['same', 'modify-1', 'modify-2', 'modify-3']
        algorithms = df_all['algorithm'].unique()

        plt.figure(figsize=(16, 10))

        # Plot each prediction type separately
        for pred_value in pred_values:
            df_pred_value = df_all[df_all['y_pred'] == pred_value]
            df_grouped = df_pred_value.groupby(['threshold', 'algorithm']).size().reset_index(name='count')

            sns.barplot(data=df_grouped, x='threshold', y='count', hue='algorithm', palette='Set2', ci=None, alpha=0.7)

        plt.title(f'Prediction Counts per Threshold by Algorithm and Prediction Type')
        plt.xlabel('Threshold')
        plt.ylabel('Count')

        # Manually handle the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.75, 1])



# Example usage:
if __name__ == '__main__':
    analyzer = ExperimentAnalyzer(dataset='movies', task='p4')
    analyzer.load_results()
    analyzer.plot_algorithm_distributions()

    # analyzer.compare_algorithms_qualitatively()
    # analyzer.analyze_differences()


    # def compare_algorithms_qualitatively(self):
    #     for df in self.results:
    #         algo = df['user'].iloc[0]
    #         max_acc = df['y_true'].sum()
    #         print(f"For Algorithm {algo}:")
    #         print(f"Maximum Accuracy: {max_acc}")
    #         print(f"Average Accuracy: {df['y_true'].mean()}")
    #         print()
    #
    # def analyze_differences(self):
    #     algo_data = {}
    #     for df in self.results:
    #         algo = df['user'].iloc[0]
    #         algo_data[algo] = df
    #
    #     for metric in ['y_true']:
    #         print(f"Comparison based on {metric}:")
    #         print("----------------------------------")
    #         for i in range(len(self.results)):
    #             for j in range(i + 1, len(self.results)):
    #                 algo1 = self.results[i]['user'].iloc[0]
    #                 algo2 = self.results[j]['user'].iloc[0]
    #                 mean1 = self.results[i][metric].mean()
    #                 mean2 = self.results[j][metric].mean()
    #                 if mean1 > mean2:
    #                     print(f"{algo1} performs better than {algo2} in terms of {metric}")
    #                 elif mean2 > mean1:
    #                     print(f"{algo2} performs better than {algo1} in terms of {metric}")
    #                 else:
    #                     print(f"{algo1} and {algo2} have similar performance in terms of {metric}")
    #         print()