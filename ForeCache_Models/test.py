import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class ExperimentAnalyzer:
    def __init__(self, datasets, tasks):
        """Initialize the ExperimentAnalyzer with datasets and tasks."""
        self.datasets = datasets
        self.tasks = tasks
        self.results = []
        self.all_data = pd.DataFrame()

    def load_results(self):
        """Load the results from all datasets and tasks."""
        for dataset in self.datasets:
            for task in self.tasks:
                data_folder = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
                result_files = [f for f in os.listdir(data_folder) if f.endswith('predictions_data.csv')]
                for file in result_files:
                    df = pd.read_csv(os.path.join(data_folder, file))
                    df['algorithm'] = os.path.splitext(file)[0][0]  # Add algorithm name to the dataframe
                    df['task'] = task  # Add task name to the dataframe
                    df['dataset'] = dataset  # Add dataset name to the dataframe
                    self.results.append(df)
        # Concatenate all the dataframes
        self.all_data = pd.concat(self.results)
        print(f"Loaded {len(self.all_data)} results across {len(self.datasets)} datasets and {len(self.tasks)} tasks.")
        print(self.all_data.head())

    def encode_predictions(self):
        """Encode y_true and y_pred as numerical values (0, 1, 2, 3)."""
        encoding = {'same': 0, 'modify-1': 1, 'modify-2': 2, 'modify-3': 3}
        self.all_data['y_true_encoded'] = self.all_data['y_true'].map(encoding)
        self.all_data['y_pred_encoded'] = self.all_data['y_pred'].map(encoding)

    def count_transitions(self):
        """Calculate transition matrices for algorithms and true labels."""
        self.encode_predictions()
        algorithms = self.all_data['algorithm'].unique()
        transition_matrix = {}

        # Define the prediction types
        predictions = [0, 1, 2, 3]

        for algorithm in algorithms:
            matrix = pd.DataFrame(0, index=predictions, columns=predictions)
            for task in self.tasks:
                subset = self.all_data[(self.all_data['algorithm'] == algorithm) & (self.all_data['task'] == task)]
                for i in range(len(subset) - 1):
                    current = subset.iloc[i]['y_pred_encoded']
                    next_pred = subset.iloc[i + 1]['y_pred_encoded']
                    matrix.at[current, next_pred] += 1
            transition_matrix[algorithm] = matrix

        # Calculate the transition matrix for y_true within the same task
        y_true_matrix = pd.DataFrame(0, index=predictions, columns=predictions)
        for task in self.tasks:
            subset = self.all_data[self.all_data['task'] == task]
            for i in range(len(subset) - 1):
                current = subset.iloc[i]['y_true_encoded']
                next_true = subset.iloc[i + 1]['y_true_encoded']
                y_true_matrix.at[current, next_true] += 1 / len(self.tasks)

        transition_matrix['True'] = y_true_matrix

        return transition_matrix

    def calculate_match_percentage(self):
        """Calculate the match percentage between y_true and y_pred."""
        self.encode_predictions()
        algorithms = self.all_data['algorithm'].unique()
        match_percentages = {}

        for algorithm in algorithms:
            subset = self.all_data[self.all_data['algorithm'] == algorithm]
            matches = subset[subset['y_true_encoded'] == subset['y_pred_encoded']].shape[0]
            total = subset.shape[0]
            match_percentages[algorithm] = (matches / total) * 100

        return match_percentages

    def plot_action_distribution(self):
        """Plot the distribution of actions for y_true and y_pred."""
        self.encode_predictions()
        algorithms = self.all_data['algorithm'].unique()

        fig, axs = plt.subplots(2, len(algorithms), figsize=(5 * len(algorithms), 8), sharey=True)

        for i, algorithm in enumerate(algorithms):
            # Plot y_true distribution
            y_true_counts = self.all_data[self.all_data['algorithm'] == algorithm]['y_true_encoded'].value_counts().sort_index()
            y_true_total = y_true_counts.sum()
            axs[0, i].bar(y_true_counts.index, y_true_counts.values, color='skyblue')
            axs[0, i].set_title(f'{algorithm} - y_true')
            axs[0, i].set_xticks([0, 1, 2, 3])
            axs[0, i].set_xticklabels(['same', 'modify-1', 'modify-2', 'modify-3'])

            # Add percentage labels on the bars
            for idx, count in enumerate(y_true_counts):
                percentage = count / y_true_total * 100
                axs[0, i].text(idx, count, f'{percentage:.2f}%', ha='center', va='bottom')

            # Plot y_pred distribution
            y_pred_counts = self.all_data[self.all_data['algorithm'] == algorithm]['y_pred_encoded'].value_counts().sort_index()
            y_pred_total = y_pred_counts.sum()
            axs[1, i].bar(y_pred_counts.index, y_pred_counts.values, color='lightgreen')
            axs[1, i].set_title(f'{algorithm} - y_pred')
            axs[1, i].set_xticks([0, 1, 2, 3])
            axs[1, i].set_xticklabels(['same', 'modify-1', 'modify-2', 'modify-3'])

            # Add percentage labels on the bars
            for idx, count in enumerate(y_pred_counts):
                percentage = count / y_pred_total * 100
                axs[1, i].text(idx, count, f'{percentage:.2f}%', ha='center', va='bottom')

        # Add a common y-axis label
        axs[0, 0].set_ylabel('Count')
        axs[1, 0].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

    def calculate_metrics(self):
        """Calculate precision, recall, F1 score, false positive rate, and accuracy per action for each algorithm."""
        self.encode_predictions()
        algorithms = self.all_data['algorithm'].unique()
        metrics = {}

        for algorithm in algorithms:
            subset = self.all_data[self.all_data['algorithm'] == algorithm]
            y_true = subset['y_true_encoded']
            y_pred = subset['y_pred_encoded']

            # Calculate metrics
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

            # Calculate false positive rate for each class
            fp = cm.sum(axis=0) - np.diag(cm)
            tn = cm.sum() - (fp + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
            fp_rate = fp / (fp + tn + 1e-6)  # Avoid division by zero

            # Calculate accuracy per action
            accuracy_per_action = calculate_accuracy(y_true, y_pred)

            metrics[algorithm] = {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'False Positive Rate (avg)': fp_rate.mean(),
                'Accuracy per Action': accuracy_per_action
            }

        return metrics

    def print_metrics_table(self):
        """Print a table of calculated metrics for each algorithm, including accuracy per action."""
        metrics = self.calculate_metrics()

        # Print table header
        print(
            f"{'Algorithm':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'FPR (avg)':<15} {'Accuracy per Action':<30}")
        print("=" * 90)

        for algorithm, values in metrics.items():
            accuracy_str = ', '.join(
                [f'Action {action}: {accuracy:.4f}' for action, accuracy in values['Accuracy per Action'].items()])
            print(f"{algorithm:<15} {values['Precision']:<10.4f} {values['Recall']:<10.4f} "
                  f"{values['F1 Score']:<10.4f} {values['False Positive Rate (avg)']:<15.4f} {accuracy_str}")


def create_bubble_plot(transition_matrices, match_percentages):
    """Create a bubble plot for transition matrices."""
    subplot_titles = [
        f'Algorithm Bayesian (Match: {match_percentages.get("B", 0):.2f}%)',
        f'Algorithm Greedy (Match: {match_percentages.get("G", 0):.2f}%)',
        f'Algorithm QLearning (Match: {match_percentages.get("Q", 0):.2f}%)',
        f'True (Count: {int(transition_matrices["True"].values.sum())})'
    ]
    fig = sp.make_subplots(rows=1, cols=4, subplot_titles=subplot_titles)

    # Find the global maximum count for normalization
    global_max_count = max(matrix.values.max() for matrix in transition_matrices.values())

    for i, (alg, matrix) in enumerate(transition_matrices.items()):
        # Create bubble plot data
        for from_state in matrix.index:
            for to_state in matrix.columns:
                fig.add_trace(go.Scatter(
                    x=[from_state],
                    y=[to_state],
                    mode='markers',
                    marker=dict(
                        size=matrix.at[from_state, to_state] / global_max_count * 100,
                        sizemode='diameter',
                        color='blue',  # Fixed color for all circles
                        showscale=False
                    ),
                    name=f"{from_state} to {to_state}",
                    hovertext=f"Count: {matrix.at[from_state, to_state]}"
                ), row=1, col=i + 1)

    fig.update_layout(
        title='Transition Matrices for Algorithms Bayesian, Greedy, QLearning, and True',
        xaxis_title='From Action',
        yaxis_title='To Action',
        showlegend=False
    )

    fig.show()

#calculate accuracy for each action
def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy for each action."""
    accuracy = {}
    for action in range(4):
        correct = np.sum((y_true == action) & (y_pred == action))
        total = np.sum(y_true == action)
        accuracy[action] = correct / total if total > 0 else 0
    return accuracy


if __name__ == '__main__':
    # Initialize analyzer for the tasks across multiple datasets
    analyzer = ExperimentAnalyzer(datasets=['movies', 'birdstrikes'], tasks=['p1', 'p2', 'p3', 'p4'])

    # Load results for the tasks
    analyzer.load_results()

    # Count transitions
    transitions = analyzer.count_transitions()

    # Calculate match percentages
    match_percentages = analyzer.calculate_match_percentage()

    # Create bubble plot
    create_bubble_plot(transitions, match_percentages)

    # Plot action distribution
    analyzer.plot_action_distribution()

    # Print metrics table, including accuracy per action
    analyzer.print_metrics_table()

