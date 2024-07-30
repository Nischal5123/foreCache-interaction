import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


class ExperimentAnalyzer:
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task
        self.data_folder = f"Experiments_Folder/VizRec/{dataset}/{task}/data"
        self.result_files = [f for f in os.listdir(self.data_folder) if f.endswith('predictions_data.csv')]
        self.results = []
        self.all_data = pd.DataFrame()

    def load_results(self):
        for file in self.result_files:
            df = pd.read_csv(os.path.join(self.data_folder, file))
            df['algorithm'] = os.path.splitext(file)[0][0]  # Add algorithm name to the dataframe
            self.results.append(df)
        # Concatenate all the dataframes
        self.all_data = pd.concat(self.results)
        # Save to a single CSV file
        self.all_data.to_csv(f"Experiments_Folder/VizRec/{self.dataset}/{self.task}/data/interaction_all_data.csv",
                             index=False)

    def plot_ytrue_vs_ypred(self, algorithm, user, threshold):
        # Filter data based on algorithm, user, and threshold
        subset = self.all_data[(self.all_data['algorithm'] == algorithm) &
                               (self.all_data['user'] == user) &
                               (self.all_data['threshold'] == threshold)]

        # Determine matches and non-matches
        subset['match'] = subset['y_true'] == subset['y_pred']

        # Create two traces: one for matches and one for non-matches
        match_trace = go.Scatter(
            x=subset[subset['match']]['interaction_point'],
            y=subset[subset['match']]['y_true'],
            mode='markers',
            marker=dict(symbol='square', color='darkred', size=10, opacity=0.8),
            name='Match'
        )

        non_match_trace = go.Scatter(
            x=subset[~subset['match']]['interaction_point'],
            y=subset[~subset['match']]['y_true'],
            mode='markers',
            marker=dict(symbol='circle', color='blue', size=10, opacity=0.8),
            name='Non-match'
        )

        # Create the figure
        fig = go.Figure()

        fig.add_trace(match_trace)
        fig.add_trace(non_match_trace)

        fig.update_layout(
            title=f'y_true vs y_pred for {algorithm} - User: {user}, Threshold: {threshold}',
            xaxis_title='Interaction Point',
            yaxis_title='y_true',
            legend_title='Legend'
        )

        fig.show()

    def plot_algorithm_performance(self):
        # Subset the data
        data_subset = self.all_data.copy()

        # Create a figure with subplots
        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=['same', 'modify-1', 'modify-2', 'modify-3'])

        y_true_values = ['same', 'modify-1', 'modify-2', 'modify-3']
        row_col_map = {
            'same': (1, 1),
            'modify-1': (1, 2),
            'modify-2': (2, 1),
            'modify-3': (2, 2)
        }

        for y_true in y_true_values:
            row, col = row_col_map[y_true]
            subset = data_subset[data_subset['y_true'] == y_true]

            # Group by algorithm and count matches
            matches = subset[subset['y_true'] == subset['y_pred']].groupby('algorithm').size().reset_index(
                name='match_count')

            # Create bar trace for matches
            bar_trace = go.Bar(
                x=matches['algorithm'],
                y=matches['match_count'],
                name=f'Matches for {y_true}',
                marker=dict(color='darkred')
            )

            fig.add_trace(bar_trace, row=row, col=col)

        fig.update_layout(
            title_text="Algorithm Performance for Each y_true Value",
            height=800,
            showlegend=False
        )

        fig.show()


# Example usage:
if __name__ == '__main__':
    analyzer = ExperimentAnalyzer(dataset='movies', task='p4')
    analyzer.load_results()
    analyzer.plot_algorithm_performance()
