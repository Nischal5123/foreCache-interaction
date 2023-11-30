import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ChangePointDetection:
    def __init__(self, filename, threshold=0.1, LA=5):
        self.filename = filename
        self.threshold = threshold
        self.LA = LA

    def detect_change_points(self):
        # Load the data from the CSV file
        df = pd.read_csv(self.filename)

        # Extract the "HighLevelStateActionProbab" column as a NumPy array
        data = df['HighLevelStateActionProbab'].values

        # Initialize variables for CUSUM algorithm
        cusum = np.zeros_like(data)
        avg = np.mean(data)
        change_points = []

        # Initialize a list to store "IsChangePoint" values
        is_change_point = ['No'] * len(df)  # Initially, set all values to 'No'

        for i in range(1, len(data)):
            cusum[i] = max(0, cusum[i - 1] + data[i] - avg - self.threshold)

            if cusum[i] > 0 and i + self.LA < len(data) and all(data[i + 1:i + self.LA + 1] <= avg + self.threshold):
                change_points.append(i)
                is_change_point[i] = 'Yes'

        # Add the "IsChangePoint" and "CUSUM" columns to the DataFrame
        df['IsChangePoint'] = is_change_point
        df['CUSUM'] = cusum

        return df, change_points

    def plot_cusum(self, df):
        # Extract the CUSUM values
        cusum = df['CUSUM'].values

        # Plot the CUSUM values
        plt.figure(figsize=(10, 5))
        plt.plot(cusum, marker='o', linestyle='-', color='b', label='CUSUM')
        plt.axhline(0, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Data Points')
        plt.ylabel('CUSUM Value')
        plt.legend()
        plt.title('Change Point Detection using CUSUM with Look Ahead')
        plt.grid(True)
        plt.show()


# Usage example:
if __name__ == "__main__":
    # Initialize the ChangePointDetection class with a filename
    cpd = ChangePointDetection('/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/NDSI-2D/U_1.csv')  # Replace 'your_dataset.csv' with your actual CSV file

    # Perform change point detection and get the DataFrame with the "IsChangePoint" column
    df, change_points = cpd.detect_change_points()
    print("Change points detected at data points:", change_points)

    # Plot the CUSUM values
    cpd.plot_cusum(df)

    # Display the updated DataFrame
    print(df)
