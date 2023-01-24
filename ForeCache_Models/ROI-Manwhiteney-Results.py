import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the data and convert the ROI_Cycle column to a list of integers
df = pd.read_csv("data/NDSI-2D\\roi-manwhitneyresults.csv")
df["ROI_Cycle"] = df["ROI_Cycle"].apply(lambda x: [int(y) for y in x[1:-1].split(", ")])

# # Filter the dataframe to exclude rows with an NED value in the Result column
# df = df[df["Result"] != "NED"]

# Add a new column that calculates the absolute difference between the elements in the ROI_Cycle column
df["ROI_Cycle_Difference"] = df["ROI_Cycle"].apply(lambda x: abs(x[0] - x[1]))
df["User"] = df["User"].apply(lambda x: x[7:12])




# Filter the dataframe to include only the first 10 users
df = df[df["User"].isin(df["User"].unique()[:10])]

# Group the data by the User, State, and ROI_Cycle_Difference columns and count the number of True and False values in the Result column for each group
grouped_df = df.groupby(["User", "State", "ROI_Cycle_Difference"])["Result"].value_counts(normalize=True).reset_index(name="Count")


# Create a faceted barplot of the grouped data
g = sns.FacetGrid(grouped_df, col="User", row="State")
g.map(sns.barplot, "ROI_Cycle_Difference", "Count", "Result",palette=["red", "blue","orange"], hue_order=["True", "False", "NED"])


# Add a legend to the plot
g.add_legend()

# Show the plot
plt.show()

