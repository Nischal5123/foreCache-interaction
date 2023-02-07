import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the data and convert the ROI_Cycle column to a list of integers
df = pd.read_csv("data/NDSI-2D\\region-manwhitneyresults.csv")
df["ROI_Cycle"] = df["ROI_Cycle"].apply(lambda x: [int(y) for y in x[1:-1].split(", ")])

df["User"] = df["User"].apply(lambda x: x[7:12])

df=df.sort_values('User',ascending=True)
# # # Filter the dataframe to include only the first 10 users
# df = df[df["User"].isin(df["User"].unique()[:10])]
# Add a new column that calculates the absolute difference between the elements in the ROI_Cycle column
df["ROI_Cycle_Difference"] = df["ROI_Cycle"].apply(lambda x: abs(x[0] - x[1]))


# # Filter the dataframe to include only next to each other comparisons
# df = df[df["ROI_Cycle_Difference"]==1]

# Group the data by the User, State columns and count the number of True and False values in the Result column for each group
grouped_df = df.groupby(["User", "State"])["Result"].value_counts(normalize=True).reset_index(name="Percentage")

grouped_df.to_csv("data/NDSI-2D\\region-aggregated-roi-manwhitneyresults.csv")

print(grouped_df)

# Create a faceted barplot of the grouped data
g = sns.FacetGrid(grouped_df, col="User", row="State")
g.map(sns.barplot,"Percentage","Result",palette=["red", "blue","orange"], hue_order=["True", "False", "NED"])


# Add a legend to the plot
g.add_legend()

# Show the plot
plt.show()

