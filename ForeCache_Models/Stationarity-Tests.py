import pandas as pd
import numpy as np
import environment2
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

class StationarityTests:
    def __init__(self, significance=.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None
        self.trend= None
    def kpss_test(self, timeseries):
        kpsstest = kpss(timeseries, regression="c", nlags="auto")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        return kpss_output


def get_probabilities(dataframe,phase):
    dataframe['high-level-action']=np.zeros(len(dataframe),str)
    probabilities=[]
    map=dict()
    # print(dataframe.head(5))
    states=['Sensemaking','Foraging','Navigation']
    actions=['change','same']
    for state in states:
      for action in actions:
        map[state+action]=0


    current_state=dataframe['State'][0]

    for i in range(1,len(dataframe)):
      if dataframe["State"][i]== current_state:
        dataframe["high-level-action"][i-1]="same"
      else:
        dataframe["high-level-action"][i-1]="change"
      current_state=dataframe["State"][i]
    dataframe["high-level-action"][len(dataframe)-1]="same"

    probs=1/2
    for i in range(len(dataframe)):
      if phase == 'Navigation' and dataframe['State'][i]=="Navigation":
        map[str(dataframe['State'][i]+dataframe['high-level-action'][i])] += 1
        probs = (map['Navigationsame'])/(map['Navigationchange']  + map['Navigationsame'])

      elif phase == 'Foraging' and dataframe['State'][i]=="Foraging":
        map[str(dataframe['State'][i]+dataframe['high-level-action'][i])] += 1
        probs = (map['Foragingsame'])/(map['Foragingchange'] + map['Foragingsame'])

      elif  phase == 'Sensemaking' and dataframe['State'][i]=="Sensemaking":
        map[str(dataframe['State'][i]+dataframe['high-level-action'][i])] += 1
        probs = (map['Sensemakingsame'])/(map['Sensemakingchange'] + map['Sensemakingsame'])
      probabilities.append(probs)
    dataframe['probabilities']=probabilities
    return dataframe, map


if __name__ == '__main__':
    env = environment2.environment2()
    user_list_2D = env.user_list_2D
    # Create an empty list to store DataFrames
    dfs = []
    for i, u in enumerate(user_list_2D):
        df = pd.read_csv(u)
        u = u.lstrip('data/NDSI-2D\\')
        u = u.rstrip('.csv')

        # Create a list of DataFrames for each state
        state_dfs = []
        for state in ['Foraging', 'Navigation', 'Sensemaking']:
            df_subset, _ = get_probabilities(df, state)
            assert (len(df_subset['probabilities']) == len(df))

            try:
                if not df_subset['probabilities'].empty and not df_subset['probabilities'].isnull().any():
                    kpss_stat, kpss_p_value, _, _ = kpss(df_subset['probabilities'])

                    # Create a DataFrame for the current state
                    state_df = pd.DataFrame({'User': [u], 'State': [state], 'KPSS_Result': [kpss_p_value < 0.05], 'KPSS_P':[kpss_p_value], 'KPSS_stats':[kpss_stat]})
                    state_dfs.append(state_df)
                else:
                    print(f"User {u}, State {state}: Empty or contains null values in 'probabilities'")
            except Exception as e:
                print(f"Error processing User {u}, State {state}: {e}")
                # Create a DataFrame for the current state
                state_df = pd.DataFrame({'User': [u], 'State': [state], 'KPSS_Result': False, 'KPSS_P':[1], 'KPSS_stats':[-1000]})
                state_dfs.append(state_df)

        # Concatenate DataFrames for each state into a single DataFrame for the user
        user_df = pd.concat(state_dfs, ignore_index=True)
        dfs.append(user_df)

    # Concatenate DataFrames for each user into the final DataFrame
    results_df = pd.concat(dfs, ignore_index=True)

    # Save results to CSV
    results_df.to_csv('data/NDSI-2D/kpssresults.csv')

