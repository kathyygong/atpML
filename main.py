import numpy as np
import pandas as pd
import os
import gc
import sys
import warnings
warnings.filterwarnings('ignore')
import h2o
from h2o.automl import H2OAutoML
from datetime import date, timedelta

pd.options.display.max_columns = 999

# load mens match csv
df_atp = pd.read_csv("data.csv")

# include hard and indoor hard only
df_atp = df_atp.loc[df_atp.Court_Surface.isin(['Hard','Indoor Hard'])]

# exclude qualifying rounds
df_atp = df_atp.loc[df_atp.Round_Description != 'Qualifying']

# store shape of data for reference check later
atp_shape = df_atp.shape

numeric_columns = ['Winner_Rank', 'Loser_Rank', 'Retirement_Ind',
                   'Winner_Sets_Won', 'Winner_Games_Won', 'Winner_Aces',
                   'Winner_DoubleFaults', 'Winner_FirstServes_Won',
                   'Winner_FirstServes_In', 'Winner_SecondServes_Won',
                   'Winner_SecondServes_In', 'Winner_BreakPoints_Won',
                   'Winner_BreakPoints', 'Winner_ReturnPoints_Won',
                   'Winner_ReturnPoints_Faced', 'Winner_TotalPoints_Won', 'Loser_Sets_Won',
                   'Loser_Games_Won', 'Loser_Aces', 'Loser_DoubleFaults',
                   'Loser_FirstServes_Won', 'Loser_FirstServes_In',
                   'Loser_SecondServes_Won', 'Loser_SecondServes_In',
                   'Loser_BreakPoints_Won', 'Loser_BreakPoints', 'Loser_ReturnPoints_Won',
                   'Loser_ReturnPoints_Faced', 'Loser_TotalPoints_Won']
''

text_columns = ['Winner', 'Loser',  'Tournament', 'Court_Surface','Round_Description']

date_columns = ['Tournament_Date']

# set **erros** to coerce so any non-numerical values (text,special characters) will return an NA
df_atp[numeric_columns] = df_atp[numeric_columns].apply(pd.to_numeric,errors = 'coerce')


df_atp[date_columns] = df_atp[date_columns].apply(pd.to_datetime)

# before splitting data frame into winner/loser, create feature that captures total number of games match takes
# do this before split or will lose this information
df_atp['Total_Games'] = df_atp.Winner_Games_Won + df_atp.Loser_Games_Won


# get column names for the winner/loser stats
winner_cols = [col for col in df_atp.columns if col.startswith('Winner')]
loser_cols = [col for col in df_atp.columns if col.startswith('Loser')]

# create winner/loser data frames to store winner/loser stats
# In addition to the winner and loser columns, we are adding common columns as well (e.g. tournamnt dates)
common_cols = ['Total_Games','Tournament','Tournament_Date', 'Court_Surface','Round_Description']
df_winner_atp = df_atp[winner_cols + common_cols]
df_loser_atp = df_atp[loser_cols + common_cols]

# create new column to show whether player has won or not
df_winner_atp["won"] = 1
df_loser_atp["won"] = 0


# rename columns for winner/loser data frames to append later on.
# rename Winner_ / Loser_ columns to Player_
new_column_names = [col.replace('Winner','Player') for col in winner_cols]
df_winner_atp.columns = new_column_names + common_cols + ['won']
df_loser_atp.columns  = df_winner_atp.columns

# append winner/loser data frames
df_long_atp= df_winner_atp.append(df_loser_atp)

# define function to apply it to both atp and wta data frames
def get_new_features(df):
# Input:
# df: data frame to get the data from
# Return: the df with the new features
    # Point Win ratio when serving
    df['Player_Serve_Win_Ratio'] = (df.Player_FirstServes_Won + df.Player_SecondServes_Won - df.Player_DoubleFaults) \
                                  /(df.Player_FirstServes_In + df.Player_SecondServes_In + df.Player_DoubleFaults)
    # Point win ratio when returning
    df['Player_Return_Win_Ratio'] = df.Player_ReturnPoints_Won / df.Player_ReturnPoints_Faced
    # Breakpoints per receiving game
    df['Player_BreakPoints_Per_Return_Game'] = df.Player_BreakPoints/df.Total_Games
    df['Player_Game_Win_Percentage'] = df.Player_Games_Won/df.Total_Games

    return df

# apply the function just created to long data frames
df_long_atp = get_new_features(df_long_atp)

# long table should have exactly twice of the rows of the original data
assert df_long_atp.shape[0] == atp_shape[0]*2

# tournaments to use for training
tournaments = ['U.S. Open, New York','Australian Open, Melbourne']

# store dates for loops
tournament_dates_atp = df_atp.loc[df_atp.Tournament.isin(tournaments)].groupby(['Tournament','Tournament_Date']) \
.size().reset_index()[['Tournament','Tournament_Date']]

# add one more date for final prediction
tournament_dates_atp.loc[-1] = ['Australian Open, Melbourne',pd.to_datetime('2019-01-15')]

# define a function to calculate the rolling averages
def get_rolling_features (df, date_df=None,rolling_cols = None, last_cols= None):

    # Input:
#     df: data frame to get the data from
#     date_df: data frame that has the start dates for each tournament
#     rolling_cols: columns to get the rolling averages
#     last_cols: columns to get the last value (most recent)

     # Return: the df with the new features


    # sort data by player and dates so most recent matches are at bottom
    df = df.sort_values(['Player','Tournament_Date','Tournament'], ascending=True)

    # For each tournament, get the rolling averages of that player's past matches before the tournament start date
    for index, tournament_date in enumerate(date_df.Tournament_Date):
        # create a temp df to store the interim results
        df_temp = df.loc[df.Tournament_Date < tournament_date]

        # for ranks, we only take the last one. (comment this out if want to take avg of rank)
        df_temp_last = df_temp.groupby('Player')[last_cols].last().reset_index()

        # take the most recent 15 matches for the rolling average
        df_temp = df_temp.groupby('Player')[rolling_cols].rolling(15, min_periods=1).mean().reset_index()
        df_temp = df_temp.groupby('Player').tail(1) # take the last row of the above

        df_temp= df_temp.merge(df_temp_last, on='Player', how='left')

        if index ==0:
            df_result = df_temp
            df_result['tournament_date_index'] = tournament_date # so we know which tournament this feature is for
        else:
            df_temp['tournament_date_index'] = tournament_date
            df_result = df_result.append(df_temp)

    df_result.drop('level_1', axis=1,inplace=True)

    return df_result

# columns we are applying the rolling averages on
rolling_cols = ['Player_Serve_Win_Ratio', 'Player_Return_Win_Ratio',
               'Player_BreakPoints_Per_Return_Game', 'Player_Game_Win_Percentage']

# columns we are taking the most recent values on
# For the player rank, we think we can just use the latest rank (before the tournament starts)
# as it should refect the most recent performance of the player
last_cols = ['Player_Rank']


# Apply the rolling average function to the long data frames (it will take a few mins to run)
df_rolling_atp = get_rolling_features (df_long_atp, tournament_dates_atp, rolling_cols, last_cols= last_cols )

# Randomise the match_wide dataset so the first player is not always the winner
# set a seed so the random number is reproducable
np.random.seed(2)

# randomise a number 0/1 with 50% chance each
# if 0 then take the winner, 1 then take loser
df_atp['random_number'] = np.random.randint(2, size=len(df_atp))
df_atp['randomised_player_1'] = np.where(df_atp['random_number']==0,df_atp['Winner'],df_atp['Loser'])
df_atp['randomised_player_2'] = np.where(df_atp['random_number']==0,df_atp['Loser'],df_atp['Winner'])

# set the target (win/loss) based on the new randomise number
df_atp['player_1_win'] = np.where(df_atp['random_number']==0,1,0)

print ('After shuffling, the win rate for player 1 for the mens is {}%'.format(df_atp['player_1_win'].mean()*100))

# To get data frames ready for model training, exclude other tournaments from data now because have gotten the rolling averages from them and
# for training, only need US and Australian Open matches
df_atp = df_atp.loc[df_atp.Tournament.isin(tournaments)]

# remove other stats columns (using the differences)
cols_to_keep = ['Winner','Loser','Tournament','Tournament_Date',
                    'player_1_win','randomised_player_1',
                    'randomised_player_2']

df_atp = df_atp[cols_to_keep]


# joining the rolling average data frames to the individual matches.
# do it twice - one for player 1 and one for player 2

# Get the rolling features for player 1
df_atp = df_atp.merge(df_rolling_atp, how='left',
                      left_on = ['randomised_player_1','Tournament_Date'],
                      right_on = ['Player','tournament_date_index'],
                      validate ='m:1')

# Get the rolling features for player 2
# use '_p1' to denote player 1 and '_p2' for player 2
df_atp = df_atp.merge(df_rolling_atp, how='left',
                      left_on = ['randomised_player_2','Tournament_Date'],
                      right_on = ['Player','tournament_date_index'],
                      validate ='m:1',
                      suffixes=('_p1','_p2'))

def get_player_difference(df, diff_cols=None):
# Input:
# df: data frame to get the data from
# diff_cols: columns we take the difference on. For example is diff_cols = win rate. This function will calculate the
# difference of the win rates between player 1 and player 2

    # Return: the df with the new features
    p1_cols = [i + '_p1' for i in diff_cols]  # column names for player 1 stats
    p2_cols = [i + '_p2' for i in diff_cols]  # column names for player 2 stats

    # For any missing values, we will fill them by zeros except the ranking where we will use 999
    df['Player_Rank_p1'] = df['Player_Rank_p1'].fillna(999)
    df[p1_cols] = df[p1_cols].fillna(0)

    df['Player_Rank_p2'] = df['Player_Rank_p2'].fillna(999)
    df[p2_cols] = df[p2_cols].fillna(0)

    new_column_name = [i + '_diff' for i in diff_cols]

    # Take the difference
    df_p1 = df[p1_cols]
    df_p2 = df[p2_cols]

    df_p1.columns = new_column_name
    df_p2.columns = new_column_name

    df_diff = df_p1 - df_p2
    df_diff.columns = new_column_name

    # drop the p1 and p2 columns because We have the differences now
    df.drop(p1_cols + p2_cols, axis=1, inplace=True)

    # Concat the df_diff and raw_df
    df = pd.concat([df, df_diff], axis=1)

    return df, new_column_name

diff_cols = ['Player_Serve_Win_Ratio',
            'Player_Return_Win_Ratio',
            'Player_BreakPoints_Per_Return_Game',
            'Player_Game_Win_Percentage','Player_Rank']

# Apply the function and get the difference between player 1 and 2
df_atp,_ = get_player_difference(df_atp,diff_cols=diff_cols)

# Make a copy of the data frames in case we need to come back to check the values
df_atp_final = df_atp.copy()

# MODELING
df_train_atp = df_atp_final.loc[(df_atp_final.Tournament_Date != '2018-01-15') # excluding Aus Open 2018, and
                                & (df_atp_final.Tournament_Date > '2012-01-16')] # excluding first year
df_valid_atp = df_atp_final.loc[df_atp_final.Tournament_Date == '2018-01-15'] # Australian Open 2018 only

# target variable
target= 'player_1_win'

# features being fed into the models
feats = ['Player_Serve_Win_Ratio_diff',
         'Player_Return_Win_Ratio_diff',
         'Player_BreakPoints_Per_Return_Game_diff',
         'Player_Game_Win_Percentage_diff',
         'Player_Rank_diff']

print(feats)

# H20 Model for ATP
h2o.init()

# Convert to an h2o frame
df_train_atp_h2o = h2o.H2OFrame(df_train_atp)
df_valid_atp_h2o = h2o.H2OFrame(df_valid_atp)


# For binary classification, response should be a factor
df_train_atp_h2o[target] = df_train_atp_h2o[target].asfactor()
df_valid_atp_h2o[target] = df_valid_atp_h2o[target].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml_atp = h2o.automl.H2OAutoML(max_runtime_secs=300,
                           max_models=100,
                           stopping_metric='logloss',
                           sort_metric='logloss',
                           balance_classes=True,
                           seed=183
                          )
aml_atp.train(x=feats, y=target, training_frame=df_train_atp_h2o,validation_frame=df_valid_atp_h2o)

# View the AutoML Leaderboard
lb = aml_atp.leaderboard
lb.head()

df_predict_atp = pd.read_csv("data/men_dummy_submission_file.csv")

# Before we join the features by the names and the dates, we need to convert any non-english characters to english first
translationTable = str.maketrans("éàèùâêîôûçñá", "eaeuaeioucna")


df_predict_atp['player_1'] = df_predict_atp.player_1.apply(lambda x: x.translate(translationTable))
df_predict_atp['player_2'] = df_predict_atp.player_2.apply(lambda x: x.translate(translationTable))

# Also we need to convert the names into lower cases
df_predict_atp['player_1'] =df_predict_atp['player_1'].str.lower()
df_predict_atp['player_2'] =df_predict_atp['player_2'].str.lower()

df_rolling_atp['Player'] = df_rolling_atp['Player'].str.lower()

# some players have slightly difference names in the submission data and the match data - editing them manually
df_predict_atp.loc[df_predict_atp.player_1=='jaume munar','player_1'] = 'jaume antoni munar clar'
df_predict_atp.loc[df_predict_atp.player_2=='jaume munar','player_2'] = 'jaume antoni munar clar'

# create and tournament date column and set it to 2019 so we can join the lastest features
df_predict_atp['Tournament_Date'] = pd.to_datetime('2019-01-15')

# Get the rolling features for player 1
df_predict_atp = df_predict_atp.merge(df_rolling_atp, how='left',
                     left_on = ['player_1','Tournament_Date'],
                     right_on = ['Player','tournament_date_index'],validate ='m:1')

# Get the rolling features for player 2
# For duplicate columns, use '_p1' to denote player 1 and '_p2' for player 2
df_predict_atp = df_predict_atp.merge(df_rolling_atp, how='left',
                     left_on = ['player_2','Tournament_Date'],
                     right_on = ['Player','tournament_date_index'],validate ='m:1',suffixes=('_p1','_p2'))

# Apply the function and get the difference between player 1 and 2
df_predict_atp,_ = get_player_difference(df_predict_atp,diff_cols=diff_cols)

df_predict_atp_h2o = h2o.H2OFrame(df_predict_atp[feats])

atp_preds = aml_atp.predict(df_predict_atp_h2o)['p1'].as_data_frame()

df_predict_atp['player_1_win_probability'] = atp_preds

atp_submission.to_csv('submission/atp_submission_python.csv',index=False)

