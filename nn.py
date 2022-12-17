import numpy as np
import pandas as pd
import sklearn
import math
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#ok so MLPRegressor is basically just a classifier without an activation function so it gives us n variables as an output, i think we can train these to the stats we want to predict
#also this has a partial_fit() function so if we wanted to go over the y2d results as a separate thing we could do that, the idea being past years have pretty representative power rankings, but in progress years might not be quite perfect

def main(data, training, seed, games_to_predict):
    model = train_test(data, training, seed)
    preds = {}
    for game in games_to_predict:#games to predict, probably just a list of tuples with the names of the teams playing
        preds[game] = model.predict(get_team_stats(game[0], game[1]))
    print(preds)

def train_test_C(data, labels, training, seed):
    X = read_file(data)
    Y = read_file(labels)
    y = pd.DataFrame()
    for row in X.iterrows():
        y = pd.concat([y, pd.concat([Y.loc[Y["Team-year"] == row[1]["T1"]].add_suffix("1").reset_index(drop = True).drop("Team-year1",inplace = False, axis = 1), Y.loc[Y["Team-year"] == row[1]["T2"]].add_suffix("2").reset_index(drop = True).drop("Team-year2",inplace = False, axis = 1)], axis = 1)], axis = 0)
    y = y.reset_index(drop = True)
    #print(y)
    y.to_csv('y.csv')
    X = X.drop("T1", inplace = False, axis = 1)
    X = X.drop("T2", inplace = False, axis = 1)
    X = X.drop("T1 PTS", inplace = False, axis = 1)
    X = X.drop("T2 PTS", inplace = False, axis = 1)
    X = X.drop("T1 FGA", inplace = False, axis = 1)
    X = X.drop("T2 FGA", inplace = False, axis = 1)
    X = X.drop("T1 FGM", inplace = False, axis = 1)
    X = X.drop("T2 FGM", inplace = False, axis = 1)
    X = X.drop("T1 PCT", inplace = False, axis = 1)
    X = X.drop("T2 PCT", inplace = False, axis = 1)
    X = X.drop("DIFF", inplace = False, axis = 1)
    #X = X.drop("RESULT", inplace = False, axis = 1)
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(y, X, random_state=1, train_size = training)
    model = MLPClassifier(random_state=seed, max_iter = 500).fit(X_train,  y_train.values.ravel())
    a = model.predict(X_test)
    #print(a)
    #print(y_test)
    b = model.score(X_test, y_test)
    #print(b)
    return[b,model]


def train_test_R(data, labels, training, seed):
    X = read_file(data)
    Y = read_file(labels)
    y = pd.DataFrame()
    for row in X.iterrows():
        y = pd.concat([y, pd.concat([Y.loc[Y["Team-year"] == row[1]["T1"]].add_suffix("1").reset_index(drop = True).drop("Team-year1",inplace = False, axis = 1), Y.loc[Y["Team-year"] == row[1]["T2"]].add_suffix("2").reset_index(drop = True).drop("Team-year2",inplace = False, axis = 1)], axis = 1)], axis = 0)
    y = y.reset_index(drop = True)
    print(y)
    y.to_csv('y.csv')
    X = X.drop("T1", inplace = False, axis = 1)
    X = X.drop("T2", inplace = False, axis = 1)
    X = X.drop("T1 PTS", inplace = False, axis = 1)
    X = X.drop("T2 PTS", inplace = False, axis = 1)
    X = X.drop("T1 FGA", inplace = False, axis = 1)
    X = X.drop("T2 FGA", inplace = False, axis = 1)
    X = X.drop("T1 FGM", inplace = False, axis = 1)
    X = X.drop("T2 FGM", inplace = False, axis = 1)
    X = X.drop("T1 PCT", inplace = False, axis = 1)
    X = X.drop("T2 PCT", inplace = False, axis = 1)
    X = X.drop("DIFF", inplace = False, axis = 1)
    X = X.drop("RESULT", inplace = False, axis = 1)

    print(X)
    X_train, X_test, y_train, y_test = train_test_split(y, X, random_state=1, train_size = training)
    model = MLPRegressor(random_state=seed, max_iter = 500).fit(X_train, y_train)
    a = model.predict(X_test)
    print(a)
    print(y_test)
    b = model.score(X_test, y_test)
    print(b)
    return(b)

def get_team_stats(T1, T2):
    x = T1+T2#concatenate their power rankings from data2 or something
    return x

def match_pred(model, T1, T2):
        x = get_team_stats(T1, T2)
        a = model.predict(x)
        return a


#^^^lets get this working first, this is i think the bare minimum^^^


def past_k_match_pred(data, T1, T2, y2d, seed, k):
        X, y = read_data(data)
        #make the data sequences of k games chronologically, we should record data in order to help with this, or get the dates in the data so that they can be sorted afterwards, I guess we would be training to predict the result of the final game in the sequence
        model = MLPRegressor(random_state=seed, max_iter = 500).fit(X, y)
        X, y = read_data(y2d)
        #make the data sequences of k games chronologically
        x = get_team_stats(T1, T2)
        #this would have to get added onto the last 2 games from X, y to predict it's results
        a = model.predict(x)
        return a





def read_file(file_name): #add a true/false
    df = pd.read_csv(file_name) #reading in data to pandas datafram called "df"
    return df


sum = 0
max = 0
for i in range(100):
    nn = train_test_C("NCAC21_22 - Copy of Sheet2.csv", "NCAC21_22 - Sheet1.csv", 0.75, 123+i)
    accuracy = nn[0]
    model = nn[1]
    sum += accuracy
    if accuracy > max:
        max = accuracy
        best_model = model

print("Accuracy Average: ", sum/100)
print("Best Model: ", max)


#best test of win vs loss classifier was average of 79% accuracy over 30 runs with training size of .9, also got 79% on 50 and 100 runs
