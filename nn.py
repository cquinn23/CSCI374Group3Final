import numpy as np
import pandas as pd
import sklearn
import math
from sklearn.neural_network import MLPRegressor

#ok so MLPRegressor is basically just a classifier without an activation function so it gives us n variables as an output, i think we can train these to the stats we want to predict
#also this has a partial_fit() function so if we wanted to go over the y2d results as a separate thing we could do that, the idea being past years have pretty representative power rankings, but in progress years might not be quite perfect

def main(data, training, seed, games_to_predict):
    model = train_test(data, training, seed)
    preads = model.predict(games_to_predict)


def train_test(data, training, seed):
    X, y = read_data(data)
    X_train, X_test, y_train, y_test = sklearn.train_test_split(X, y, random_state=1)
    model = MLPRegressor(random_state=seed, max_iter = 500).fit(X_train, y_train)
    a = model.predict(X_test)
    print(a)
    model.score(X_test, y_test)
    return model
    
    

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


def read_data(file_name):
    X = 0
    y = 0
    return X, y

def get_team_stats(T1, T2):
    x = T1+T2
    return x