from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from classifier_evaluator import ClassifierEvaluator
import os
import warnings
import numpy as np
import pandas as pd


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 200)
warnings.filterwarnings('ignore')


def college_data(loc):
    college_loc = './data/college/'
    file_names = os.listdir(college_loc)
    years = [f.split('_')[0] for f in file_names]
    positions = [f.split('_')[1] for f in file_names]
    frames = [pd.read_csv(college_loc + f) for f in os.listdir(college_loc)]
    for f in range(len(frames)):
        frames[f].columns = ['Rk', 'Player', 'School', 'Conf', 'G', 'Rec', 
                'RecYds', 'RecAvg', 'RecTD', 'RushAtt', 'RushYds', 'RushAvg',
                'RushTD', 'ScrimPlays', 'ScrimYds', 'ScrimAvg', 'ScrimTD']
        frames[f] = frames[f].tail(len(frames[f])-1)
        frames[f] = frames[f].reset_index(drop=True)
        frames[f]['Season'] = int(years[f])
        frames[f]['Position'] = positions[f]
        frames[f]['Player'] = [p.split('\\')[0] for p in frames[f]['Player'].values]
        frames[f]['Player'] = [p.replace('*', '') for p in frames[f]['Player'].values]
    df = pd.concat(frames)
    df = df[[c for c in df.columns if c != 'Rk']]
    return df.reset_index(drop=True)


def unique_players(df):
    players = df[['Player', 'Draft Year']]
    players = players.drop_duplicates()
    return players


def only_last_season(df):
    unique = unique_players(df)
    frames = []
    for u in unique.values:
        temp = df[df['Player']==u[0]]
        temp = temp[temp['Draft Year']==u[1]]
        temp = temp[temp['Season']==max(temp['Season'].values)]
        frames.append(temp)
    return pd.concat(frames).drop_duplicates()


def only_best_season(df):
    unique = unique_players(df)
    frames = []
    for u in unique.values:
        temp = df[df['Player']==u[0]]
        temp['Rec'] = temp['Rec'].astype(int)
        temp['RecYds'] = temp['RecYds'].astype(int)
        temp = temp[temp['Draft Year']==u[1]]
        temp = temp[temp['Rec']==max(temp['Rec'].values)]
        temp = temp[temp['RecYds']==max(temp['RecYds'].values)]

        frames.append(temp)
    return pd.concat(frames).drop_duplicates()


def weighted_stats(df):
    pass


def linreg_fill(data):
    model_data = data.dropna()
    X = model_data[model_data.columns[0]].values.reshape(-1, 1)
    Y = model_data[model_data.columns[1]].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, Y)
    X = data[data.columns[0]].values.reshape(-1, 1)
    preds = model.predict(X)
    data['pred'] = preds
    data['temp'] = data[data.columns[1]].fillna(data['pred'])
    return data['temp'].values


def main():
    combine_loc = './data/combine/'
    combine_files = [combine_loc + f for f in os.listdir(combine_loc)] 
    combine_df = pd.concat([pd.read_csv(f) for f in combine_files])
    combine_df['Draft Year'] = combine_df['Year'] - 1
    combine_df['Player'] = [p.split('\\')[0] for p in combine_df['Player'].values]
    combine_df = combine_df[[c for c in combine_df.columns if c != 'Rk']]
    college_df = college_data('./data/college/')
    df = college_df.merge(combine_df, on=['Player', 'School'])
    last = only_best_season(df)
    df = last[[
        'Rec', 'RecYds', 'RecAvg', 'RecTD', 'RushAtt', 'RushYds',
        'RushAvg', 'RushTD', 
        'Age', 'Height', 'Wt', '40YD', 'Vertical', 'BenchReps', 'Broad Jump',
        '3Cone', 'Shuttle', 'AV']]
    df['Height'] = [int(h.split('-')[0])*12 + int(h.split('-')[1]) for h in df['Height'].values]
    for col in  ['40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle']:
        temp = df[['Wt', col]]
        df[col] = linreg_fill(temp)
    for col in df.columns:
        df[col] = df[col].fillna(0) 
    X = df[df.columns[:-1]].values
    Y = df['AV'].values
    """
    df['S'] = df['AV'] > np.mean(df['AV'])
    df['S'] = [int(b) for b in df['S'].values]
    Y = df[df.columns[-1]].values
    models = [
            LogisticRegression(), 
            GaussianNB(), 
            ]
    bag = [BaggingClassifier(base_estimator=model, n_estimators=50) for model in models]
    extra = [
            GaussianProcessClassifier(), 
            KNeighborsClassifier(), 
            NearestCentroid(),
            ]
    ce = ClassifierEvaluator(X, Y)
    ce.load_models(models + bag + extra)
    ce.fit_models()
    ce.score()
    print(ce.report().sort_values('F1', ascending=False))

    """
    model = LinearRegression()
    model.fit(X, Y)
    last['pred'] = model.predict(X) 
    last = last[['Player', 'School', 'Draft Year', 'AV', 'pred']]
    last = last.sort_values('pred', ascending=False)
    print(last)





if __name__ == '__main__':
    main()
