from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 250)
warnings.filterwarnings('ignore')


def process(df):
    df['SUCCESS'] = df['CarAV'] > np.mean(df['CarAV'])
    df['SUCCESS'] = df['SUCCESS'].astype(int)
    X_train = df['Pick'].values
    Y_train = df['SUCCESS'].values
    model = LogisticRegression()
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    model.fit(X_train.reshape(-1,1), Y_train)
    preds = model.predict_proba(np.array(range(1,257)).reshape(-1,1))
    preds = [p[1] for p in preds]
    pos = list(set(list(df['Pos'].values)))[0]
    poss = [pos for p in preds]
    df = pd.DataFrame([poss, preds])
    return df.T


def main():
    loc = './data/picks/'
    no = ['Salary.csv', 'results.csv']
    files = [loc + f for f in os.listdir(loc) if f not in no] 
    df = pd.concat([pd.read_csv(f) for f in files])
    translate = {v: v for v in df['Pos'].values}
    translate['NT'] = 'DT'
    translate['G'] = 'iOL'
    translate['C'] = 'iOL'
    translate['DE'] = 'EDGE'
    translate['OLB'] = 'EDGE'
    translate['ILB'] = 'LB'
    translate['S'] = 'DB'
    translate['CB'] = 'DB'
    df['Pos'] = [translate[p] for p in df['Pos'].values]
    frames = [df[df['Pos'] == p] for p in list(set(list(df['Pos'].values)))]
    info = [process(f) for f in frames]
    probs = pd.concat(info)
    probs.columns = ['Position', 'Probability']
    probs['Pick'] = [p+1 for p in probs.index]
    probs = probs.sort_values('Pick') 
    positions = [probs[probs['Position']==p] for p in list(set(list(probs['Position'].values)))]
    posits = []
    probabilities = []
    for p in positions:
        pos = list(set(list(p['Position'].values)))[0]
        temp = p['Probability']
        posits.append(pos)
        probabilities.append(temp)
    df = pd.DataFrame(probabilities)
    df.index = posits
    df = df.T
    df.index = df.index + 1
    df = df[['QB', 'RB', 'WR', 'TE', 'T', 'iOL', 'EDGE', 'DT', 'LB', 'DB']]
    df.to_csv('./data/picks/results.csv')

    for pos in df.columns:
        plt.plot(df[pos].values)
    plt.legend(labels = df.columns)
    plt.show()
    plt.savefig('./data/picks/results.png')

    print(df)


if __name__ == '__main__':
    main()
