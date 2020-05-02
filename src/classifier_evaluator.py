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


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 225)
warnings.filterwarnings('ignore')


class ClassifierEvaluator:

    def __init__(self, X, Y):
        self.models = []
        self.data = []
        self.scores = []
        self.accuracy = []
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.split()

    def load_models(self, models):
        for model in models:
            self.models.append(model)
    
    def split(self, test_size=.2):
        self.data = train_test_split(self.X, self.Y, test_size=test_size)
    
    def fit_models(self):
        for model in self.models:
            print(model)
            model.fit(self.data[0], self.data[2])

    def score(self):
        for model in self.models:
            X_test = self.data[1]
            Y_pred = model.predict(X_test)
            Y_test = self.data[3]
            self.scores.append(f1_score(Y_test, Y_pred))
            self.accuracy.append(model.score(X_test, Y_test))
            
    def report(self):
        models = [str(m) for m in self.models]
        report_models = []
        for m in models:
            while '  ' in m:
                m = m.replace('  ', ' ')
            m = m.replace('\n', '')
            report_models.append(m)
        data = [report_models, self.accuracy, self.scores]
        df = pd.DataFrame(data).T
        df.columns = ['Model', 'Accuracy', 'F1']
        return df


def foo(df):
    df['Success'] = df['CarAV'] > np.mean(df['CarAV'])
    df['Success'] = df['Success'].astype(int)
    return df['Pick'].values, df['Success'].values


def main():
    loc = './data/'
    files = [loc + f for f in os.listdir(loc) if f != 'Salary.csv']
    df = pd.concat([pd.read_csv(f) for f in files])
    X, Y = foo(df)
    models = [
            AdaBoostClassifier(), ExtraTreesClassifier(), 
            GradientBoostingClassifier(), RandomForestClassifier(), 
            BaggingClassifier(), GaussianProcessClassifier(), 
            LogisticRegression(), Perceptron(), RidgeClassifier(), 
            SGDClassifier(), PassiveAggressiveClassifier(),
            ]
    ce = ClassifierEvaluator(X, Y)
    ce.load_models(models)
    ce.fit_models()
    ce.score()
    print(ce.report())

if __name__ == '__main__':
    main()
