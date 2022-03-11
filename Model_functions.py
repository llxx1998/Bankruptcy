from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np


class Winsor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_new = pd.DataFrame(np.array(X).copy())
        self.High = []
        self.Low = []
        for col in X_new.columns:
            self.High.append(X_new[col].quantile(0.9))
            self.Low.append(X_new[col].quantile(0.1))
        return self

    def transform(self, X, y=None):
        X_new = pd.DataFrame(np.array(X).copy())
        for i, col in enumerate(X_new.columns):
            high = self.High[i]
            low = self.Low[i]
            X_new.loc[X_new[col] > high, col] = high
            X_new.loc[X_new[col] < low, col] = low
        return X_new.values


class Mfunc:
    """
    Function returning Imputer Pipelines
    """
    def __init__(self):
        self.strategy = "most_frequent"
        self.max_iter = 10

    def utilityFunc(self, y, y_pred):
        return (5/6) * recall_score(y, y_pred) + (1/6) * precision_score(y, y_pred)

    def utilityFunc2(self, y, y_pred):
        return recall_score(y, y_pred)

    def simpleimputer(self, type='most_frequent'):
        return Pipeline([('simpleimputer', SimpleImputer(strategy=type))]), 'simpleimputer'

    def multivariateimputer(self, max_iter=10):
        return Pipeline([('multivariateimputer', IterativeImputer(max_iter=max_iter))]), "multivariateimputer"

    def combinedimputator(self, mv_list):
        simplepipe, simplename = self.simpleimputer("most_frequent")
        mvpipe, mvname = self.multivariateimputer(max_iter=10)
        column_trans = make_column_transformer((mvpipe, mv_list), remainder=simplepipe)
        return column_trans, 'combinedimputator'

    def winsortransform(self):
        return Pipeline([('winsortransform', Winsor())]), 'winsortransform'

    def oversample(self):
        return Pipeline([("oversample", SMOTE())]), 'oversample'

    def pca(self):
        return Pipeline([('pca', PCA(n_components=0.95))]), 'pca'

    def scaler(self):
        return Pipeline([('scaler', StandardScaler())]), 'scaler'

    def PipelineDict(self, classifier, clf_name):
        simp = SimpleImputer(strategy=self.strategy)
        mimp = IterativeImputer(max_iter=self.max_iter)
        out = Winsor()
        pca = PCA(n_components=0.95)
        scl = StandardScaler()
        smt = SMOTE()

        pipe_dict = dict()
        pipe_dict[1] = Pipeline([('simp', simp), ('out', out), ('scl', scl), ('pca', pca), ('smt', smt), (clf_name, classifier)])
        pipe_dict[2] = Pipeline([('simp', simp), ('out', out), ('scl', scl), (clf_name, classifier)])
        pipe_dict[3] = Pipeline([('simp', simp), ('out', out), ('scl', scl), ('pca', pca), ('smt', smt), (clf_name, classifier)])
        pipe_dict[4] = Pipeline([('mimp', mimp), ('out', out), ('scl', scl), ('pca', pca), ('smt', smt), (clf_name, classifier)])
        pipe_dict[5] = Pipeline([('mimp', mimp), ('out', out), ('scl', scl), (clf_name, classifier)])
        pipe_dict[6] = Pipeline([('mimp', mimp), ('out', out), ('scl', scl), ('pca', pca), (clf_name, classifier)])
        pipe_dict[7] = Pipeline([('mimp', mimp), ('out', out), ('scl', scl), ('smt', smt), (clf_name, classifier)])

        return pipe_dict

    def FitPipeDict(self, X_train, y_train, pipe_dict, scoring, cv=5):
        result = []
        for i in range(len(pipe_dict)):
            pipe = pipe_dict[i+1]
            result.append([i+1] + self.evaluate(pipe, X_train, y_train, scoring, cv))
        return result

    def evaluate(self, pipe, X_train, y_train, scoring, cv):
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_train)
        return [sum(cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring))/cv, self.utilityFunc2(y_train, y_hat),
                sum(cross_val_score(pipe, X_train, y_train, cv=cv))/cv]






