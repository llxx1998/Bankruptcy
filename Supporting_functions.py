import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


class Sfunc:
    def ShuffleSplit(self, df: pd.DataFrame, test_size=0.1):
        X, y = df.iloc[:, :-2], df.iloc[:, -2]
        return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)

    def DesTable(self, df: pd.DataFrame):
        return df.describe().T

    def DesBoxplot(self, df: pd.DataFrame):
        return df.plot(kind='density', subplots=True, layout=(8, 8), figsize=(20, 20), sharex=False)

    def CondPlot(self, X: pd.DataFrame, y: pd.DataFrame):
        df = (X - X.min()) / (X.max() - X.min())
        df['y'] = y
        return df.boxplot(by=['y'], layout=(8, 8), figsize=(20, 20))

    def CountNa(self, X: pd.DataFrame):
        return X.isna().sum(axis=0)

    def ReportNa(self, na_number: pd.Series):
        plot = na_number.plot()
        na_variables = na_number > 0
        na_num = na_variables.sum()
        print("{} out of 64 variables have NaN.".
              format(na_num))
        print("5 variables with the most NaN:\n\n",
              na_number.sort_values(ascending=False).head())
        return plot

    def OutlierPlot(self, X_train, n_std=3):
        temp = X_train.describe().T[["mean", "max", "min", "std"]]
        temp['max'] = (temp['max'] - temp['mean']) / temp['std']
        temp['min'] = (temp['min'] - temp['mean']) / temp['std']
        temp['std_down'], temp['std_up'] = -n_std, n_std
        return temp[['max', 'min', 'std_down', 'std_up']].plot()

    def OutlierPercentage(self, X_train, n_std=3):
        temp = ((X_train - X_train.mean()) / X_train.std()).abs()
        temp = (temp > n_std)
        temp = temp.sum() / temp.count()
        print("Highest outlier percentage defined by >= {b} std is {a: .4f}.".format(a=temp.max(), b=n_std))
        print("Lowest outlier percentage defined by >= {b} std is {a: .4f}.\n".format(a=temp.min(), b=n_std))
        print("5 variables with largest percentage of outliers:\n")
        print(temp.sort_values(ascending=False).head())
        return temp.plot()

    def ImbalanceShow(self, y_train):
        count_y = [(y_train == 0).sum(), (y_train == 1).sum()]
        print(count_y)
        c = plt.pie(count_y, labels=['Normal', 'Bankruptcy'])
        c = plt.legend()
        print("Bankrupted companies add up to {a: .4f}.".format(a=count_y[1] / sum(count_y)))
        print("Normal companies add up to {a: .4f}.".format(a=count_y[0] / sum(count_y)))
        return c

    def CorrelationShow(self, X_train):
        return X_train.corr().round(2)








        
        