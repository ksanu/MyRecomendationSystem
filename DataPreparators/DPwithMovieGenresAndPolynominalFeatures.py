import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import PolynomialFeatures


class DPwithMovieGenresAndPolynominalFeatures:
    def __init__(self, nrows):
        self.myNrows = nrows

    def getXy(self):
        datasetFilesPath = "C:\\hetrec2011-movielens-2k-v2\\"

        # wczytywanie danych:
        userRatedMoviesFileName = "user_ratedmovies.dat"
        DFUserRatedMovies = pd.read_csv(datasetFilesPath + userRatedMoviesFileName, header=0, delimiter="\t",
                                        usecols=['userID', 'movieID', 'rating'], nrows=self.myNrows)

        DFMovieGenresFileName = "movie_genres.dat"
        DFMovieGenres = pd.read_csv(datasetFilesPath + DFMovieGenresFileName, header=0, delimiter="\t")

        # przygotowanie wczytanych danych
        # movie genres:
        DFMovieGenres['dummyColumn'] = 1
        DFMovieGenresPivoted = DFMovieGenres.pivot_table(index="movieID", columns="genre", values="dummyColumn")
        DFMovieGenresPivoted['movieID'] = DFMovieGenresPivoted.index
        DFUserRatedMoviesWithMovieGenres = pd.merge(DFUserRatedMovies, DFMovieGenresPivoted, on='movieID')
        DFUserRatedMoviesWithMovieGenres = DFUserRatedMoviesWithMovieGenres.fillna(value=0)

        DFUserRatedMoviesWithMovieGenres["rating"] = DFUserRatedMoviesWithMovieGenres["rating"] > \
                                                     DFUserRatedMoviesWithMovieGenres["rating"].mean()
        DFUserRatedMoviesWithMovieGenres_y = DFUserRatedMoviesWithMovieGenres["rating"]
        yTemp = DFUserRatedMoviesWithMovieGenres_y.values
        # macierz y
        y = np.where(yTemp, 1, -1)
        DFUserRatedMoviesWithMovieGenres_X = DFUserRatedMoviesWithMovieGenres.drop("rating", 1)

        XUserID = DFUserRatedMoviesWithMovieGenres_X["userID"].values
        XMovieID = DFUserRatedMoviesWithMovieGenres_X["movieID"].values
        DFUserRatedMoviesWithMovieGenres_X = DFUserRatedMoviesWithMovieGenres_X.drop("userID", 1)
        DFUserRatedMoviesWithMovieGenres_X = DFUserRatedMoviesWithMovieGenres_X.drop("movieID", 1)

        XMovieGenres = csr_matrix(DFUserRatedMoviesWithMovieGenres_X.values)
        OHE = OneHotEncoder()
        XUserIDOHEncoded = OHE.fit_transform(XUserID.reshape(-1, 1))
        XMovieIDOHEncoded = OHE.fit_transform(XMovieID.reshape(-1, 1))
        # macierz X
        X = hstack([XUserIDOHEncoded, XMovieIDOHEncoded, XMovieGenres])

        # dodanie cech wielomianowych
        PF = PolynomialFeatures(degree=2, interaction_only=True)
        polyX = PF.fit_transform(X=X.toarray(), y=y)
        sparsePolyX = csr_matrix(polyX)
        return sparsePolyX, y

"""
#podział na zbiór treningowy i testowy
X_train,X_test,y_train,y_test=train_test_split(sparsePolyX,y,test_size=0.2)


def xgbClassification(X_train, X_test, y_train, y_test):
    xgTrain = xgb.DMatrix(X_train, label=y_train)
    xgTest = xgb.DMatrix(X_test, label=y_test)

    #num_rounds = 10

    #params = {}
    #params["max_depth"] = 1000
    #params["max_delta_step"] = 20000
    # params["max_delta_step"]=200
    # params["gamma"]=0.001
    # params["eta"]=0.1

    #!Trenowanie modelu:
    xgb_model = xgb.XGBClassifier()
    #recSys = xgb.train(params, xgTrain, num_rounds)
    xgb_model.fit(X_train, y_train)

    #ypred = recSys.predict(xgTest)
    #threshold = 0.5
    #y_pred = np.where(ypred > threshold, 1, 0)

    # make predictions for test data
    y_pred = xgb_model.predict(X_test)
    #predictions = [round(value) for value in y_pred]

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print('GB AUC: {}'.format(roc_auc))


def logisticRegressionClassification(X_train,X_test,y_train,y_test):
    LRC = LogisticRegression()
    LRC.fit(X_train, y_train)
    y_probs = LRC.predict_proba(X_test)
    y_pred = y_probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print('LR AUC: {}'.format(roc_auc))


xgbClassification(X_train,X_test,y_train,y_test)

logisticRegressionClassification(X_train,X_test,y_train,y_test)
"""




