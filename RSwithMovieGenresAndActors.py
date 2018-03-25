import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack


datasetFilesPath="C:\\hetrec2011-movielens-2k-v2\\"

#wczytywanie danych:
userRatedMoviesFileName="user_ratedmovies.dat"
DFUserRatedMovies=pd.read_csv(datasetFilesPath+userRatedMoviesFileName, header=0, delimiter="\t",usecols=['userID','movieID','rating'],nrows=100000)

DFMovieGenresFileName="movie_genres.dat"
DFMovieGenres=pd.read_csv(datasetFilesPath+DFMovieGenresFileName, header=0, delimiter="\t")

DFMovieActorsFileName="movie_actors.dat"
DFMovieActors=pd.read_csv(datasetFilesPath+DFMovieActorsFileName, header=0, delimiter="\t", encoding='iso-8859-1', usecols=['movieID', 'actorID', 'ranking'], nrows=10000)

#przygotowanie wczytanych danych
    #movie genres:

DFMovieGenres['dummyColumn']=1
DFMovieGenresPivoted=DFMovieGenres.pivot_table(index="movieID",columns="genre",values="dummyColumn")
DFMovieGenresPivoted['movieID']=DFMovieGenresPivoted.index
DFUserRatedMoviesWithMovieGenres=pd.merge(DFUserRatedMovies,DFMovieGenresPivoted,on='movieID')

    #actors:
DFMovieActorsPivoted = DFMovieActors.pivot_table(index="movieID", columns="actorID", values="ranking")
DFMovieActorsPivoted['movieID']=DFMovieActorsPivoted.index
#DFUserRatedMoviesWithActors=pd.merge(DFUserRatedMovies, DFMovieActorsPivoted, on='movieID')
#DFUserRatedMoviesWithActors=DFUserRatedMoviesWithActors.fillna(value=0)
#DFUserRatedMoviesWithActors_X=DFUserRatedMoviesWithActors.drop("rating",1)
#DFUserRatedMoviesWithActors_X=DFUserRatedMoviesWithActors_X.drop("userID",1)
#DFUserRatedMoviesWithActors_X=DFUserRatedMoviesWithActors_X.drop("movieID",1)

#XMovieActors=csr_matrix(DFUserRatedMoviesWithActors_X.values)

DFUserRatedMoviesWithMovieGenresActors=pd.merge(DFUserRatedMoviesWithMovieGenres, DFMovieActorsPivoted, on='movieID')
DFUserRatedMoviesWithMovieGenresActors=DFUserRatedMoviesWithMovieGenresActors.fillna(value=0)

#DFUserRatedMoviesWithMovieGenresActors_X=DFUserRatedMoviesWithMovieGenresActors.drop("rating",1)
#DFUserRatedMoviesWithMovieGenresActors_X=DFUserRatedMoviesWithMovieGenresActors_X.drop("userID",1)
#DFUserRatedMoviesWithMovieGenresActors_X=DFUserRatedMoviesWithMovieGenresActors_X.drop("movieID",1)


DFUserRatedMoviesWithMovieGenresActors["rating"]=DFUserRatedMoviesWithMovieGenresActors["rating"]>DFUserRatedMoviesWithMovieGenresActors["rating"].mean()
DFUserRatedMoviesWithMovieGenresActors_y=DFUserRatedMoviesWithMovieGenresActors["rating"]
yTemp=DFUserRatedMoviesWithMovieGenresActors_y.values
#macierz y
y=np.where(yTemp,1,-1)
DFUserRatedMoviesWithMovieGenresActors_X=DFUserRatedMoviesWithMovieGenresActors.drop("rating",1)

XUserID=DFUserRatedMoviesWithMovieGenresActors_X["userID"].values
XMovieID=DFUserRatedMoviesWithMovieGenresActors_X["movieID"].values
DFUserRatedMoviesWithMovieGenresActors_X=DFUserRatedMoviesWithMovieGenresActors_X.drop("userID",1)
DFUserRatedMoviesWithMovieGenresActors_X=DFUserRatedMoviesWithMovieGenresActors_X.drop("movieID",1)

XMovieGenresAndActors=csr_matrix(DFUserRatedMoviesWithMovieGenresActors_X.values)

OHE=OneHotEncoder()
XUserIDOHEncoded=OHE.fit_transform(XUserID.reshape(-1, 1))
XMovieIDOHEncoded=OHE.fit_transform(XMovieID.reshape(-1, 1))
#macierz X
X=hstack([XUserIDOHEncoded,XMovieIDOHEncoded,XMovieGenresAndActors])

#podział na zbiór treningowy i testowy
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


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





