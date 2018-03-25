from RSwithMovieGenresCountriesAndDirectors import getXy_MovieGenresCountriesAndDirectors, getTrainedLRClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

X, y = getXy_MovieGenresCountriesAndDirectors(nrows=10000)
#podział na zbiór treningowy i testowy
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

myLRC = LogisticRegression(max_iter=100)
gridSearchParametersLRC={
    'penalty': ['l2'],
    'C': [0.001,0.01,0.1,1,10,100,1000],
    'fit_intercept': [True, False],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    'verbose': [0, 1, 2, 4, 32]
}

#uczenie LRC metodą GridSearch:
GSAuROCOptimalLRC=GridSearchCV(myLRC, gridSearchParametersLRC, scoring='roc_auc')
GSAuROCOptimalLRC.fit(X=X,y=y)
print('Best GridSearch LR roc_AUC: {0} with:\n{1}'.format(GSAuROCOptimalLRC.best_score_, GSAuROCOptimalLRC.best_params_))
