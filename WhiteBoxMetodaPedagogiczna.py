from sklearn.tree import DecisionTreeClassifier
from DataPreparators.DPwithMovieGenresCountriesAndDirectors import DPwithMovieGenresCountriesAndDirectors
from DataPreparators.DPForWhiteBoxMetodaPedagogiczna import DPForWhiteBoxMetodaPedagogiczna
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import pickle
from ParamsTunerWithGridAndRandomSearch import ParamsTunerWithGridAndRandomSearch


DP = DPwithMovieGenresCountriesAndDirectors(nrows=100)
X, y = DP.getXy()
#podział na zbiór treningowy i testowy
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#load trained "black box" model from disk
filename = ".\\TrainedModels\\LogisticRegression_bestmodel_DPwithMovieGenresCountriesAndDirectors"
loaded_model = pickle.load(open(filename, 'rb'))

#optimalBlackBoxClassifier = getTrainedLRClassifier(X_train,X_test,y_train,y_test)
optimalBlackBoxClassifier = loaded_model

"""
probabilities=optimalBlackBoxClassifier.predict_proba(X_train)[:,0]
bestProbas=np.argsort(probabilities)[:int(np.sum(y_train))]
y_trainPrim=np.zeros(len(y_train))

for i in bestProbas:
	y_trainPrim[i]=1
X_trainPrim=X_train

WhiteBoxClassifier = DecisionTreeClassifier()

#max_depth - maksymalna wysokość drzewa
#max_leaf_nodes - maksymalna liczba liści w drzewie
#min_samples_split - minimalna liczba próbek definiujących nowy węzeł
#min_samples_leaf - minimalna liczba próbek definiujących nowy węzeł będący liściem

gridSearchCVParameters={
	'max_depth': [2, 3, 5, None],
	'min_samples_leaf': [0.001, 0.00183, 0.00336, 0.00616, 0.01136, 0.02074, 0.03805, 0.06979, 0.128],
	'max_leaf_nodes': [16, 32, 63, 127],
	'min_samples_split': [0.0001, 0.001, 0.01, 0.1]}

#uczenie drzewa decyzyjnego metodamą GridSearch:
GSAuROCOptimalDTC=GridSearchCV(WhiteBoxClassifier,gridSearchCVParameters, scoring='roc_auc')
GSAuROCOptimalDTC.fit(X=X_trainPrim,y=y_trainPrim)

randomSearchCVParameters={
	'max_depth': [2, 3, 4, 5, 6, 7, 8, None],
	'min_samples_leaf': [0.001, 0.001834, 0.00336, 0.00616, 0.01136, 0.02074, 0.03805, 0.069797, 0.128],
	'max_leaf_nodes': [16, 20, 26, 34, 45, 58, 76, 98, 127],
	'min_samples_split': [0.0001, 0.000237, 0.000562, 0.00133, 0.00316, 0.007498, 0.0177, 0.0421, 0.1]}

#Kombinatoryka- reguła mnożenia:
temp=1
for val in gridSearchCVParameters.values():
	temp=temp * val.__len__()
numberOfGridSearchPoints = temp
#numberOfGridSearchPoints powinno wyjść 4*4*4*9=576
#uczenie drzewa decyzyjnego metodamą RandomizedSearch:
RSAuROCOptimalDTC=RandomizedSearchCV(WhiteBoxClassifier, randomSearchCVParameters, scoring='roc_auc',n_iter=numberOfGridSearchPoints)
RSAuROCOptimalDTC.fit(X=X_trainPrim,y=y_trainPrim)


if GSAuROCOptimalDTC.best_score_>RSAuROCOptimalDTC.best_score_:
	WhiteBoxClassifier=GSAuROCOptimalDTC.best_estimator_
else:
	WhiteBoxClassifier=RSAuROCOptimalDTC.best_estimator_
"""
#max_depth - maksymalna wysokość drzewa
#max_leaf_nodes - maksymalna liczba liści w drzewie
#min_samples_split - minimalna liczba próbek definiujących nowy węzeł
#min_samples_leaf - minimalna liczba próbek definiujących nowy węzeł będący liściem

gridSearchCVParameters={
	'max_depth': [2, 3, 5, None],
	'min_samples_leaf': [0.001, 0.00183, 0.00336, 0.00616, 0.01136, 0.02074, 0.03805, 0.06979, 0.128],
	'max_leaf_nodes': [16, 32, 63, 127],
	'min_samples_split': [0.0001, 0.001, 0.01, 0.1]}

randomSearchCVParameters={
	'max_depth': [2, 3, 4, 5, 6, 7, 8, None],
	'min_samples_leaf': [0.001, 0.001834, 0.00336, 0.00616, 0.01136, 0.02074, 0.03805, 0.069797, 0.128],
	'max_leaf_nodes': [16, 20, 26, 34, 45, 58, 76, 98, 127],
	'min_samples_split': [0.0001, 0.000237, 0.000562, 0.00133, 0.00316, 0.007498, 0.0177, 0.0421, 0.1]}

DP_white_box = DPForWhiteBoxMetodaPedagogiczna(optimalBlackBoxClassifier, X_train, y_train, X_test, y_test)

WhiteBoxClassifier = DecisionTreeClassifier()
myTunerForWhiteBox = ParamsTunerWithGridAndRandomSearch(myClassifier=WhiteBoxClassifier,
	data_preparators=[DP_white_box], grid_params=gridSearchCVParameters, rand_params=randomSearchCVParameters, info="WhiteBoxs\\WhiteBox")

WhiteBoxClassifier, _ = myTunerForWhiteBox.startTuning()

#tree.export_graphviz(WhiteBoxClassifier, out_file='whiteBoxClassifier_tree.dot')