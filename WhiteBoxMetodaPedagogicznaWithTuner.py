from sklearn.tree import DecisionTreeClassifier
from DataPreparators.DPwithMovieGenresCountriesAndDirectors import DPwithMovieGenresCountriesAndDirectors
from DataPreparators.DPForWhiteBox import DPForWhiteBox
from sklearn.model_selection import train_test_split
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

DP_white_box = DPForWhiteBox(optimalBlackBoxClassifier, X_train, y_train)

WhiteBoxClassifier = DecisionTreeClassifier()
myTunerForWhiteBox = ParamsTunerWithGridAndRandomSearch(myClassifier=WhiteBoxClassifier,
	data_preparators=[DP_white_box], grid_params=gridSearchCVParameters, rand_params=randomSearchCVParameters, info="WhiteBoxs\\WhiteBox")

WhiteBoxClassifier = myTunerForWhiteBox.startTuning()
print("whiteBox tuning result:\n params: " + str(myTunerForWhiteBox.best_params) + "\nroc_auc score: " + str(myTunerForWhiteBox.best_score))
#tree.export_graphviz(WhiteBoxClassifier, out_file='whiteBoxClassifier_tree.dot')