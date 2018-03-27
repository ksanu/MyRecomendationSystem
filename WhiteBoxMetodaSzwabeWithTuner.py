from sklearn.tree import DecisionTreeClassifier
from DataPreparators.DPwithMovieGenresCountriesAndDirectors import DPwithMovieGenresCountriesAndDirectors
from DataPreparators.DPForWhiteBoxMetodaSzwabe import DPForWhiteBoxMetodaSzwabe
import pickle
from ParamsTunerWithGridAndRandomSearch import ParamsTunerWithGridAndRandomSearch
from sklearn.metrics import roc_curve, auc

#przykladowy fragment kodu uczenia drzewa decyzyjnego metodami grid i random search:
#GSAuROCOptimalDTC=GridSearchCV(DecisionTreeClassifier(),gridSearchCVParameters, scoring='roc_auc')
#GSAuROCOptimalDTC.fit(X=X_trainPrim,y=y_trainPrim)
#RSAuROCOptimalDTC=RandomizedSearchCV(DecisionTreeClassifier(), randomSearchCVParameters,scoring='roc_auc',n_iter=numberOfGridSearchPoints)
#RSAuROCOptimalDTC.fit(X=X_trainPrim,y=y_trainPrim)

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

nrows=10000
DP = DPwithMovieGenresCountriesAndDirectors(nrows=nrows)
#load trained "black box" model from disk
filename = ".\\TrainedModels\\10000XGBClassifier_bestmodel_DPwithMovieGenresCountriesAndDirectors"
loaded_model = pickle.load(open(filename, 'rb'))

optimalBlackBoxClassifierXGBC = loaded_model
currentBlackBoxAlgorithm = "GB"
DP_white_box = DPForWhiteBoxMetodaSzwabe(optimalBlackBoxClassifier=optimalBlackBoxClassifierXGBC,
										 currentBlackBoxAlgorithm=currentBlackBoxAlgorithm, normalDP=DP, nrows=nrows)

WhiteBoxClassifier = DecisionTreeClassifier()
myTunerForWhiteBox = ParamsTunerWithGridAndRandomSearch(myClassifier=WhiteBoxClassifier,
	data_preparators=[DP_white_box], grid_params=gridSearchCVParameters, rand_params=randomSearchCVParameters,
														info="WhiteBoxs\\WhiteBox")

BestFoundWhiteBoxClassifier = myTunerForWhiteBox.startTuning()

#testowanie modelu white box zanalezionego w wyniku random i grid search
X_test, y_test = DP_white_box.getX_test_y_test()

y_probs = BestFoundWhiteBoxClassifier.predict_proba(X_test)
y_pred = y_probs[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('LR AUC: {}'.format(roc_auc))

#zapisanie modelu whiteBox i wyniku testu
filename = str(nrows)+"rowsTestedWhiteBox" + type(BestFoundWhiteBoxClassifier).__name__ + type(DP_white_box).__name__
# zapisywanie modelu:
pickle.dump(BestFoundWhiteBoxClassifier, open(file=filename, mode='wb'))
# zapisywanie parametrów i wyniku
mfile = open(file=filename + "_testResults", mode="w")
mfile.write("WBparams: " + str(BestFoundWhiteBoxClassifier.get_params()) + "\n" +
			"test result(roc_auc): " + str(roc_auc) + "\nnormalDP: " + type(DP).__name__ +
			"\nwhiteboxDP: " + type(DP_white_box).__name__ + "\nnrows: " + str(nrows))
mfile.flush()
mfile.close()