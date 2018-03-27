from sklearn.linear_model import LogisticRegression
from DataPreparators.DPwithMovieGenres import DPwithMovieGenres
from DataPreparators.DPwithMovieGenresAndCountries import DPwithMovieGenresAndCountries
from DataPreparators.DPwithMovieGenresCountriesAndDirectors import DPwithMovieGenresCountriesAndDirectors
from DataPreparators.DPwithMovieGenresAndPolynominalFeatures import DPwithMovieGenresAndPolynominalFeatures
from DataPreparators.DPwithPolyMovieGenresAndCountries import DPwithPolyMovieGenresAndCountries
from ParamsTunerWithGridAndRandomSearch import ParamsTunerWithGridAndRandomSearch
import xgboost as xgb
import TestAndSaveTrainedEstimator
from DataPreparators.DPForWhiteBoxMetodaSzwabe import DPForWhiteBoxMetodaSzwabe
from DataPreparators.DPForWhiteBoxMetodaPedagogiczna import DPForWhiteBoxMetodaPedagogiczna
from sklearn.tree import DecisionTreeClassifier

testname = "test1"
mNrows = 100
DP1 = DPwithMovieGenres(nrows=mNrows)
DP2 = DPwithMovieGenresAndCountries(nrows=mNrows)
DP3 = DPwithMovieGenresCountriesAndDirectors(nrows=mNrows)
DP4 = DPwithMovieGenresAndPolynominalFeatures(nrows=mNrows)
DP5 = DPwithPolyMovieGenresAndCountries(nrows=mNrows)
mDataPreparators = [DP1, DP2, DP3, DP4, DP5]

#Część 'blackBox:
myLRC = LogisticRegression()
gridSearchParametersLRC={
    'penalty': ['l2'],
    'C': [0.001,0.01,1,10,100],
    'fit_intercept': [True, False],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    'verbose': [0, 1, 2, 32]
}

randomSearchParametersLRC={
    'penalty': ['l2'],
    'C': [0.0001,0.001,0.01,0.1,1,10,100,1000],
    'fit_intercept': [True, False],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    'verbose': [0.001,0.01,1,10,100, 1000]
}

myTuner = ParamsTunerWithGridAndRandomSearch(myClassifier=myLRC, data_preparators=mDataPreparators,
                                            grid_params=gridSearchParametersLRC, rand_params=randomSearchParametersLRC,
                                             info="BlackBox"+"_" + testname+"_")
BestFoundLRC, DPforBestFoundLRC = myTuner.startTuning()
TestAndSaveTrainedEstimator.TestAndSaveTrainedEstimator(BestFoundLRC, DPforBestFoundLRC)

myXGBC = xgb.XGBClassifier()
gridSearchParametersXGB={
 'base_score': [0.2, 0.5, 0.7],
 #'booster': ['gbtree','dart'],#which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
 #'colsample_bylevel': [0.2, 1],
 'colsample_bytree': [0.2,1],
 'gamma': [0,0.01, 10],
 #'learning_rate': [0.3, 0.9],
 'max_delta_step': [0, 9],
 'max_depth': [3, 6],
 #'reg_alpha': [0, 6, 10],
 #'reg_lambda': [1, 3, 7, 10],
 'silent': [1],#0 means printing running messages, 1 means silent mode
 }
randomSearchParametersXGB={
 'base_score': [0.2, 0.5, 0.7],
 'booster': ['gbtree','dart'],#which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
 'colsample_bylevel': [0.2, 0.5, 0.8, 1],
 'colsample_bytree': [0.2, 0.5, 0.8, 1],
 'gamma': [0,0.001, 0.001, 0.01, 1, 10, 100, 1000],
 'learning_rate': [0.1, 0.3, 0.6, 0.9],
 'max_delta_step': [0, 2, 6, 9],
 'max_depth': [2, 3, 6, 8, 16],
 'reg_alpha': [0, 2, 6, 10],
 'reg_lambda': [1, 3, 7, 10],
 'min_child_weight': [1, 4, 8, 10, 20, 100],
 #'subsample ': [0.001, 0.01, 0.1, 1],
 'silent': [1],#0 means printing running messages, 1 means silent mode
 }

myTuner = ParamsTunerWithGridAndRandomSearch(myClassifier=myXGBC, data_preparators=mDataPreparators,
                                             grid_params=gridSearchParametersXGB, rand_params=randomSearchParametersXGB,
                                             info="BlackBox"+"_"+testname+"_")
BestFoundXGBC, DPforBestFoundXGBC = myTuner.startTuning()
TestAndSaveTrainedEstimator.TestAndSaveTrainedEstimator(BestFoundXGBC, DPforBestFoundXGBC)

#Część 'WhiteBox:
#Użycie BestFoundXGBC nastrojonego przez grid i random search jako 'Black Box'
optimalBlackBoxClassifierXGBC = BestFoundXGBC
currentBlackBoxAlgorithm = "GB"

DP_white_box_metodaSzwabe = DPForWhiteBoxMetodaSzwabe(optimalBlackBoxClassifier=optimalBlackBoxClassifierXGBC,
										 currentBlackBoxAlgorithm=currentBlackBoxAlgorithm, normalDP=DPforBestFoundXGBC)
DP_white_box_metodaPedagogiczna = DPForWhiteBoxMetodaPedagogiczna(optimalBlackBoxClassifier=optimalBlackBoxClassifierXGBC,
										 normalDP=DPforBestFoundXGBC)

WBDataPreps = [DP_white_box_metodaSzwabe, DP_white_box_metodaPedagogiczna]
gridSearchCVParametersDTC={
	'max_depth': [2, 3, 5, None],
	'min_samples_leaf': [0.001, 0.00183, 0.00336, 0.00616, 0.01136, 0.02074, 0.03805, 0.06979, 0.128],
	'max_leaf_nodes': [16, 32, 63, 127],
	'min_samples_split': [0.0001, 0.001, 0.01, 0.1]}

randomSearchCVParametersDTC={
	'max_depth': [2, 3, 4, 5, 6, 7, 8, None],
	'min_samples_leaf': [0.001, 0.001834, 0.00336, 0.00616, 0.01136, 0.02074, 0.03805, 0.069797, 0.128],
	'max_leaf_nodes': [16, 20, 26, 34, 45, 58, 76, 98, 127],
	'min_samples_split': [0.0001, 0.000237, 0.000562, 0.00133, 0.00316, 0.007498, 0.0177, 0.0421, 0.1]}

#strojenie parametrów:
WhiteBoxClassifier = DecisionTreeClassifier()
myTunerForWhiteBox = ParamsTunerWithGridAndRandomSearch(myClassifier=WhiteBoxClassifier,
	data_preparators=WBDataPreps, grid_params=gridSearchCVParametersDTC, rand_params=randomSearchCVParametersDTC,
														info="WhiteBoxs\\WhiteBox" + testname)
BestFoundWhiteBoxClassifier, DPforBestFoundWBC = myTunerForWhiteBox.startTuning()
#Testowanie białej skrzynki
TestAndSaveTrainedEstimator.TestAndSaveTrainedEstimator(BestFoundWhiteBoxClassifier, DPforBestFoundWBC, type(DPforBestFoundWBC.normalDP).__name__)

#użycie BestFoundLRC jako black box. Użycie metody pedagogicznej, bo metoda Szwabe wymaga GradientBoostingClassifier
#lub RandomForestClassifier
DP_white_box_metodaPedagogiczna = DPForWhiteBoxMetodaPedagogiczna(optimalBlackBoxClassifier=BestFoundLRC,
										 normalDP=DPforBestFoundLRC)
WhiteBoxClassifier = DecisionTreeClassifier()
myTunerForWhiteBox = ParamsTunerWithGridAndRandomSearch(myClassifier=WhiteBoxClassifier,
	data_preparators=[DP_white_box_metodaPedagogiczna], grid_params=gridSearchCVParametersDTC, rand_params=randomSearchCVParametersDTC,
														info="WhiteBoxs\\WhiteBox" + testname)
BestFoundWhiteBoxClassifier, DPforBestFoundWBC = myTunerForWhiteBox.startTuning()
#Testowanie białej skrzynki
TestAndSaveTrainedEstimator.TestAndSaveTrainedEstimator(BestFoundWhiteBoxClassifier, DPforBestFoundWBC, type(DPforBestFoundWBC.normalDP).__name__)

