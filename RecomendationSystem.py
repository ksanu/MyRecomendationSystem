from sklearn.linear_model import LogisticRegression
from DataPreparators.DPwithMovieGenres import DPwithMovieGenres
from DataPreparators.DPwithMovieGenresAndCountries import DPwithMovieGenresAndCountries
from DataPreparators.DPwithMovieGenresCountriesAndDirectors import DPwithMovieGenresCountriesAndDirectors
import xgboost as xgb
from ParamsTunerWithGridAndRandomSearch import ParamsTunerWithGridAndRandomSearch
import xgboost as xgb



mNrows = 10000
DP1 = DPwithMovieGenres(nrows=mNrows)
DP2 = DPwithMovieGenresAndCountries(nrows=mNrows)
DP3 = DPwithMovieGenresCountriesAndDirectors(nrows=mNrows)
mDataPreparators = [DP1, DP2, DP3]

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


#myTuner = ParamsTunerWithGridAndRandomSearch(myClassifier=myLRC, data_preparators=mDataPreparators,
#                                             grid_params=gridSearchParametersLRC, rand_params=randomSearchParametersLRC)
#myTuner.startTuning()

myXGBC = xgb.XGBClassifier()
gridSearchParametersXGB={
 'base_score': [0.2, 0.5, 0.7],
 #'booster': ['gbtree','dart'],#which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
 #'colsample_bylevel': [0.2, 1],
 'colsample_bytree': [0.2,1],
 'gamma': [0, 10, 1000],
 #'learning_rate': [0.3, 0.9],
 #'max_delta_step': [0, 9],
 'max_depth': [3, 6],
 #'reg_alpha': [0, 6, 10],
 #'reg_lambda': [1, 3, 7, 10],
 'silent': [0],#0 means printing running messages, 1 means silent mode
 }
randomSearchParametersXGB={
 'base_score': [0.2, 0.5, 0.7],
 'booster': ['gbtree','dart'],#which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function.
 'colsample_bylevel': [0.2, 0.5, 0.8, 1],
 'colsample_bytree': [0.2, 0.5, 0.8, 1],
 'gamma': [0, 1, 10, 100, 1000],
 'learning_rate': [0.1, 0.3, 0.6, 0.9],
 'max_delta_step': [0, 2, 6, 9],
 'max_depth': [2, 3, 6, 8],
 'reg_alpha': [0, 2, 6, 10],
 'reg_lambda': [1, 3, 7, 10],
 'silent': [0],#0 means printing running messages, 1 means silent mode
 }

myTuner = ParamsTunerWithGridAndRandomSearch(myClassifier=myXGBC, data_preparators=mDataPreparators,
                                             grid_params=gridSearchParametersXGB, rand_params=randomSearchParametersXGB)
myTuner.startTuning()
