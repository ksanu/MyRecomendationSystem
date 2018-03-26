from DataPreparators.DPwithMovieGenresAndCountries import DPwithMovieGenresAndCountries
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import pickle



#XGBClassifier_bestmodel_DPwithMovieGenresAndCountries_bestparams
#{'booster': 'gbtree', 'colsample_bytree': 1, 'silent': 0, 'reg_lambda': 1, 'colsample_bylevel': 0.2, 'gamma': 1, 'learning_rate': 0.1, 'reg_alpha': 2, 'max_depth': 3, 'max_delta_step': 9, 'base_score': 0.5}
#score: 0.761446886447

xgbBestParams = {'booster': 'gbtree', 'colsample_bytree': 1, 'silent': 1, 'reg_lambda': 1, 'colsample_bylevel': 0.2, 'gamma': 1, 'learning_rate': 0.1, 'reg_alpha': 2, 'max_depth': 3, 'max_delta_step': 9, 'base_score': 0.5}
nrows=10000
DP = DPwithMovieGenresAndCountries(nrows)
X, y = DP.getXy()
#podział na zbiór treningowy i testowy
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#!Trenowanie modelu:
xgb_model = xgb.XGBClassifier(**xgbBestParams)
xgb_model.fit(X_train, y_train)

# make predictions for test data
#y_pred = xgb_model.predict(X_test)
y_probs = xgb_model.predict_proba(X_test)
y_pred = y_probs[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('GB AUC: {}'.format(roc_auc))
print(xgb_model.get_xgb_params())


def saveResult(ClassifierModel, Params, score, data_preparator, nrows, info ="10000rows",):
    filename = info + type(ClassifierModel).__name__ + type(data_preparator).__name__
    # zapisywanie modelu:
    pickle.dump(ClassifierModel, open(file=filename, mode='wb'))
    # zapisywanie parametrów i wyniku
    mfile = open(file=filename + "_info", mode="w")
    mfile.write("params: " + str(Params) + "\n" + "score: " + str(score) + "\nnrows: " + str(nrows))
    mfile.flush()
    mfile.close()

saveResult(xgb_model, xgbBestParams, roc_auc, DP, nrows,)