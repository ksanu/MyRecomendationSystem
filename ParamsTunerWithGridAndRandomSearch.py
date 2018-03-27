import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ParamsTunerWithGridAndRandomSearch:
    def __init__(self, myClassifier, data_preparators, grid_params, rand_params, info=""):
        self.data_preparators = data_preparators
        self.grid_params = grid_params
        self.rand_params = rand_params
        self.myClassifier = myClassifier
        self.best_trained_model = 0
        self.best_score = 0
        self.best_params = 0
        self.info = info
        self.DPforBestModel = None

    def startTuning(self):
        for DP in self.data_preparators:
            #uzyskanie danych treningowych
            X_train, y_train = DP.getX_train_y_train()
            # uczenie modelu klasyfikatora metodą GridSearch:
            GSAuROCOptimalC = GridSearchCV(self.myClassifier, self.grid_params, scoring='roc_auc')
            GSAuROCOptimalC.fit(X=X_train, y=y_train)

            # Kombinatoryka- reguła mnożenia:
            temp = 1
            for val in self.grid_params.values():
                temp = temp * val.__len__()
            numberOfGridSearchPoints = temp

            # uczenie modelu klasyfikatora metodą RandomizedSearch:
            RSAuROCOptimalC = RandomizedSearchCV(self.myClassifier, self.rand_params, scoring='roc_auc',
                                                   n_iter=numberOfGridSearchPoints)
            RSAuROCOptimalC.fit(X=X_train, y=y_train)

            #zapisanie lepszego wyniku
            if GSAuROCOptimalC.best_score_ > RSAuROCOptimalC.best_score_:
                bestClassifierModel = GSAuROCOptimalC.best_estimator_
                bestParams = GSAuROCOptimalC.best_params_
                score = GSAuROCOptimalC.best_score_
                self.saveResult(bestClassifierModel=bestClassifierModel, bestParams=bestParams, score=score, data_preparator=DP, info=self.info)
                if GSAuROCOptimalC.best_score_>self.best_score:
                    self.best_trained_model = GSAuROCOptimalC.best_estimator_
                    self.best_score = GSAuROCOptimalC.best_score_
                    self.best_params = GSAuROCOptimalC.best_params_
                    self.DPforBestModel = DP
            else:
                bestClassifierModel = RSAuROCOptimalC.best_estimator_
                bestParams = RSAuROCOptimalC.best_params_
                score = RSAuROCOptimalC.best_score_
                self.saveResult(bestClassifierModel=bestClassifierModel, bestParams=bestParams, score=score, data_preparator=DP, info=self.info)
                if RSAuROCOptimalC.best_score_>self.best_score:
                    self.best_trained_model = RSAuROCOptimalC.best_estimator_
                    self.best_score = RSAuROCOptimalC.best_score_
                    self.best_params = RSAuROCOptimalC.best_params_
                    self.DPforBestModel = DP

        return self.best_trained_model, self.DPforBestModel

    def saveResult(self,bestClassifierModel, bestParams, score, data_preparator, info):
        filename = ".\\TrainedModels\\" + info +str(data_preparator.myNrows)+ type(bestClassifierModel).__name__ + "_bestmodel_" + type(data_preparator).__name__
        #zapisywanie modelu:
        pickle.dump(bestClassifierModel, open(file=filename, mode='wb'))
        #zapisywanie parametrów i wyniku
        mfile = open(file=filename+"_bestparams", mode="w")
        mfile.write("params: " + str(bestParams) + "\n" + "score: " + str(score) +
                    "\nnrows: "+ str(data_preparator.myNrows))
        mfile.flush()
        mfile.close()
