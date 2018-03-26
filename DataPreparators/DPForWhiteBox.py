import numpy as np

class DPForWhiteBox:
    def __init__(self, optimalBlackBoxClassifier, X_train, y_train):
        self.optimalBlackBoxClassifier = optimalBlackBoxClassifier
        self.X_train = X_train
        self.y_train = y_train

    def getXy(self):
        probabilities = self.optimalBlackBoxClassifier.predict_proba(self.X_train)[:, 0]
        #liczba próbek „pozytywnych” w nowym zbiorze etykiet  y_trainPrim  ma być równa
        #liczbie próbek „negatywnych” w oryginalnym zbiorze y_train
        bestProbas = np.argsort(probabilities)[:int(np.sum( self.y_train))]
        y_trainPrim = np.zeros(len( self.y_train))

        for i in bestProbas:
            y_trainPrim[i] = 1
        #X nie zmieniamy
        X_trainPrim = self.X_train
        return X_trainPrim, y_trainPrim