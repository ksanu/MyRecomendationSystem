import numpy as np

class DPForWhiteBoxMetodaPedagogiczna:
    def __init__(self, optimalBlackBoxClassifier, X_train, y_train, X_test, y_test, nrows=None):
        self.optimalBlackBoxClassifier = optimalBlackBoxClassifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.myNrows = nrows

    #dane treningowe do trenowanie modelu whitebox
    def getX_test_y_test(self):
        return self.X_test, self.y_test

    #dane treningowe do trenowanie modelu whitebox
    def getX_train_y_train(self):
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