


class DPForWhiteBoxMetodaSzwabe:
    def __init__(self, optimalBlackBoxClassifier, currentBlackBoxAlgorithm, normalDP):
        self.optimalBlackBoxClassifier = optimalBlackBoxClassifier
        self.currentBlackBoxAlgorithm = currentBlackBoxAlgorithm
        self.normalDP = normalDP
        self.myNrows = normalDP.myNrows
        self.X_train, self.X_test, self.y_train, self.y_test = self.getX_train_X_test_y_train_y_test()

    #Zbiór treningowy do trenowania modelu whitebox
    def getX_train_y_train(self):
        return self.X_train, self.y_train

    #zbiór testowy do testowania modelu whitebox
    def getX_test_y_test(self):
        return self.X_test, self.y_test

    def getX_train_X_test_y_train_y_test(self):

        X_train, y_train = self.normalDP.getX_train_y_train()
        X_test, y_test = self.normalDP.getX_test_y_test()

        # fragment dotyczacy przetwarzania zbioru treningowego z wykorzystaniem metody blackBoxClassifier.apply() dostepnej tylko dla GB i RF
        X_trainPrim = self.optimalBlackBoxClassifier.apply(X_train)
        X_testPrim = self.optimalBlackBoxClassifier.apply(X_test)
        # techniczna modyfikacja zbioru dla przypadku GB (niepotrzebna w przypadku RF)
        #if self.currentBlackBoxAlgorithm == "GB":
            #X_trainPrim = X_trainPrim[:, :, 0]
            #X_testPrim = X_testPrim[:, :, 0]
        # użycie niezmienionego wektora etykiet
        y_trainPrim = y_train
        # modyfikacja zbioru testowego (dla potrzeb pozniejszego testowania wyników random i grid search)
        X_test = X_testPrim
        #wektor etykiet dla zbioru testowego nie zmieniony
        return X_trainPrim, X_test, y_trainPrim, y_test