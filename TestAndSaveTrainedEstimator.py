import pickle
from sklearn.metrics import roc_curve, auc



def TestAndSaveTrainedEstimator(theEstimator, DPforTheEstimator, info=""):
    # testowanie wytrenowanego modelu
    X_test, y_test = DPforTheEstimator.getX_test_y_test()

    y_probs = theEstimator.predict_proba(X_test)
    y_pred = y_probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print('{0} AUC: {1}'.format(type(theEstimator).__name__, roc_auc))

    filename = str(DPforTheEstimator.myNrows) + "Tested"+ info + type(theEstimator).__name__ + type(DPforTheEstimator).__name__
    # zapisywanie parametrów i wyniku
    if info == "WB":
        filename = filename + type(DPforTheEstimator.optimalBlackBoxClassifier).__name__
    mfile = open(file=filename + "_testResults", mode="w")
    if info == "WB":
        mfile.write("\nWhiteBox class name: {}".format(type(theEstimator).__name__))
        mfile.write("\nUsed BlackBox: {}\n".format(type(DPforTheEstimator.optimalBlackBoxClassifier).__name__))
        mfile.write("\nBlackBoxDP: {}\n".format(type(DPforTheEstimator.normalDP).__name__))

    mfile.write("params: " + str(theEstimator.get_params()) + "\n" +
                "test result(roc_auc): " + str(roc_auc) + "\nDP: " + type(DPforTheEstimator).__name__ +
                "\nnrows: " + str(DPforTheEstimator.myNrows) + "\ninfo: " + info)
    mfile.flush()
    mfile.close()
    # zapisywanie modelu:
    pickle.dump(theEstimator, open(file=filename, mode='wb'))
