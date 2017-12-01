import sklearn
from sklearn.metrics import roc_curve,auc,precision_recall_curve

from matplotlib import pyplot as plt

def roc_auc(model,X,y):
    probs = model(X, len(X))
    fpr, tpr, thresholds = roc_curve(y[pidx], probs[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_str = 'ROC (AUC Gain = %0.2f)' % (roc_auc - 0.5)
    plt.plot(fpr, tpr, lw=1,label=roc_str)
    plt.plot([0,1],[0,1],label="RAN CLF")
    plt.title(roc_str)
    plt.show()


def prrc_auc(model,X,y):
    probs = model(X, len(X))
    pr, rc, thresholds = precision_recall_curve(y, probs[:, 1])
    roc_auc = auc(rc, pr)
    roc_str = 'Prec vs Recall (AUC Gain = %0.2f)' % (roc_auc - np.mean(y))
    plt.plot(rc,pr, lw=1,label=roc_str)
    plt.plot([0,1],[np.mean(y),np.mean(y)],label="RAN CLF")
    plt.axis([0,1,0,1])
    plt.title(roc_str)
    plt.show()



def evaluate(model,X,y):
    yhat = model(X, len(X))
    accu = np.mean(yhat == y)
    prec = np.mean(y[yhat == 1])
    recl = np.mean(yhat[y == 1])
    f1 = 2 * prec * recl / (prec + recl)
    print("Accuracy",accu,"Precision",prec,"Recall",recl,"F1",f1)


def runTests(model, X_train, y_train, X_test, y_test):
	print("TRAIN")
	%time evaluate(model,X_train,y_train)

	print("TEST")
	%time evaluate(model,X_test,y_test)

	print("ROC AUC")
	%time roc_auc(model,X,y)

	print("PRECISION/RECALL AUC")
	%time prrc_auc(model,X,y)