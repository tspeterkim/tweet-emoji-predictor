import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



ground_truth = []
pred_labels = []

with open('english.output2.txt', 'r') as predictions, open('data/us_trial.labels', 'r') as gnd_truth:
    for p in predictions:
        pred_labels.append(p[:-1])
        
    for g in gnd_truth:
#        print(g[-1:])
        if(g[-1:] == '\n'):
            ground_truth.append(g[:-1])
        else:
            ground_truth.append(g)
    

pred_np = np.array(pred_labels).astype(int)
#print(ground_truth)
gnd_np = np.array(ground_truth).astype(int)
#gnd_np = gnd_np[:-1]

print(pred_np)
print(pred_np.size)
print(gnd_np)
print(gnd_np.size)

classes = []
for i in range(20):
    classes.append('Emoji ' + str(i))

    
cnf_matrix = confusion_matrix(gnd_np, pred_np)
# Plot normalized confusion matrix
plt.figure()
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()