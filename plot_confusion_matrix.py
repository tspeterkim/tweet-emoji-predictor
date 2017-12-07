from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import ast

input_file = open('./out/lstm_best')
ground_truth = input_file.readline()
predict = input_file.readline()
results = input_file.readline()
input_file.close()

ground_truth = ast.literal_eval(ground_truth)
predict = ast.literal_eval(predict)
print(ground_truth)
print(predict)

classes = []
for i in range(20):
    classes.append('Emoji ' + str(i))

cnf_matrix = confusion_matrix(ground_truth, predict)
plt.figure()
plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(20)
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()
