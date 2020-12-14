#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:04:00 2020

@author: root
"""
# Needed imports
import pandas as pd
import matplotlib.pyplot as plt;
from sklearn import model_selection, tree, metrics, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.io import arff
import seaborn as sns

# Load data
data = arff.loadarff("seismic-bumps.arff")


### Data preprocess:
#      split input and output dataset
#      convert string data to int
df = pd.DataFrame(data[0])
y = df["class"]
samples = len(y)
for i in range(samples):
    y.at[i] = int(y.at[i])
y = y.astype('int')
    
seismic_dict = dict()
seismic_dict["a"] = 0
seismic_dict["b"] = 1
seismic_dict["c"] = 2
seismic_dict["d"] = 3

shift_dict = dict()
shift_dict["W"] = 0
shift_dict["N"] = 1

for i in range(samples):
    df.at[i, "seismic"] = seismic_dict[df.at[i, "seismic"].decode("utf-8")]
    df.at[i, "seismoacoustic"] = seismic_dict[df.at[i, "seismoacoustic"].decode("utf-8")]
    df.at[i, "shift"] = shift_dict[df.at[i, "shift"].decode("utf-8")]
    df.at[i, "ghazard"] = seismic_dict[df.at[i, "ghazard"].decode("utf-8")]

X = df.drop(["class"], axis=1)
X["seismic"] = X["seismic"].astype("int")
X["seismoacoustic"] = X["seismoacoustic"].astype("int")
X["shift"] = X["shift"].astype("int")
X["ghazard"] = X["ghazard"].astype("int")

names = X.columns
print("Number of records: ", df.shape[0])
print("Number of attributes: ", df.shape[1])



# Partition data into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=2020)




# Classification with Logistic Regression
logreg = linear_model.LogisticRegression(solver="lbfgs", max_iter=500)
logreg.fit(X_train, y_train)
intercept_logreg = logreg.intercept_[0]
coefs_logreg = logreg.coef_[0,:]
score_train_logreg = logreg.score(X_train, y_train)
score_test_logreg = logreg.score(X_test, y_test)
y_test_pred_logreg = logreg.predict(X_test)
y_train_pred_logreg = logreg.predict(X_train)
cm_logreg_test = metrics.confusion_matrix(y_test, y_test_pred_logreg)
cm_logreg_train = metrics.confusion_matrix(y_train, y_train_pred_logreg)
yprobab_logreg = logreg.predict_proba(X_test)



# Classification with Naive Bayes
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
score_train_naive_bayes = naive_bayes_classifier.score(X_train, y_train)
score_test_naive_bayes = naive_bayes_classifier.score(X_test, y_test)
y_test_pred_naive_bayes = naive_bayes_classifier.predict(X_test)
y_train_pred_naive_bayes = naive_bayes_classifier.predict(X_train)
cm_naive_bayes_train = metrics.confusion_matrix(y_train, y_train_pred_naive_bayes)
cm_naive_bayes_test = metrics.confusion_matrix(y_test, y_test_pred_naive_bayes)
yprobab_naive_bayes = naive_bayes_classifier.predict_log_proba(X_test)



# Classification with Nearest Neighbor
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
score_train_knn = knn_classifier.score(X_train, y_train)
score_test_knn = knn_classifier.score(X_test, y_test)
y_test_pred_knn = knn_classifier.predict(X_test)
y_train_pred_knn = knn_classifier.predict(X_train)
cm_knn_train = metrics.confusion_matrix(y_train, y_train_pred_knn)
cm_knn_test = metrics.confusion_matrix(y_test, y_test_pred_knn)
yprobab_knn = knn_classifier.predict_proba(X_test)


# Classification with Neural Network
neural_classifier = MLPClassifier(hidden_layer_sizes=(16), activation='logistic', solver="adam", max_iter=5000)
neural_classifier.fit(X_train, y_train)
score_train_neural = neural_classifier.score(X_train, y_train)
score_test_neural = neural_classifier.score(X_test, y_test)
y_test_pred_neural = neural_classifier.predict(X_test)
y_train_pred_neural = neural_classifier.predict(X_train)
cm_neural_train = metrics.confusion_matrix(y_train, y_train_pred_neural)
cm_neural_test = metrics.confusion_matrix(y_test, y_test_pred_neural)
yprobab_neural = neural_classifier.predict_proba(X_test)



# Comparing the scores of the different methods
plot_logreg_score = pd.DataFrame({'type':"Logreg",'score':[score_test_logreg]})
plot_naive_bayes_score = pd.DataFrame({'type':"Naive-Bayes",'score':[score_test_naive_bayes]})
plot_knn_score = pd.DataFrame({'type':"KNNeighbors",'score':[score_test_knn]})
plot_neural_score = pd.DataFrame({'type':"Neural",'score':[score_test_neural]})
dataf = pd.concat([plot_logreg_score, plot_naive_bayes_score, plot_knn_score, plot_neural_score])
splot=sns.barplot(data=dataf, x="type", y="score", ci=None, palette=["blue", "orange", "green", "red"])
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.5f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha="center", va="center",
                   xytext=(0, 9),
                   textcoords = "offset points")
plt.title("Classification scores", fontsize=16)
plt.savefig("classification_scores.png")
plt.close()



### Making Decision trees with 'entropy' and 'gini' methods

## Entropy
crit = "entropy"
class_tree = tree.DecisionTreeClassifier(criterion=crit)

class_tree.fit(X_train, y_train)
score_entropy = class_tree.score(X_test, y_test)

fig = plt.figure(figsize = (100,20), dpi=200)
tree.plot_tree(
    class_tree, 
    feature_names=X.columns, 
    class_names=["seismic", "no_seismic"],
    filled=True, fontsize=8)
fig.savefig("seismic_tree_entropy.png")
plt.close(fig)


## Gini
crit = "gini"
class_tree_gini = DecisionTreeClassifier(criterion=crit)

class_tree_gini.fit(X_train, y_train)
score_gini = class_tree_gini.score(X_test, y_test)

fig = plt.figure(figsize = (100,20), dpi=200)
tree.plot_tree(
    class_tree_gini, 
    feature_names=names, 
    class_names=["no_seismic", "seismic"],
    filled=True, fontsize=8)
fig.savefig("seismic_tree_gini.png")
plt.close(fig)



### Comparing confusion matrices
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
classifiers = [logreg, naive_bayes_classifier, knn_classifier, neural_classifier]
cmaps = ["Blues", "Oranges", "Greens", "Reds"]
for cls, ax, cmap in zip(classifiers, axes.flatten(), cmaps):
    metrics.plot_confusion_matrix(cls, X_test, y_test, ax=ax, cmap=cmap, display_labels=["no_seismic", "seismic"])
    ax.title.set_text(type(cls).__name__)
    ax.grid(False)
plt.suptitle('Confusion matrices for test data', fontsize=16)
plt.subplots_adjust(top=0.900, bottom=0.058, left=0.000, right=0.905, hspace=0.320, wspace=0.000)
plt.grid(False)
plt.savefig("confusion_matrices.png")
plt.close()



### Drawing ROC curves

## ROC Preparation
fpr_logreg, tpr_logreg, _ = metrics.roc_curve(y_test, yprobab_logreg[:,1])
roc_auc_logreg = metrics.auc(fpr_logreg, tpr_logreg)

fpr_naive_bayes, tpr_naive_bayes, _ = metrics.roc_curve(y_test, yprobab_naive_bayes[:,1])
roc_auc_naive_bayes = metrics.auc(fpr_naive_bayes, tpr_naive_bayes)

fpr_knn, tpr_knn, _ = metrics.roc_curve(y_test, yprobab_knn[:,1])
roc_auc_knn = metrics.auc(fpr_knn, tpr_knn)

fpr_neural, tpr_neural, _ = metrics.roc_curve(y_test, yprobab_neural[:,1])
roc_auc_neural = metrics.auc(fpr_neural, tpr_neural)

## Plot ROC results
plt.figure()
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='blue',
         lw=lw, label='Logistic regression (area = %0.2f)' % roc_auc_logreg)
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='orange',
         lw=lw, label='Naive Bayes (area = %0.2f)' % roc_auc_naive_bayes)
plt.plot(fpr_knn, tpr_knn, color="green",
         lw=lw, label="NNeighbor (area = %0.2f)" % roc_auc_knn)
plt.plot(fpr_neural, tpr_neural, color="red",
         lw=lw, label="Neural (area = %0.2f)" % roc_auc_neural)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for each classification method', fontsize=16)
plt.legend(loc="lower right")
plt.grid(color="black", linestyle=":", linewidth=1)
plt.savefig("roc_curves.png")
plt.close()