from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()
clf2 = KNeighborsClassifier(n_neighbors = 2)
clf3 = SGDClassifier(loss="hinge", penalty="l2")

clf = clf.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)

#X_test = [170,70,43]

scores = cross_val_score(clf, X, Y, cv=5)
scores2 = cross_val_score(clf2, X, Y, cv=5)
scores3 = cross_val_score(clf3, X, Y, cv=5)

print("Finished")
print("Scores: ", scores.mean(), scores2.mean(), scores3.mean(), " fin")

# prediction2 = clf2.predict([X_test])
# prediction = clf.predict([X_test])
# print(prediction, prediction2)
