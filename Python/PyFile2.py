from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
# 训练
clf = tree.DecisionTreeClassifier()
# 拟合
clf = clf.fit(X, Y)
# 预测
result = clf.predict([[2., 2.]])
print(result)
