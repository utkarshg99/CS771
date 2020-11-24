import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

path = './binclassv2.txt'
data_file = open(path, 'r')
lns = data_file.readlines()
X=[]
Y=[]
for ln in lns:
    vals = ln.split(',')
    X.append([])
    for v in vals:
        X[-1].append(float(v))
    X[-1].pop()
    Y.append(int(vals[-1]))

X = np.array(X)
Y = np.array(Y)
N = len(Y)      # Number of Inputs
D = X.shape[1]  # Dimensions of Input Vector
K = 2           # Number of Classes

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Generate Prediction for Generative Classification Model (+1/-1: More probability => Belonging to that class)
def boundary(X, mu, sisq):
    Y=[]
    for x in X:
        p1 = np.exp(-np.matmul(np.transpose(x-mu[0]), x-mu[0])/(2*sisq[0]))/sisq[0]
        p2 = np.exp(-np.matmul(np.transpose(x-mu[1]), x-mu[1])/(2*sisq[1]))/sisq[1]
        Y.append(p1-p2)
    return Y

# Plots the decision boundary
def plot(Z):
    Z = np.array(Z).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.scatter(x=np.transpose(X)[0], y=np.transpose(X)[1], s=4, c=["#ff0000" if y == 1 else "#0000ff" for y in Y])
    plt.show()

def exp1():
    # MLE of Mean
    mu = np.zeros((K, D))
    # Sigma^2
    sisq = np.zeros(K)
    Xseg = [[] for _ in range(K)]

    i = 0

    #Seperate the examples of the 2 classes
    for y in Y:
        if y == 1:
            Xseg[0].append(X[i])
        elif y == -1:
            Xseg[1].append(X[i])
        i+=1

    # Estimate the mean and covariance(s)
    for i in range(K):
        mu[i] = np.sum(Xseg[i], 0)/len(Xseg[i])
        for x in Xseg[i]:
            sisq[i] += np.matmul(np.transpose(x-mu[i]), (x-mu[i]))
        sisq[i] = sisq[i]/(D*len(Xseg[i]))
        Xseg[i] = np.transpose(Xseg[i])
    Z = boundary(np.c_[xx.ravel(), yy.ravel()], mu, sisq)
    plot(Z)
    print(sisq)

def exp2():
    # MLE of Mean
    mu = np.zeros((K, D))
    # Sigma^2
    sisq = np.zeros(K)
    Xseg = [[] for _ in range(K)]

    i = 0

    #Seperate the examples of the 2 classes
    for y in Y:
        if y == 1:
            Xseg[0].append(X[i])
        elif y == -1:
            Xseg[1].append(X[i])
        i+=1

    # Estimate the mean and covariance(s)
    for i in range(K):
        mu[i] = np.sum(Xseg[i], 0)/len(Xseg[i])
        for x in Xseg[i]:
            sisq[0] += np.matmul(np.transpose(x-mu[i]), (x-mu[i]))
        Xseg[i] = np.transpose(Xseg[i])
    sisq[0]/=(D*N)
    for i in range(K):
        sisq[i] = sisq[0]
    
    Z = boundary(np.c_[xx.ravel(), yy.ravel()], mu, sisq)
    plot(Z)
    print(sisq)

def exp3():
    # The SVM Part
    model = SVC(kernel='linear', C=1)
    clf = model.fit(X, Y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Plotting the data points with the support vectors and decision boundaries
    Z = np.array(Z).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.scatter(x=np.transpose(X)[0], y=np.transpose(X)[1], s=4, c=["#ff0000" if y == 1 else "#0000ff" for y in Y])
    plt.show()

# Generative Classification with different Covariance Martices
exp1()
# Generative Classification with same Covariance Martices
exp2()
# SVM Part
exp3()