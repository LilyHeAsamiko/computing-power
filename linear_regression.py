import numpy as np
import sklearn
class linear_regression():
    def __init__(self,learning_rate= 0.01, n = 1000):
        self.learning_rate = learning_rate
        self.n = n


    def gradient(self):
        return ((1/self.m*(self.X.dot(self.w[:self.X.shape[1]].reshape(-1,1))-self.Y.reshape(-1,1))).reshape(1,-1)).dot(self.X)
    
    def renew(self):
        self.w = self.w - self.gradient()*self.learning_rate

    def costfun(self):
        return 1/(2*self.m)*((self.X.dot(self.w[:self.X.shape[1]].reshape(-1,1))-self.Y[:self.w.shape[0]].reshape(-1,1))**2).sum()

    def fit(self,X,Y):
 #       self.w = np.array([0 for x in range(X.shape[1]+1)])
        self.w = np.array([0 for x in range(X.shape[1])])
        self.m = X.shape[0]
 #       self.X = np.c_[X,[1 for x in range(self.m)]]
        self.X = X
        self.Y = Y
        cost1 = self.costfun()
        cost2 = 0
        for i in range(self.n):
            if cost1 - cost2 <= 1e-6:
                break
            else:
                cost1 = self.costfun()
                self.renew()
                cost2 = self.costfun()

    def predict(self,xt):
        xt = np.c_[xt, [1 for x in range(xt.shape[0])]]
        return xt.dot(self.w.T)




