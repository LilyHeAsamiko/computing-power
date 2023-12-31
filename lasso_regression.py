import numpy as np
import sklearn
class lasso_regression():
    def __init__(self,learning_rate= 0.01, n = 1000, L1 = 0.9):
        self.learning_rate = learning_rate
        self.n = n
        self.L1 = L1

    def coordinate(self):
        for i in range(self.num + 1):
            a = (self.X[:,i].T.dot(*self.X[:,i].reshape(1,-1)))
            dw = np.matrix(np.zeros((1,self.num +1)))
            dw[0,i] = self.w[i]
            b = 2*(self.X.T*(self.Y.reshape(-1,1)-self.X.dot((self.w-dw).T)))[i,0]
            if b <-self.L1:
                self.w[i] =(-self.L1-b)/a/2
            elif b > self.L1:
                self.w[i] = (self.L1-b)/a/2
            else:
                self.w[i] = 0
        self.w = self.w - self.gradient()*self.learning_rate

    def costfun(self):
        return 1/(2*self.m)*((self.X.dot(self.w.reshape(-1,1))-self.Y.reshape(-1,1))**2).sum() + self.L1*(abs(self.w)).sum()

    def fit(self,X,Y):
        self.w = np.array([0 for x in range(X.shape[1])])
        self.m = X.shape[0]
        self.X = np.c_[X,[1 for x in range(self.m)]]
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

