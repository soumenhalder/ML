
# https://www.kaggle.com/code/residentmario/gradient-descent-with-linear-regression
import numpy as np

class GradientDescentLinearRegression_online:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations
    
    def fit(self, X, y):
        b = 0
        m = 1
        n = X.shape[0]
        for _ in range(self.iterations):
            b_gradient = -2 * np.sum(y - m*X + b) / n       # there should be `y-(m*X +b)`
            m_gradient = -2 * np.sum(X*(y - (m*X + b))) / n 
            b = b + (self.learning_rate * b_gradient)       # should be -ve sign
            m = m - (self.learning_rate * m_gradient)
        self.m, self.b = m, b
        
    def predict(self, X):
        return self.m*X + self.b

class GradientDescentLinearRegression_modified:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations

    def fit(self, X, y):
        b = 0
        m = 1
        n = X.shape[0]
        for _ in range(self.iterations):
            b_gradient = -2 * np.sum(y - (m*X + b)) / n
            m_gradient = -2 * np.sum(X*(y - (m*X + b))) / n
            b = b - (self.learning_rate * b_gradient)
            m = m - (self.learning_rate * m_gradient)
            #print(b,m)
        self.m, self.b = m, b

    def predict(self, X):
        return self.m*X + self.b

class GradientDescentLinearRegression_sh:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations
    
    def fit(self, X, y):
        b = 0
        m = 1
        Theta = np.array([b,m])        
        n = X.shape[0]
        
        X = np.hstack((np.ones((n,1)),X.reshape(n,1)))
        for _ in range(self.iterations):
            xth_y = (X.dot(Theta)-Y)
            
            J = 1/(2*n) * xth_y.T.dot(xth_y)
            #print('Iteratation %s'%(_),Theta,J)
            
            deltaTheta = self.learning_rate/n* X.T.dot(xth_y)
            Theta = Theta - deltaTheta
        
        self.m, self.b = Theta[1], Theta[0] 
        
    def predict(self, X):
        return self.m*X + self.b

def find_theta(X,y):
    #  usin x0=1
    m = df.shape[0]
    X = np.hstack((np.ones((m,1)),X.reshape(m,1)))
    
    #return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    ##  use pseudo inverse
    return np.linalg.pinv(X).dot(y)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('linear-regression.csv')
    X = df['x'].to_numpy()
    Y = df['y'].to_numpy()
    
    ## online simplest gradiant decent
    clf = GradientDescentLinearRegression_online(learning_rate=0.0001, iterations=10000)
    clf.fit(X, Y)
    print('Online(raw) simplest gradiant decent',clf.m,clf.b)

    ## online simplest gradiant decent
    clf = GradientDescentLinearRegression_modified(learning_rate=0.0001, iterations=10000)
    clf.fit(X, Y)
    print('Online(corrected) simplest gradiant decent',clf.m,clf.b)

    ## my simplest gradiant decent
    clf = GradientDescentLinearRegression_sh(learning_rate=0.0001, iterations=10000)
    clf.fit(X, Y)
    print('my simplest gradiant decent',clf.m,clf.b)

    ## exact solution, normal eqn
    a  = find_theta(X,Y)
    print('Exact solution',a[1],a[0])

    # sklearn
    from sklearn.linear_model import LinearRegression
    mod_lr = LinearRegression()
    mod_lr.fit(df['x'].to_numpy().reshape(df.shape[0],1),df['y'].to_numpy())
    a1,a0 = mod_lr.coef_[0],mod_lr.intercept_
    print('Sklearn',a1,a0)
