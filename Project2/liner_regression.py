import numpy as np
import matplotlib.pyplot as plt

class LinerRegression():
    def __init__(self, file_name):
        '''
        Constructor
        Input:
            file_name: file data contains data and label
        Output:
            None
        '''
        self.data, self.label = self.__read_data(file_name)
        self.data_num = self.data.shape[0]
        self.data = np.hstack((np.ones((self.data_num, 1)), self.data))
        # Add one column contains all 1 to get theta_0
    
    def __read_data(self, file_name):
        '''
        Read data from file
        Input:
            file_name: file data contains data and label
        Output:
            data: 2d matrix (data_num, vector_dimension)
            label: label of feature vector (data_num, )
        '''
        data = []
        label = []
        with open(file_name, "r") as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                elements = line.strip('\n').split(',')
                label.append(int(elements[0]))
                data.append([int(i) for i in elements[1:]])
        return np.asarray(data), np.asarray(label)
    
    def fit_opt(self):
        '''
        Get optimal weights for the liner model
        Input: 
            None
        Output:
            (x.T@x)^-1@x.T@y.T
        '''
        return np.matmul(np.matmul(np.linalg.pinv(np.matmul(self.data.T, self.data)), self.data.T), self.label.T) 
   
    def fit_gd(self, epoch = 300, rl=0.0000001):
        '''
        Get estimate weights using Gradient Descent
        Input:
            epoch: Traning iteration number
            rl: learning rate
        '''
        # Initialize weights to all zeros
        weights = np.zeros((self.data.shape[1], ))
        y = self.label
        
        costs = []
        for _ in range(epoch):
            delta_weights_sum = np.zeros(weights.shape)
            # store the sum of weights changes
            cost = 0
            # cost for current weight
            for i, vect in enumerate(self.data):
                pred = np.dot(weights, vect)
                # predicted label
                delta_weights_sum += np.dot(pred - y[i], vect)
                # weights changes for current vector
                cost += (pred - y[i])**2
                
                
            costs.append(cost / self.data.shape[0])    
            delta_weights = delta_weights_sum / self.data.shape[1]
            weights -= rl * delta_weights
            
        return weights, costs
    
    def evaluate(self, file_name, weights):
        '''
        Evaluate given weights on data from input file
        Input:
            file_name: file data contains data and label
            weights: weights of linear model
        Output:
            accuracy of model
        '''
        data, label = self.__read_data(file_name)
        data = np.hstack((np.ones((data.shape[0], 1)), data))
        pred = [1 if np.dot(vect, weights) > 0.5 else 0 for vect in data]
        # get predicted result
        correct = 0
        for gt, p in zip(label, pred):
            if gt == p:
                correct += 1
        # get correct result count
        return correct / len(label)
                
        
                
    
    
# a = np.array([1, 2, 3]).reshape((3, 1))

# lb = np.array([1, 2, 2])

lr = LinerRegression("./lr_test.csv")

weights, costs = lr.fit_gd()

print(lr.evaluate("./lr_test.csv", weights))


x, y = zip(*enumerate(costs))
plt.plot(x, y)
plt.show()