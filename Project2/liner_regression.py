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
   
    def fit_gd(self, epoch = 300, learning_rate = 1e-7):
        '''
        Get estimate weights using Gradient Descent
        Input:
            epoch: Traning iteration number
            learning_rate: learning rate
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
            weights -= learning_rate * delta_weights
            
        return weights, costs
    
    def evaluate(self, file_name, weights, threshold):
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
        pred = [1 if np.dot(vect, weights) > threshold else 0 for vect in data]
        # get predicted result
        correct = 0
        for gt, p in zip(label, pred):
            if gt == p:
                correct += 1
        # get correct result count
        return correct / len(label)
                
if __name__ == '__main__':

    lr = LinerRegression("./lr_training.csv")

    # Task 1. Classifying lrdata
    
    # 1. Train a linear regression model
    b_opt = lr.fit_opt()
    
    # 2. Display the optimal coefficients (denoted by b_opt)
    
    print('Optimal Coefficients:')
    for i, b in enumerate(b_opt):
        print(f'b_{i} = {b}')
    
    # 3. Classify test data (lr_test.csv) with a threshold of 0.5
    acc = lr.evaluate('./lr_test.csv', b_opt, 0.5)
    
    # 4. Display the accuracy
    
    print(f'Accuracy for the b_opt is : {acc}')
    
    
    # Task 2. Implementation of Gradient Descent with lrdata
    
    # 1. Run "gradient descent" algorithm with the hyper-parameters
    b_est, costs = lr.fit_gd(epoch = 300, learning_rate = 1e-7)
    
    # 2. Display "Learning Curve"
    x, y = zip(*enumerate(costs))
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Lerning Curve')
    plt.plot(x, y)
    plt.show()
    
    # 3. Display the estimated coefficients (denoted by b_opt)
    
    print('Estimated Coefficients:')
    for i, b in enumerate(b_est):
        print(f'b_{i} = {b}')

    # 4. Classify test data with a threshold of 0.5
    acc = lr.evaluate("./lr_test.csv", b_est, 0.5)

    # 5. Display the accuracy
    print(f'Accuracy for the b_est is : {acc}')
    
    # 6. Display the total differences between b_opt and b_est
    diff = sum(abs(b_opt - b_est))
    print(f'Total differences between b_opt and b_est is : {diff}')

    