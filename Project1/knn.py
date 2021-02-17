"""
    Source File Name: jli36kNN.py
    Author: Jiaming Li
    Data: 02/03/2021
"""

import numpy as np

class KNNClassifer():
    def __init__(self):
        pass

    def train(self,
            train_data_path, 
            eval_data_path,
            eval_data_type = 'val',
            distance_type = 1):
        """
        train the knn classifer, actually is calculating the distance matrix and
        corrospoding label martix
        """
        
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.eval_data_type = eval_data_type
        self.distance_type = distance_type
        
        self.__build_word_dict()
        
        self.train_data, self.train_label = self.__read_data(self.train_data_path, data_type = 'train')
        if self.eval_data_type == 'train':
            self.eval_data, self.eval_label = self.train_data, self.train_label
        else:
            self.eval_data, self.eval_label = self.__read_data(self.eval_data_path, data_type = self.eval_data_type)
        
        distance_matrix = self.__get_distance_matrix(self.train_data, self.eval_data)
        self.label_matrix = self.__get_label_matrix(distance_matrix)
        
        return self.train_data.shape[0]

    def __build_word_dict(self):
        """
        build a mapping from the words to numbers
        """
        
        self.word_set = set()
        # use set to store unique words
        self.label_set = set()
        # use set to store unique label
        
        with open(self.train_data_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                # skip the first line   
                line = line.strip('\n').split(",")
                for word in line[0].split(" "):
                    self.word_set.add(word)
                self.label_set.add(line[1])
        
        # add words in traning dataset to word set \ label set
        self.word_dict= {}
        self.label_dict = {}
        for i, word in enumerate(self.word_set):
            self.word_dict[word] = i + 1 
        # create a maping betwen word and index (index 0 is reserved to the unknow words)
        
        for i, label in enumerate(self.label_set):
            self.label_dict[label] = i 
        # also mapping labels to numbers
        
        self.encoding_length = len(self.word_set) + 1
        # the lenght of the encoding should be the size of the word set plus one (for the word not in the dictionary) 
        
    def __read_data(self, file_name, data_type):
        """
        read data from the .csv file to numpy array
        """
        data = []
        label = []

        with open(file_name, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                # skip the first line
                line = line.strip('\n').split(',')
                encoding = np.zeros((self.encoding_length, ))
                
                if data_type == 'test':
                    for i, word in enumerate(line[1].split(' ')):
                        if word in self.word_set:
                            encoding[self.word_dict[word]] = 1
                        else:
                            encoding[0] = 1 
                            # index 0 for the world do not exist in tranning set
                    data.append(encoding)
                    # test dataset doesn't have label
                else:
                    for i, word in enumerate(line[0].split(' ')):
                        if word in self.word_set:
                            encoding[self.word_dict[word]] = 1
                        else:
                            encoding[0] = 1 
                    data.append(encoding)
                    label.append(self.label_dict[line[1]])
        
        # return numpy array of the sensence encoding and list of label        
        return np.stack(data), label 

    def __get_distance_matrix(self,
                            train_data,
                            eval_data):
        '''
        calculate the distance matrix of train and eval data
        rows: eval data
        cols: train data
        elements: distances
        '''
        def get_distance(v1, v2):
            '''
            calculate the L1 or L2 distance between two vectors
            '''
            return np.linalg.norm(v1-v2, ord = self.distance_type)

        distance_matrix = np.zeros((eval_data.shape[0], train_data.shape[0]))
        for i, eval_data_vector in enumerate(eval_data):
            for j, train_data_vector in enumerate(self.train_data):
                if self.eval_data_type == 'train' and i == j:
                    distance_matrix[i, j] = float('inf')
                    # set the distance of itself as inf when the evaluation set type is 'train'
                else:
                    distance_matrix[i, j] = get_distance(eval_data_vector, train_data_vector)
        
        return distance_matrix
        
    def __get_label_matrix(self, distance_matrix):
        """
        compute label matrix
        label matrix:
            row: eval data
            col: order
            elements: labels (each row is sorted by distance)
        """

        label_matrix = np.zeros(distance_matrix.shape)
        for i, distance in enumerate(distance_matrix):
            index_order = np.argsort(distance, kind='mergesort') 
            
            # sort the indcies of distance by its value e.g. The index of
            # closest setence's will be in index_order[0], the second closet
            # will be in index_order[1]
            
            label_matrix[i] = np.array([self.train_label[i] for i in index_order]) # fill out the label matrix with index
        return label_matrix
    
    def predict(self, k):
        """
        make prediction on eval data
        """
        
        self.pred_label = []
        self.k = k
        for label_vector in self.label_matrix:
            k_closest = label_vector[:k]
            # the labels of first k closest sentences
            values, counts = np.unique(k_closest, return_counts=True)
            # get valuses and counts of unique label
            ind = np.argmax(counts)
            self.pred_label.append(values[ind])
            # the predicted label will be the one most frequent
            
        return self.pred_label
        
    def evaluate(self):
        """
        compute correct rate
        """
        
        total = len(self.eval_label)
        correct = 0
        
        for gt, pred in zip(self.eval_label, self.pred_label):
            if gt == pred:
                correct += 1
        # compare ground truth labels and predicted labels
        return correct / total           
       
    def save_result(self):
        '''
        save the result
        '''
        inx_to_label = dict(zip(self.label_dict.values(), self.label_dict.keys()))
        with open(self.eval_data_path, "r") as f:
            strings = f'k = {self.k}\n{self.distance_type}\n'
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    strings += line
                    continue
                line = line.strip('\n')
                line += inx_to_label[self.pred_label[i-1]]
                line += '\n'
                strings += line
                
        with open('my_result.csv', "w+") as f:
            f.write(strings)


if __name__ == "__main__":

    
    train_data_path = 'train_set.csv'
    val_data_path = 'validation_set.csv'
    test_data_path = 'test_set.csv'

    
    knn = KNNClassifer()
    
    opt_k = 1
    opt_acc = 0

    
    for eval_data_type in ('val', 'train'):
        data_num = knn.train(train_data_path, val_data_path, eval_data_type)
        for k in range(3, int(data_num**0.5)):
            knn.predict(k=k)
            acc = knn.evaluate()
            if eval_data_type == 'val' and acc > opt_acc:
                opt_k = k
                opt_acc = acc               
            print(f'K = {k}, {eval_data_type} set correct rate: {acc}')

    print(f'K = {opt_k} yeilds the optimal validation set correct rate: {opt_acc}')
    
    knn.train(train_data_path, test_data_path, 'test')
    knn.predict(k=opt_k)
    knn.save_result()
            
    
    
    