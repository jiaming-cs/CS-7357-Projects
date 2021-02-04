"""
    Source File Name: jli36kNN.py
    Author: Jiaming Li
    Data: 02/03/2021
"""

import argparse
import os

import numpy as np

class KNNClassifer():
    def __init__(self,
                train_data_path,
                val_data_path, 
                test_data_path, 
                distance_type):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.distance_type = distance_type
        self.build_word_dict()
        self.train_data, self.train_label = self.read_data(self.train_data_path)
        self.val_data, self.val_label = self.read_data(self.val_data_path)
        self.test_data, self.test_label = self.read_data(self.test_data_path)
        
        

    def build_word_dict(self):
        def add_words(file_name, word_set, label_set):
            with open(file_name, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    # skip the first line
                    line = line.split(",")
                    for word in line[0].split(" "):
                        word_set.add(word)
                    label_set.add(line[1])
        
        self.word_set = set()
        # use set to store unique words
        self.label_set = set()
        # use set to store unique label
        
        add_words(self.train_data_path, self.word_set, self.label_set)
        # add words in traning dataset to word set
        
        self.word_dict= {}
        self.label_dict = {}
        for i, word in enumerate(self.word_set):
            self.word_dict[word] = i + 1 
        # create a maping betwen word and index (index 0 is reserved to the unknow words)
        
        for i, label in enumerate(self.label_set):
            self.label_dict[label] = i 
        
        
        
        self.encoding_length = len(set) + 1
        # the lenght of the encoding should be the size of the word set plus one (for the word not in the dictionary) 
        
        
        
    
    
    def read_data(self, file_name):
        """
        read data from csv file to numpy array
        """
        data = []
        label = []

        with open(self.file_name, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                # skip the first line
                line = line.split(",")
                encoding = np.zeros((self.encoding_length, ))
                for i, word in enumerate(line[0].split(" ")):
                    if word in self.word_set:
                        encoding[i] = self.word_dict[word]
                data.append(encoding)
                label.append(line[1])
        
        # return numpy array of the sensence encoding and list of label        
        return np.stack(data), label



    def get_distance_matrix(self,
                            train_data,
                            eval_data,
                            distance_type):
        '''
        calculate the distance matrix of train and test data
        '''
        def get_distance(v1, v2):
            '''
            calculate the L1 or L2 distance between two vectors
            '''
            return np.linalg.norm(v1-v2, ord = distance_type)

        distance_matrix = np.zeros((eval_data.shape[0], train_data.shape[0]))
        for i, eval_data_vector in enumerate(eval_data):
            for j, train_data_vector in enumerate(self.train_data):
                distance_matrix[i, j] = get_distance(eval_data_vector, train_data_vector, distance_type)
        return distance_matrix
        
    def get_label_matrix(self, distance_matrix):
        """
        sort every rows of distance matrix
        """

        distance_matrix_ordered = np.zeros(distance_matrix.shape)
        label_matrix = np.zeros(distance_matrix.shape)
        for i, distance in enumerate(self.distance_matrix):
            self.distance_matrix_ordered[i] = np.sort(distance)
            index_order = np.argsort(distance) # get the index change after change so that get the label after sorting
            self.label_matrix[i] = np.array([self.label_dict[i] for i in index_order])
        return label_matrix
        
    def train(self, train_data, eval_data):
        """
        train the knn classifer, actually is get the distance matrix
        """
        distance_matrix = self.get_distance_matrix(train_data, eval_data)
        label_matrix = self.get_label_matrix(distance_matrix)
        
        return label_matrix
        

    def predict(self, label_matrix, k):
        """
        make prediction on test data
        """
        pred_label = []
        
        for label_vector in label_matrix:
            k_closest = label_vector[:k]
            values, counts = np.unique(k_closest, return_counts=True)
            ind = np.argmax(counts)
            pred_label.append(values[ind])
            
        return np.array(pred_label)    
        
        

    # def get_confusion_matrix(self):
    #     """
    #     compute the confusion matrix for the prediction result
    #     """

    #     if self.pred_label is None:
    #         print("Call method predict first.")
    #         return 
    #     self.confusion_matrix = np.zeros((2, 2))
    #     for i in range(self.test_data_num):
    #         true = self.test_label[i]
    #         pred = self.pred_label[i]
    #         if true == -1 and pred == -1:
    #             self.confusion_matrix[0, 0] += 1
    #         elif true == -1 and pred == 1:
    #             self.confusion_matrix[0, 1] += 1
    #         elif true == 0 and pred == 0:
    #             self.confusion_matrix[1, 0] += 1
    #         else:
    #             self.confusion_matrix[1, 1] += 1
    #     return self.confusion_matrix

    def save_result(self):
        '''
        save the result
        '''
        def matrix_to_str(matrix):
            out_str = ""
            for row in matrix:
                for i, n in enumerate(row):
                    if i==0:
                        out_str += str(n)
                    else:
                        out_str += ","+str(n)
                out_str += "\n"
            return out_str
        with open("jli36{}NNEvaluate{}{}".format(self.k, self.percentage, self.input_file), "w+") as f:
            f.write(matrix_to_str(self.distance_matrix))
            f.write("\n")
            f.write(matrix_to_str(self.distance_matrix_ordered))
            f.write("\n")
            f.write(matrix_to_str(self.label_matrix))

        with open("jli36{}nnApply{}{}".format(self.k, self.percentage, self.input_file), "w+") as f:
            strings = ''
            for i, row in enumerate(self.test_data):
                for j, n in enumerate(row):
                    if j==0:
                        strings += str(n)
                    else:
                        strings += ","+str(n)
                strings += ","+str(self.pred_label[i])+"\n"
            f.write(strings)


if __name__ == "__main__":
    paser = argparse.ArgumentParser()

    paser.add_argument("-i", "--input_file", type=str, default="wdbc.data.mb.csv", help="input file name")
    paser.add_argument("-k", "--k_neighbors", type=int, default=3, help="number of nearnest neighbors")
    paser.add_argument("-c", "--class_attribute", type=int, default=31, help="column number of target feature")
    paser.add_argument("-P", "--percentage", type=int, default=80, help="percentage of samples to be used for training")
    paser.add_argument("--normalize", action="store_true", default=False, help="Whether normalize the data before processing")

    args = paser.parse_args()

    if not os.path.exists(args.input_file):
        print("Can't find data file!")
        exit(-1)

    if args.k_neighbors < 0 or args.k_neighbors % 2 != 1:
        print("k has to be a positive odd number")
    
    
    knn = KNNClassifer(args.input_file, args.percentage, args.k_neighbors, args.class_attribute - 1, args.normalize)
    knn.train()
    knn.predict()
    print("Confusion Matrix")
    print(knn.get_confusion_matrix())
    knn.save_result()