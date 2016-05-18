"""
File: MakeTrainSet.py
Author: Ameya Shringi(as6520@g.rit.edu)
        Vishal Garg
Description: Creating Training and Testing data
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.utils import shuffle

data_csv_file = "ecg_data.csv"
attribute_csv_file = "ecg_attribute.csv"
new_train_data = 'train_data.csv'
new_train_attribute = 'train_attribute.csv'
new_test_data = 'test_data.csv'
new_test_attribute = 'test_attribute.csv'
min_value = 750


def makeTrainSet(ecg_data, ecg_attribute):
    """
    Create the train and test set
    :param ecg_data: ecg data matrix
    :param ecg_attribute: ecg attribute matrix
    :return: None
    """
    # Scale the Data
    ecg_data = scale(ecg_data)
    # Initialize the variables
    new_train_data_set = np.empty((0, ecg_data.shape[1]))
    new_train_attribute_set = np.empty((0, ecg_attribute.shape[1]))
    new_test_data_set = np.empty((0, ecg_data.shape[1]))
    new_test_attribute_set = np.empty((0, ecg_attribute.shape[1]))
    # Create test and training dataset
    for i in range(ecg_attribute.shape[1]):
        attribute_bool = ecg_attribute[:, i] == 1
        temp_data = ecg_data[attribute_bool, :]
        temp_attribute = ecg_attribute[attribute_bool, :]
        train_data, test_data, train_attribute, test_attribute\
            = train_test_split(temp_data, temp_attribute, train_size=min_value, random_state=105)
        new_train_data_set = np.vstack((new_train_data_set, train_data))
        new_train_attribute_set = np.vstack((new_train_attribute_set, train_attribute))
        new_test_data_set = np.vstack((new_test_data_set, test_data))
        new_test_attribute_set = np.vstack((new_test_attribute_set, test_attribute))

    # Shuffle the data
    new_train_data_set, new_train_attribute_set= \
        shuffle(new_train_data_set, new_train_attribute_set, random_state=201)
    # Save the data
    np.savetxt(new_train_data, new_train_data_set)
    np.savetxt(new_train_attribute, new_train_attribute_set)
    np.savetxt(new_test_data, new_test_data_set)
    np.savetxt(new_test_attribute, new_test_attribute_set)


def main():
    """
    Main Function
    :return: None
    """
    ecg_data = np.loadtxt(data_csv_file)
    ecg_attribute = np.loadtxt(attribute_csv_file)
    makeTrainSet(ecg_data, ecg_attribute)

if __name__=="__main__":
    main()