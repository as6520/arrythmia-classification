"""
File: ecg_preprocess.py
Author: Ameya Shringi as6520@g.rit.edu
        Vishal Garg
Description: Pre-processing involved to convert ecg
             signal to feature vectors
"""
import numpy as np
import os
from scipy.signal import resample
from ecg_frame import ecg_frame
import cPickle as pickle

max_number_of_features = 256
number_of_classes = 7

class ecg_preprocess:
    __slots__ = 'data_path'

    def __init__(self, data_path):
        """
        Constructor of the class
        :param data_path:
        :return:
        """
        self.data_path = data_path
        self.max_beat_value = 0

    def create_Dataset(self):
        """
        Resampling data to create feature vector from the datasets
        :return: None
        """
        # Initialize the variables
        dataset = []
        number_normal_beats = 0
        number_pvc = 0
        number_of_fusion = 0
        number_of_unclassified = 0
        number_apb = 0
        number_rbr = 0
        number_lbr = 0
        # Read All the file in the dataset in different folder
        for root, dir, files in os.walk(self.data_path):
            if len(files) is not 0:
                file_attribute = None
                file_data = None
                # Read Attribute and data file
                if files[0].endswith("rr1.csv"):
                    file_attribute = np.loadtxt(root + "/" + files[0], dtype=object)
                    file_data = np.loadtxt(root + "/" + files[1])
                else:
                    file_attribute = np.loadtxt(root + "/" + files[1], dtype=object)
                    file_data = np.loadtxt(root + "/" + files[0])
                # Create signal frame
                current_last = 0
                signal_frame = ecg_frame(len(file_attribute))
                one_hot_frame = np.zeros([len(file_attribute), number_of_classes])
                index = 0
                # Add Attribute to the frame
                for line in file_attribute:
                    number_of_datapoints = int(line[0])
                    prev = current_last
                    current_last += number_of_datapoints
                    raw_data = file_data[prev:current_last, 1:3]
                    resample_data = resample(raw_data, max_number_of_features)
                    if line[1] == 'N':
                        one_hot_frame[index, 0] = 1
                        number_normal_beats += 1
                    elif line[1] == 'V':
                        one_hot_frame[index, 1] = 1
                        number_pvc += 1
                    elif line[1] == 'F':
                        one_hot_frame[index, 2] = 1
                        number_of_fusion += 1
                    elif line[1] == 'A':
                        one_hot_frame[index, 3] = 1
                        number_apb += 1
                    elif line[1] == 'R':
                        one_hot_frame[index, 4] = 1
                        number_rbr += 1
                    elif line[1] == 'L':
                        one_hot_frame[index, 5] = 1
                        number_lbr += 1
                    else:
                        one_hot_frame[index, 6] = 1
                        number_of_unclassified += 1
                    signal_frame.create_beat(resample_data, one_hot_frame[index, :], index)
                    index += 1
                # Create the Dataset
                print "ecg frame created"
                dataset.append(signal_frame)
        # Summarize the dataset
        print "Number of normal beats: " + str(number_normal_beats)
        print "Number of premature ventricular contraction: " + str(number_pvc)
        print "Number of fusion: " + str(number_of_fusion)
        print "Number of unclassified beats: " + str(number_of_unclassified)
        print "Number of Atrial premature beat: " + str(number_apb)
        print "Number of Right bundle beats: " + str(number_rbr)
        print "Number of Left bundle beats: " + str(number_lbr)
        with open('ecg-data-resampled.pkl', mode='w') as ecg_pkl:
            pickle.dump(dataset, ecg_pkl)


def main():
    ecg = ecg_preprocess('csv-data')
    ecg.create_Dataset()


if __name__=="__main__":
    main()
