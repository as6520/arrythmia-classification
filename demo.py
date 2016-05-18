import numpy as np
from scipy.signal import resample
from tensorflow.contrib import skflow

file_data = 'csv-data/101/101.csv'
file_attribute = 'csv-data/101/101.rr1.csv'
number_of_complex = 100
number_of_features = 256

def test():
    test_data = np.loadtxt(file_data)[:, 1:3]
    test_attribute = np.loadtxt(file_attribute, dtype=object)
    qrs_data = []
    qrs_resampled = []
    inital_counter = 0
    for i in range(number_of_complex):
        signal_length = int(test_attribute[i, 0])
        temp_data = test_data[inital_counter:
                        inital_counter+signal_length, :]
        qrs_data.append(temp_data)
        qrs_resampled.append(resample(temp_data, 256).flatten())
    classifier = None
    #with open('svmclassifier.pkl','r') as f:
    #    classifier = pkl.load(f)
    classifier = skflow.TensorFlowEstimator.restore('dnn')
    print test_attribute[0:12]
    print classifier.predict(np.asarray(qrs_resampled[0:45]))




def main():
    test()

if __name__ == '__main__':
    main()