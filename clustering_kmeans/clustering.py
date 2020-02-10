import numpy as np
import csv
import sys
from myKMeans import myKMeans

'''
    Jeremy Herman

    Clustering via k-means

    This file contains:
        - main driver for the program
        - a function to standardize the data
'''


# Standardizes the data in a matrix
def standardize(data):
    number_of_rows = np.shape(data)[0]
    mean = np.mean(data[:,:], 0)
    data[:,:] = data[:,:] - np.tile(mean, (number_of_rows, 1))
    standard_deviation = np.std(data[:,:], 0)
    sd_matrix = np.tile(standard_deviation, (number_of_rows, 1))
    return np.divide(data[:,:], sd_matrix)


def main():
    # Read first arg as filename, otherwise default to 'diabetes.csv'
    filename = 'diabetes.csv' if len(sys.argv) == 1 else sys.argv[1]

    # Read raw data from csv file
    raw_data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            raw_data.append(row)
        
    # create numpy matrix
    raw_data = np.array(raw_data).astype(np.double)

    # Separate class labels from features
    labels = []
    observable_data = []
    for row in raw_data:
        labels.append(row[0])
        observable_data.append(row[1:])

    observable_data = np.array(observable_data).astype(np.double)
    standardized_features = standardize(observable_data)

    # execute myKMeans
    np.random.seed(0)
    myKMeans(standardized_features, 2)
  
if __name__== "__main__":
    main()

