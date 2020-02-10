import numpy as np
import csv
import sys
import math

'''
    Jeremy Herman

    Closed Form Linear Regression
'''

def standardize(data):
    number_of_rows = np.shape(data)[0]
    mean = np.mean(data[:,:], 0)
    data[:,:] = data[:,:] - np.tile(mean, (number_of_rows, 1))
    standard_deviation = np.std(data[:,:], 0)
    sd_matrix = np.tile(standard_deviation, (number_of_rows, 1))
    return np.divide(data[:,:], sd_matrix)

def insert_bias(data):
    return np.insert(data, 0, 1, 1)

def main():
    np.random.seed(0)
    filename = 'x06Simple.csv' if len(sys.argv) == 1 else sys.argv[1]

    raw_data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            raw_data.append(row)
    
    raw_data = np.array(raw_data)
    raw_data = np.array(raw_data[1:, 1:]).astype(np.int)
    size = np.shape(raw_data)
    np.random.shuffle(raw_data)

    cutoff = math.ceil((size[0]-1)*2/3)

    training_data = raw_data[:cutoff, :]
    testing_data = raw_data[cutoff:, :]
    
    theta = closed_form_linear_regression(training_data)
    apply_to_testing_data(theta, testing_data)

def closed_form_linear_regression(training_data):
    x = insert_bias(training_data[:, 1:])
    y = training_data[:, 0]

    x_transpose = np.transpose(x)

    xtx = np.matmul(x_transpose, x)
    inv = np.linalg.inv(xtx)
    invxt = np.matmul(inv, x_transpose)
    theta = np.matmul(invxt, y)

    return theta

def apply_to_testing_data(theta, testing_data):
    feats = insert_bias(testing_data[:, 1:])
    pv = np.matmul(feats, theta)

    mean_square_error = np.mean(np.square(np.subtract(testing_data[:, 0], pv)))
    rmse = np.sqrt(mean_square_error)

    print("\nCFLR:")
    print("\tModel:\ty = " + str(round(theta[0],2)) + " + " + str(round(theta[1],2)) + "*x_1 + " + str(round(theta[2],2)) + "*x_2")
    print('\tRMSE:\t' + str(round(rmse, 2)))

if __name__== "__main__":
    main()