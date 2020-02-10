import numpy as np
import csv
import sys
from itertools import islice
from itertools import chain

'''
    Jeremy Herman
    Machine Learning – Assignment 3
    Part 4 - S-Folds
'''

def insert_bias(data):
    return np.insert(data, 0, 1, 1)

def main():
    np.random.seed(0)
    filename = 'x06Simple.csv' if len(sys.argv) == 2 else sys.argv[2]
    s_arg = sys.argv[1]

    raw_data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            raw_data.append(row)
    
    raw_data = np.array(raw_data)
    raw_data = np.array(raw_data[1:, 1:]).astype(np.int)

    if s_arg == 'n':
        s_num = np.shape(raw_data)[0] + 1
    else:
        s_num = int(s_arg)

    rmse_list = []

    for _ in range(20): # will be range 20
        squared_errors = []
        np.random.shuffle(raw_data)
        folds_data = get_s_folds(raw_data, s_num)
        for j in range(s_num): # will be range s_num
            testing_data = folds_data[j]
            training_data = get_training_data(folds_data, j, s_num)
            theta = closed_form_linear_regression(training_data)
            squared_errors.append(get_square_errors(theta, testing_data))

        flat_squared_errors = list(chain.from_iterable(squared_errors))
        rmse_list.append(compute_rmse(flat_squared_errors))
    
    rmse_list = np.array(rmse_list)
    mean = np.mean(rmse_list)
    standard_deviation = np.std(rmse_list, 0)
    print('\nS-Folds, S =', str(s_num), ':')
    print('\tMean:\t', round(mean, 3))
    print('\tSD:\t', round(standard_deviation, 3))


def get_s_folds(data, s):
    return np.array_split(data, s)

def get_training_data(data, i, s_num):
    data = np.delete(data, i, 0)
    training_data = data[0]
    for arr in islice(data, 1, None):
        training_data = np.concatenate((training_data, arr))
    return training_data

def closed_form_linear_regression(training_data):
    x = insert_bias(training_data[:, 1:])
    y = training_data[:, 0]

    x_transpose = np.transpose(x)

    xtx = np.matmul(x_transpose, x)
    inv = np.linalg.inv(xtx)
    invxt = np.matmul(inv, x_transpose)
    theta = np.matmul(invxt, y)

    return theta

def get_square_errors(theta, testing_data):
    feats = feats = insert_bias(testing_data[:, 1:])
    pv = np.matmul(feats, theta)
    return np.square(np.subtract(testing_data[:, 0], pv))

def compute_rmse(errors):
    e = np.array(errors)
    mse = np.mean(e)
    rmse = np.sqrt(mse)
    return rmse

if __name__== "__main__":
    main()