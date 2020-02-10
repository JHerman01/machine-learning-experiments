import numpy as np
import csv
import sys

def split_data(data):
    spam = []
    not_spam = []
    for row in data:
        if row[-1] == 0:
            not_spam.append(row)
        elif row[-1] == 1:
            spam.append(row)
    return [np.array(spam)[:, :-1], np.array(not_spam)[:, :-1]]

def my_gaussian_pdf(mean, std, n):
    # Just return 1 if the standard deviation is 0 so we don't divide by 0
    if std == 0:
        return 1
    return np.exp(-1 * (np.square(n-mean) / (2 * np.square(std)))) * (1 / (np.sqrt(2*np.pi) * std))

def get_norm(data):
    norm = []
    for attribute in zip(*data):
        norm.append((np.mean(attribute, axis=0), np.std(attribute, axis=0, ddof=1)))
    return norm

def calc_row_probability(norm_data, test_row):
    probability = 1
    for i in range(len(norm_data)):
        mean, std = norm_data[i]
        n = test_row[i]
        probability *= my_gaussian_pdf(mean, std, n)
    return probability

def classify_row(spam_norm, not_spam_norm, test_row):
    spam_probability = calc_row_probability(spam_norm, test_row)
    not_spam_probability = calc_row_probability(not_spam_norm, test_row)
    label = 1 if spam_probability > not_spam_probability else 0
    return label
    
def classify(spam_norm, not_spam_norm, testing_data):
    labels = []
    for row in testing_data:
        labels.append(classify_row(spam_norm, not_spam_norm, row))
    return labels

# Calculate classification statistics testing our predicted labels against the original labels of our testing data
def calculate_statistics(predicted_labels, testing_labels):
    r = len(testing_labels)
    true_spam = 0
    true_not = 0
    false_spam = 0
    false_not = 0
    for i in range(r):
        if predicted_labels[i] == 0 and testing_labels[i] == 0:
            true_not += 1
        elif predicted_labels[i] == 0 and testing_labels[i] == 1:
            false_not += 1
        elif predicted_labels[i] == 1 and testing_labels[i] == 1:
            true_spam += 1
        elif predicted_labels[i] ==1 and testing_labels[i] == 0:
            false_spam += 1

    precision = true_spam/(true_spam + false_spam)
    recall = true_spam/(true_spam + false_not)
    f = 2 * ((precision * recall)/(precision + recall))
    accuracy = (true_spam + true_not) / (true_spam + true_not + false_spam +  false_not)
    return [precision, recall, f, accuracy]

def main():
    np.random.seed(0)
    filename = './spambase.data' if len(sys.argv) == 1 else sys.argv[1]

    raw_data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            raw_data.append(row)
    
    raw_data = np.array(raw_data).astype(np.float)
    size = np.shape(raw_data)
    np.random.shuffle(raw_data)
    
    # Get cutoff index to split data into training/testing sets
    cutoff = np.ceil((size[0]-1)*2/3).astype(np.int)
    
    # Split into training and testing data then get spam/not_spam of training data
    training_data = raw_data[:cutoff, :]
    testing_data = raw_data[cutoff:, :]
    spam, not_spam = split_data(training_data)

    # Get normal models of spam and not_spam sets
    spam_norm = get_norm(spam)
    not_spam_norm = get_norm(not_spam)

    # Use classify function to get our predicted class labels.
    predicted_labels = classify(spam_norm, not_spam_norm, testing_data)

    # Get classification stats from calculate_statistics
    precision, recall, f, accuracy = calculate_statistics(predicted_labels, testing_data[:, -1])
    
    print(
        f'Precision: {round(precision*100, 2)}%\n'
        f'Recall: {round(recall*100, 2)}%\n'
        f'F-measure: {round(f*100, 2)}%\n'
        f'Accuracy: {round(accuracy*100, 2)}%\n'
        )

if __name__== "__main__":
    main()