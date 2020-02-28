###################################################################
# Dataset class to read in the attributes and labels from a file. #
###################################################################

import numpy as np


# function to import data from excel
class Dataset:

    def __init__(self, filename):
        self.attributes = self.import_data(filename)[0]
        self.labels = self.import_data(filename)[1]

    def import_data(self, filename):
        data = open(filename, 'r').read()

        L = data.split('\n')[:-1]
        L = L[1:]

        J = [int(row[-1]) for row in L]

        L = [row[:-2] for row in L]
        L = [i.split(",") for i in L]
        L = [row[:-1] for row in L]
        L = [[float(element) for element in row] for row in L]

        L = np.array(L)
        J = np.array(J)

        return L, J


# function to split the data into train, validation and test sets
def balance_and_split_into_train_valid_test(filename):
    # Read in the data
    data = Dataset(filename)
    original_n = data.attributes.shape[0]

    # shuffle data
    indices = np.random.permutation(original_n)
    data.attributes = data.attributes[indices]
    data.labels = data.labels[indices]

    unique_labels, count = np.unique(data.labels, return_counts=True)

    zeros = np.where(data.labels==0)
    ones = np.where(data.labels==1)

    zero_att = data.attributes[zeros]
    one_att = data.attributes[ones]

    zero_lab = data.labels[zeros]
    one_lab = data.labels[ones]

    # print(zero_att.shape)
    # print(zero_lab.shape)
    # print(one_att.shape)
    # print(one_lab.shape)

    zero_att = zero_att[:one_att.shape[0]]
    zero_lab = zero_lab[:one_att.shape[0]]

    data.attributes = np.vstack((zero_att, one_att))
    data.labels = np.append(zero_lab, one_lab)

    indices = np.random.permutation(data.attributes.shape[0])
    split_point1 = (data.attributes.shape[0] * 8) // 10
    split_point2 = (data.attributes.shape[0] * 9) // 10

    train_att = data.attributes[:split_point1]
    train_lab = data.labels[:split_point1]

    valid_att = data.attributes[split_point1:split_point2]
    valid_lab = data.labels[split_point1:split_point2]

    test_att = data.attributes[split_point2:]
    test_lab = data.labels[split_point2:]

    return train_att, train_lab, valid_att, valid_lab, test_att, test_lab


def main():
    dataset = balance_and_split_into_train_valid_test("part2_training_data.csv")
    train_att, train_lab, valid_att, valid_lab, test_att, test_lab = dataset
    print(train_att.shape)
    print(train_lab.shape)
    print(valid_att.shape)
    print(valid_lab.shape)
    print(test_att.shape)
    print(test_lab.shape)

if __name__ == "__main__":
    main()
