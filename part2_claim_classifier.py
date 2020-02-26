import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import readData
import math


class ClaimClassifier(nn.Module):

    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super(ClaimClassifier, self).__init__()
        #Attributes
        self.batch_size = 100
        self.num_epochs = 20
        self.learning_rate = 0.001
    
        #Model set-up
        self.layer1 = nn.Linear(9,4)
        #self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer2 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        #out = self.ReLU(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        max_per_col = X_raw.max(axis=0)
        min_per_col = X_raw.min(axis=0)

        Z = (X_raw - min_per_col) / (
                max_per_col - min_per_col)

        return Z

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        nr_batches = math.ceil(X_raw.shape[0] / self.batch_size)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(X_raw.shape[0])
            X_shuffled = X_clean[indices].astype(np.float32)
            #y_shuffled = y_raw[indices]
            y_shuffled = y_raw[indices].astype(np.float32)

            X_batches = np.array_split(X_shuffled, nr_batches)
            y_batches = np.array_split(y_shuffled, nr_batches)

            for i, (X, y) in enumerate(zip(X_batches, y_batches)):
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)

                #print(X.size())
                #print(y.size())

                # run forwards
                outputs = self.forward(X)
                #print(outputs.size())
                loss = criterion(outputs, y)
                #print(loss)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # track the accuracy
                total = y.size(0)
                #print(outputs.data)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y).sum().item()
                #print(total)
                #print(correct)

            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, self.num_epochs, loss.item(),
                          (correct / total) * 100))

        return self

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        X_clean = X_clean.astype(np.float32)
        X = torch.from_numpy(X_clean)
        outputs = self.forward(X)

        #print(outputs)
        print(outputs.data)

        _, predicted = torch.max(outputs.data, 1)

        print(predicted)
        
        """
        total = (predicted != 0).sum().item()
        print(total)
        """
        #print(predicted)
        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your
        # load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the
    # save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters


def main():
    data = readData.Dataset("part2_training_data.csv")

    classifier = ClaimClassifier()

    classifier.fit(data.attributes, data.labels)
    classifier.predict(data.attributes)


if __name__ == "__main__":
    main()
