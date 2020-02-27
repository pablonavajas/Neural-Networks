import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import readData
import math
from sklearn.model_selection import GridSearchCV

def linear_block(in_n, out_n):
    """
    Used to construct the hidden layers in the architecture of the Neural Network 
    """
    return nn.Sequential(
            nn.Linear(in_n, out_n),
            nn.ReLU()
            )

class ClaimClassifier(nn.Module):

#    def __init__(self):
#        
#        #Feel free to alter this as you wish, adding instance variables as
#        #necessary. 
#       
#        super(ClaimClassifier, self).__init__()
#        # Attributes
#        self.batch_size = 100
#        self.num_epochs = 20
#        self.learning_rate = 0.001
#
#        # Model set-up
#        self.layer1 = nn.Linear(9, 4)
#        # self.ReLU = nn.ReLU()
#        self.dropout = nn.Dropout()
#        self.layer2 = nn.Linear(4, 1)
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        out = self.layer1(x)
#        # out = self.ReLU(out)
#        out = self.dropout(out)
#        out = self.layer2(out)
#        out = self.sigmoid(out)
#        return out

    def __init__(self, hidden_layers, batch_size, num_epochs, learning_rate):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 

        hidden_layers is a list of the number of neurons per layer
        """
        super(ClaimClassifier, self).__init__()

        # Attributes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Model set-up
        #1) Passing hidden_layers as a list
        self.layer_neurons = [9] + hidden_layers
        linear_layers = [linear_block(in_f, out_f) 
                            for in_f, out_f in zip(self.layer_neurons, self.layer_neurons[1:])]
        self.encoder = nn.Sequential(*linear_layers)

        #2) Output part
        self.decoder = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.layer_neurons[-1], 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        """
        Override forward() method of nn.Module class to pass input through the neural network.
        """
        out = self.encoder(x)
        out = self.decoder(out)
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
        # Binary Cross Entropy
        criterion = nn.BCELoss()

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(X_raw.shape[0])
            X_shuffled = X_clean[indices].astype(np.float32)
            y_shuffled = y_raw[indices].astype(np.float32)

            X_batches = np.array_split(X_shuffled, nr_batches)
            y_batches = np.array_split(y_shuffled, nr_batches)

            for i, (X, y) in enumerate(zip(X_batches, y_batches)):
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)

                #resize from 100 to (100,1)
                y = y.view(-1,1)
                
                # run forwards
                outputs = self.forward(X)
                loss = criterion(outputs, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # track the accuracy
                total = y.size(0)
                #use first parameter from max function to get tensors of shape [100]
                #for both the outputs and labels (converting from a 
                #tensor of 1x100, each element being a list containing
                #1 element)
                output_values, predicted = torch.max(outputs.data, 1)
                y_values, predicted_y = torch.max(y, 1)

                #round the output probabilities in order to calculate accuracy
                rounded_output_values = torch.round(output_values)
                
                correct = (rounded_output_values == y_values).sum().item()

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

        # create a (rx1 numpy array; here r = 20000)
        arr = outputs.data.numpy()
        arr = [x[0] for x in arr]
        arr = np.array(arr)

        return arr

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
    # -------- possible hyperparameters ------------
    # batch size, num_epochs
    # a 1-D array of nr_neurons in each layer
    # a 1-D array of activation functions for each layer
    # a 1-D array of 1s or 0s indicating presence of dropout function
    # optimization function
    # dropout
    # learning rate and momentum

    data = readData.Dataset("part2_training_data.csv")

    # different options for hyperparameters
    batch_size_arr = [50, 100, 200, 300, 400, 500]
    num_epochs_arr = [10, 20, 30, 50]
    optimizer_arr = ['SGD', 'Adam', 'RMSprop']

    classifier = ClaimClassifier()

    accuracies = []
    params = []

    for batch_size in batch_size_arr:
        for num_epochs in num_epochs_arr:
            for optimizer in optimizer_arr:
                classifier = ClaimClassifier(batch_size, num_epochs, optimizer)

                #evaluate architecture should store self.accuracy,
                # self.precision, self.f1score etc.
                classifier.evaluate_architecture();
                accuracies.append(classifier.accuracy)
                params.append([batch_size, num_epochs, optimizer])

    #need to find index of max accuracies
    max_accuracy = max(accuracies)

    #get the respective params of the max accuracy index
    #params = #to be completed (Iurie)

    return params


def main():
    #Read in the data
    data = readData.Dataset("part2_training_data.csv")
   
    #Splitting off a training set and a test set from the data after randomisation
    #90% training set, 10% test set
    indices = np.random.permutation(data.attributes.shape[0])
    split_point = (data.attributes.shape[0] * 9)//10
    training_idx, test_idx = indices[:split_point], indices[split_point:]
    training_attributes, test_attributes = data.attributes[training_idx], data.attributes[test_idx]
    training_labels, test_labels = data.labels[training_idx], data.labels[test_idx]

    #Create an instance of a classifier
    #Pass in the hidden layers as a list
    hidden_layers = [4, 7, 3]
    classifier = ClaimClassifier(hidden_layers = hidden_layers, batch_size = 100, num_epochs = 20, learning_rate = 0.001)

    #Debugging: print the architecture of the NN.
    print(classifier)

    #Fit the model to the training data
    classifier.fit(training_attributes, training_labels)

    #Test the model on the test data
    arr = classifier.predict(test_attributes)

    print(arr)

if __name__ == "__main__":
    main()
