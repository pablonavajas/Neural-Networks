import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import readData
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import eval_plots
from plots import print_data_split


def linear_block(in_n, out_n):
    """
    Used to construct the hidden layers in the architecture of the Neural
    Network
    """
    return nn.Sequential(
        nn.Linear(in_n, out_n),
        #nn.ReLU()
        nn.Tanh()
    )


class ClaimClassifier(nn.Module):
    def __init__(self, hidden_layers, batch_size, num_epochs,
                 learning_rate):
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
        self.losses = np.zeros(self.num_epochs, dtype = float)

        # Model set-up
        # 1) Passing hidden_layers as a list
        self.layer_neurons = [9] + hidden_layers
        linear_layers = [linear_block(in_f, out_f)
                         for in_f, out_f in
                         zip(self.layer_neurons, self.layer_neurons[1:])]
        self.encoder = nn.Sequential(*linear_layers)

        # 2) Output part
        self.decoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.layer_neurons[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Override forward() method of nn.Module class to pass input through
        the neural network.
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

        # Preprocess the data
        X_clean = self._preprocessor(X_raw)

        nr_batches = math.ceil(X_raw.shape[0] / self.batch_size)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Binary Cross Entropy
        criterion = nn.BCELoss()

        # Array of losses to be used for plotting
        losses = np.zeros(self.num_epochs, dtype=float)

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(X_raw.shape[0])
            X_shuffled = X_clean[indices].astype(np.float32)
            y_shuffled = y_raw[indices].astype(np.float32)

            X_batches = np.array_split(X_shuffled, nr_batches)
            y_batches = np.array_split(y_shuffled, nr_batches)

            for i, (X, y) in enumerate(zip(X_batches, y_batches)):
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)

                # resize from 100 to (100,1)
                y = y.view(-1, 1)

                # run forwards
                outputs = self.forward(X)
                loss = criterion(outputs, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # track the accuracy
                total = y.size(0)
                # use first parameter from max function to get tensors of
                # shape [100]
                # for both the outputs and labels (converting from a
                # tensor of 1x100, each element being a list containing
                # 1 element)
                output_values, predicted = torch.max(outputs.data, 1)
                y_values, predicted_y = torch.max(y, 1)

                # round the output probabilities in order to calculate accuracy
                rounded_output_values = torch.round(output_values)

                correct = (rounded_output_values == y_values).sum().item()

            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, self.num_epochs, loss.item(),
                          (correct / total) * 100))

            self.losses[epoch] = loss.item()

        # Debug
        #print(losses)

        # Plot the epochs-loss curve
        # self.plot_epochs_loss(self.num_epochs, losses)

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

        # create a (rx1 numpy array; here r = 2000)
        arr = outputs.data.numpy()
        arr = [x[0] for x in arr]
        arr = np.array(arr)

        return arr

    # def evaluate_architecture(self):
    def evaluate_architecture(self, y_predict, y_labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # Calculate AUC-ROC graph and AUC metric
        fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_predict)
        # print(fpr)
        # print(tpr)
        # print(thresholds)
        auc = metrics.auc(fpr, tpr)

        # Calculate Precision, Recall and F1_Score
        
        y_rounded = np.where(y_predict < 0.5, 0, 1)
        print(y_rounded)
        print(y_rounded.sum().item())
        print(metrics.classification_report(y_labels, y_rounded, target_names 
        = ['Class 0', 'Class 1']))
        

        return auc, [fpr, tpr]

    def plot_ROC_AUC(self, auc, fpr, tpr):
        """
        Plots the ROC_AUC curve.
        """

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_epochs_loss(self, num_epochs, losses):
        """
        Plots the epochs on the x-axis, the value of the loss function on the
        y-axis.
        """

        x = np.arange(1,num_epochs+1)
        y = losses[x-1]
        plt.title("Loss vs Number of Epochs")
        plt.xlabel("Number of Epochs")
        plt.xticks(x)
        plt.ylabel("Loss")
        plt.plot(x, y)
        plt.show()

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


############################################################
#           HELPERS FOR HYPERPARAM TUNING                  #
############################################################


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

    # different options for hyperparameters
    num_epochs_arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    eval_plots.plot_AUC_num_epochs_chosen(num_epochs_arr, [9, 8, 6, 3])

    num_batches_arr = [50, 100, 150, 200, 250, 300, 350, 400]
    eval_plots.plot_AUC_batch_size(num_batches_arr, [9, 8, 6, 3])



def main():
    # import of data (tested and working correctly)
    dataset = readData.balance_and_split_into_train_valid_test("part2_training_data.csv")
    train_att, train_lab, valid_att, valid_lab, test_att, test_lab = dataset

    print(train_att.shape)
    print(train_lab.shape)
    print(valid_att.shape)
    print(valid_lab.shape)
    print(test_att.shape)
    print(test_lab.shape)


    hidden_layers = [9, 6, 4, 3]
    #hidden_layers = [5]
    classifier = ClaimClassifier(hidden_layers=hidden_layers, batch_size=100,
                                 num_epochs=100, learning_rate=0.001)

    classifier.fit(train_att, train_lab)

    test_predicted_labels = classifier.predict(test_att)

    print(test_predicted_labels.shape)
    print(test_lab)

    auc, [fpr, tpr] = classifier.evaluate_architecture(test_predicted_labels,
                                                       test_lab)

    #print(tpr)
    #print(fpr)
    print("AUC value is: ", auc)

    # Plot the ROC_AUC curve
    classifier.plot_ROC_AUC(auc, fpr, tpr)
    # ClaimClassifierHyperParameterSearch()

    # Plot the epochs-loss curve
    classifier.plot_epochs_loss(classifier.num_epochs, classifier.losses)


    """
    # Debugging: print the architecture of the NN.
    print(classifier)

    # Fit the model to the training data
    classifier.fit(training_attributes, training_labels)

    # Test the model on the test data
    test_predicted_labels = classifier.predict(test_attributes)

    # Debug
    print(test_predicted_labels)

    # Evaluate the performance of the architecture
    auc, [fpr, tpr] = classifier.evaluate_architecture(test_predicted_labels,
                                                       test_labels)
    print("AUC value is: ", auc)

    # Plot the ROC_AUC curve
    classifier.plot_ROC_AUC(auc, fpr, tpr)
    #ClaimClassifierHyperParameterSearch()


    # Plot the epochs-loss curve
    classifier.plot_epochs_loss(classifier.num_epochs, classifier.losses)
    """
if __name__ == "__main__":
    main()
