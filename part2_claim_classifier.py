import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import readData
import math
from sklearn import metrics
import matplotlib.pyplot as plt


def linear_block(in_n, out_n):
    """
    Used to construct the hidden layers in the architecture of the Neural
    Network
    """
    return nn.Sequential(
        nn.Linear(in_n, out_n),
        nn.ReLU()
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
        self.losses = np.zeros(self.num_epochs, dtype=float)
        self.valid_losses = np.zeros(self.num_epochs, dtype=float)

        # Model set-up
        # 1) Passing hidden_layers as a list
        self.layer_neurons = [9] + hidden_layers
        linear_layers = [linear_block(in_f, out_f)
                         for in_f, out_f in
                         zip(self.layer_neurons, self.layer_neurons[1:])]
        self.encoder = nn.Sequential(*linear_layers)

        # 2) Output part
        self.decoder = nn.Sequential(
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
        max_per_col = X_raw.max(axis=0)
        min_per_col = X_raw.min(axis=0)

        Z = (X_raw - min_per_col) / (
                max_per_col - min_per_col)

        return Z

    def fit(self, X_raw, y_raw, X_valid, y_valid ):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable
        X_valid: validation feature set
        y_valid: validation label set

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        ################
        # training
        ################

        # Preprocess the data
        x_clean = self._preprocessor(X_raw)
        X_valid_clean = self._preprocessor(X_valid)

        nr_batches = math.ceil(X_raw.shape[0] / self.batch_size)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Binary Cross Entropy
        criterion = nn.BCELoss()

        # Array of losses to be used for plotting
        losses = np.zeros(self.num_epochs, dtype=float)

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(X_raw.shape[0])
            X_shuffled = x_clean[indices].astype(np.float32)
            y_shuffled = y_raw[indices].astype(np.float32)

            X_batches = np.array_split(X_shuffled, nr_batches)
            y_batches = np.array_split(y_shuffled, nr_batches)

            losses_arr = []
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
                losses_arr.append(loss.item())

            avg_loss = sum(losses_arr)/len(losses_arr)

            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, self.num_epochs, avg_loss,
                          (correct / total) * 100))

            self.losses[epoch] = avg_loss

            ######################
            # validate the model #
            ######################
            self.eval()  # prep model for evaluation
            X_valid_clean = X_valid_clean.astype(np.float32)
            y_valid = y_valid.astype(np.float32)
            X_valid_clean_t = torch.from_numpy(X_valid_clean)
            y_valid_clean_t = torch.from_numpy(y_valid)
            y_valid_clean_t = y_valid_clean_t.view(-1, 1)
            outputs = self.forward(X_valid_clean_t)
            loss = criterion(outputs, y_valid_clean_t)

            self.valid_losses[epoch] = loss.item()

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

        x_clean = self._preprocessor(X_raw)
        x_clean = x_clean.astype(np.float32)
        x = torch.from_numpy(x_clean)
        outputs = self.forward(x)

        arr = outputs.data.numpy()
        arr = [z[0] for z in arr]
        arr = np.array(arr)

        return arr

    def evaluate_architecture(self, y_predict, y_labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # Calculate AUC-ROC graph and AUC metric
        fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_predict)
        auc = metrics.auc(fpr, tpr)

        # Calculate Precision, Recall and F1_Score
        y_rounded = np.where(y_predict < 0.5, 0, 1)
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
        plt.savefig('./images/auc_plot.png', bbox_inches='tight')
        plt.show()

    def plot_epochs_loss(self, num_epochs, losses, valid_losses):
        """
        Plots the epochs on the x-axis, the value of the loss function on the
        y-axis.
        """
        # visualize the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, num_epochs+1), losses,
                 label='Training Loss')
        plt.plot(range(1, num_epochs+1), valid_losses,
                 label='Validation Loss')

        # find position of lowest validation loss
        idx = np.argwhere(np.diff(np.sign(valid_losses-losses))).flatten ()

        intersections_points = np.arange(1, num_epochs+1)[idx]
        if intersections_points.size != 0:
           intersection_point = intersections_points[0]
           plt.axvline(intersection_point, linestyle='--', color='r',
                      label='Intersection of Validation and Training')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 1)  # consistent scale
        plt.xlim(0, num_epochs + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./images/loss_plot.png', bbox_inches='tight')
        plt.show()

    def print_confusion_matrix(self, labels, predicted):
        predicted = np.where(predicted < 0.5, 0, 1)

        assert len(labels) == len(predicted)
        cm = metrics.confusion_matrix(labels, predicted)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        ax.grid(False)

        # annotate with exact numbers in boxes
        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.annotate(cm[i, j], xy=(j, i),
                             horizontalalignment='center',
                             verticalalignment='center',
                             size=25, color='orange')

        ax.tick_params(axis='both', labelsize=13)
        plt.xlabel('Predicted Label', fontsize=18, labelpad=30)
        plt.ylabel('True Label', fontsize=18)
        plt.rcParams["axes.edgecolor"] = "0.6"
        plt.rcParams["axes.grid"] = False

        plt.title("Confusion Matrix Plot", pad=50, fontdict={'fontsize': 20})
        plt.savefig('./images/matrix_plot.png')
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

def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters. 
    """
    # -------- hyper-parameters to tune ------------
    # learning rate
    # number of epochs
    # batch size
    # activation function
    # layers

    """
    We have implemented a more manual tuning of the hyperparameters
    by updating the main function in the order as above - see the report for details.
    """

def main():
    """
    Used to tune the hyperparameters of the model through manual testing. 
    Current set-up is the best model that we found during testing.
    """

    #Import the dataset
    dataset = readData.balance_and_split_into_train_valid_test("part2_training_data.csv")

    #Split into training, validation and test sets
    train_att, train_lab, valid_att, valid_lab, test_att, test_lab = dataset

    #Set the hidden layer neurons
    hidden_layers = [10,20,30]

    #Initiate a classifier
    classifier = ClaimClassifier(hidden_layers=hidden_layers, batch_size=100,
                                 num_epochs=30, learning_rate=0.0001)

    # Train the NN.
    classifier.fit(train_att, train_lab, valid_att, valid_lab)

    # Save the model
    classifier.save_model()

    # Calculate the predicted labels for the test set
    test_predicted_labels = classifier.predict(test_att)

    # Plot the Confusion Matrix
    classifier.print_confusion_matrix(test_lab, test_predicted_labels)

    #Evaluate the architecture
    auc, [fpr, tpr] = classifier.evaluate_architecture(test_predicted_labels,
                                                       test_lab)

    #Print the AUC value
    print("AUC value is: ", auc)

    # Plot the ROC_AUC curve
    classifier.plot_ROC_AUC(auc, fpr, tpr)

    # Plot the Loss-Epochs curve
    classifier.plot_epochs_loss(classifier.num_epochs, classifier.losses, classifier.valid_losses)


if __name__ == "__main__":
    main()
