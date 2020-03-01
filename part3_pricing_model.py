from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# Import extra libraries for building a classifier
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# External library to process categorical data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

import pandas as pd
import DataRead


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    #def __init__(self, calibrate_probabilities=False):
    def __init__(self, initial_layer, hidden_layers,
                 batch_size, num_epochs, learning_rate, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None  #will be a number (the mean of the claims) set in the fit() function
        self.calibrate = calibrate_probabilities
        #self.base_classifier = None # ADD YOUR BASE CLASSIFIER HERE
        self.base_classifier = BinaryClaimClassifier(initial_layer, hidden_layers, batch_size,
                                    num_epochs, learning_rate)

    
     # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
    
        # Establish a maximum cardinality for categorical variables
        max_card = 50
        
        # Find all categorical columns
        categorical_cols = X_raw.select_dtypes(exclude=np.number)

        # Select columns with too high cardinality
        excs_card_cols = [col for col in categorical_cols if categorical_cols[col].nunique() > max_card]

        print(excs_card_cols)
    
        # drop columns with excessive cardinality
        X_raw = X_raw.drop(columns=excs_card_cols)

        # ... continue with solution
        # listp = [i[p] for i in l1np for p in range(len(l1np[0])) if isinstance(i[p], (str))]

        # Find all columns with categorical data
        categorical_cols = X_raw.select_dtypes(exclude=np.number)
        
        X_cols = X_raw.columns
        X_cat_cols = categorical_cols.columns
        strlist = [i for i in range(len(X_cols)) if X_cols[i] in X_cat_cols]

        X_np = X_raw.to_numpy()
        
        # Replace all missing values
        # (will be 0 for numerical data and "missing_val" for categorical data
        imputer_str = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value='missing_val')
        imputer_num = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value=-1)

        for col in range(len(X_np[0])):
            if col in strlist:
                imputer_str = imputer_str.fit(X_np[:,col:col+1])
                X_np[:,col:col+1] = imputer_str.transform(X_np[:,col:col+1])
            else:
                imputer_num = imputer_num.fit(X_np[:,col:col+1])
                X_np[:,col:col+1] = imputer_num.transform(X_np[:,col:col+1])

        # Convert Categorical data into 1-Hot format
        label_coder = LabelBinarizer()

        for col in range(len(X_np[0])):
            if col in strlist:
                onehot = label_coder.fit_transform(X_np[:,col])
                if col == 0:
                    X = onehot
                else:
                    X = np.concatenate((X, onehot), axis=1)
            else:
                if col == 0:
                    X = X_np[:,col:col+1]
                else:
                    X = np.concatenate((X, X_np[:,col:col+1]), axis=1)
                    
        return X

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        #Used to calculate the severity constant.
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
       
        #Clean the raw data
        X_clean = self._preprocessor(X_raw)

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        """
        We won't be calibrating our probabilities hence the else branch
        will be executed.
        """
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)

        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        #Clean the data, then pass it into the predict method
        #of the binary classifier
        X_clean = self._preprocessor(X_raw)
        return self.base_classifier.predict(X_clean)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor
        

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


##############################################################################
# Modified Binary Classifier from Part 2 used to create the frequency model. #
##############################################################################

def linear_block(in_n, out_n):
    """
    Used to construct the hidden layers in the architecture of the
    Binary Claim Classifier.
    """
    return nn.Sequential(
        nn.Linear(in_n, out_n),
        nn.ReLU()
    )

class BinaryClaimClassifier(nn.Module):
    def __init__(self, initial_layer, hidden_layers, batch_size, num_epochs,
                 learning_rate):
        """
        initial_layer is the number of neurons in the input layer.
        hidden_layers is a list of the number of neurons per layer.
        batch_size is the size per batch used in training.
        num_epochs is the number of epochs used in training.
        learning_rate is the rate applied for gradient descent.
        """
        super(BinaryClaimClassifier, self).__init__()

        # Attributes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.losses = np.zeros(self.num_epochs, dtype=float)
        #self.valid_losses = np.zeros(self.num_epochs, dtype=float)

        # Model set-up
        # 1) Passing hidden_layers as a list
        self.layer_neurons = [initial_layer] + hidden_layers
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

    def fit(self, X_clean, y_raw):
        """ 
        Parameters
        ----------
        X_raw : ndarray
            An array, this is the clean data used after pre-processing
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """
        # Preprocess the data
        #X_clean = self._preprocessor(X_raw)
        #X_valid_clean = self._preprocessor(X_valid)

        nr_batches = math.ceil(X_clean.shape[0] / self.batch_size)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Binary Cross Entropy
        criterion = nn.BCELoss()

        # Array of losses to be used for plotting
        losses = np.zeros(self.num_epochs, dtype=float)

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(X_clean.shape[0])
            X_shuffled = X_clean[indices].astype(np.float32)
            y_shuffled = y_raw[indices].astype(np.float32)

            X_batches = np.array_split(X_shuffled, nr_batches)
            y_batches = np.array_split(y_shuffled, nr_batches)

            losses_arr = []
            for i, (X, y) in enumerate(zip(X_batches, y_batches)):
                X = torch.from_numpy(X)
                y = torch.from_numpy(y)

                # resize from batch_size to (batch_size,1)
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
            #self.eval()  # prep model for evaluation
            #X_valid_clean = X_valid_clean.astype(np.float32)
            #y_valid = y_valid.astype(np.float32)
            #X_valid_clean_t = torch.from_numpy(X_valid_clean)
            #y_valid_clean_t = torch.from_numpy(y_valid)
            #y_valid_clean_t = y_valid_clean_t.view(-1, 1)
            #outputs = self.forward(X_valid_clean_t)
            #loss = criterion(outputs, y_valid_clean_t)

            #self.valid_losses[epoch] = loss.item()

        return self

    def predict(self, X_clean):
        """ 
        Parameters
        ----------
        X_clean : ndarray
            An array, this is the preprocessed data

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        X_clean = X_clean.astype(np.float32)
        X = torch.from_numpy(X_clean)
        outputs = self.forward(X)

        arr = outputs.data.numpy()
        arr = [x[0] for x in arr]
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
        print(intersections_points)
        if intersections_points.size != 0:
           intersection_point = intersections_points[0]
           print(intersection_point)
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
        plt.savefig('./images/matrix_plot_part3.png')
        plt.show()

def main():
    """
    Used to tune the hyperparameters of the model through manual testing. 
    Current set-up is the best model that we found during testing.
    """

    # Import the dataset
    dat = pd.read_csv("part3_training_data.csv")

    #TODO - Split into appropriate sets (maybe training and test, then will 
    #pass training into the fit() function of the classifier and split this into
    #training and validation within the function itself for training
    #Remember to randomise this dataset before doing splits.
    #dataset = DataRead.balance_and_split_into_train_valid_test(X_raw, y_raw)
    #train_att, train_lab, valid_att, valid_lab, test_att, test_lab = dataset

    msk = np.random.rand(len(dat)) < 0.8

    train = dat[msk]
    test = dat[~msk]
    
       
    #TODO - Need to split off a claims_raw np.array
    #Calling it claims_raw in code below
    X_raw = train.drop(columns=["claim_amount", "made_claim"])

    y_raw = train["made_claim"]
    y_raw = y_raw.to_numpy()
    
    claims_raw = train["claim_amount"]
    claims_raw = claims_raw.to_numpy()

    test_X_raw = test.drop(columns=["claim_amount", "made_claim"])

    test_y_raw = test["made_claim"]
    test_y_raw = test_y_raw.to_numpy()
    
    test_claims_raw = test["claim_amount"]
    test_claims_raw = test_claims_raw.to_numpy()

    #TODO - Set the input layer
    input_layer = 43

    #TODO - PICK YOUR OWN EXAMPLE
    #Set the hidden layer neurons
    hidden_layers = [10,20,30]

    #Initiate a Pricing Model
    pricingmodel = PricingModel(input_layer, hidden_layers,batch_size = 100, num_epochs = 30,
                                learning_rate = 0.001, calibrate_probabilities=False)

    # Train the NN.
    #TODO - CHANGE THE Parameter names as appropriate
    pricingmodel.fit(X_raw, y_raw, claims_raw)

    # TODO = Save the best model - commented out for now
    #pricingmodel.save_model()

    #Calculate the predicted_probabilities
    predicted_prob = pricingmodel.predict_claim_probability(test_X_raw)

    #Calculate the premiums
    #TODO - Change parameter name as appropriate
    pricingmodel.predict_premium(test_X_raw)

    # Plot the Confusion Matrix
    #TODO - not sure if you need this
    #Change the params as needed
    pricingmodel.base_classifier.print_confusion_matrix(test_y_raw, predicted_prob)

    #Evaluate the architecture
    #TODO - change y_test for variable name you need
    auc, [fpr, tpr] = pricingmodel.base_classifier.evaluate_architecture(predicted_prob,
                                                       test_y_raw)

    #Print the AUC value
    #TODO - need this above 60%
    print("AUC value is: ", auc)

    # Plot the ROC_AUC curve
    #TODO - May not need this
    pricingmodel.base_classifier.plot_ROC_AUC(auc, fpr, tpr)

    # Plot the Loss-Epochs curve
    #TODO - May not need this
    #TODO - Will need me to amend the fit() function of the binary classifier if using this
    # need to declare valid_losses in base_classifier
    #pricingmodel.base_classifier.plot_epochs_loss(pricingmodel.base_classifier.num_epochs,
    #        pricingmodel.base_classifier.losses, pricingmodel.base_classifier.valid_losses)


if __name__ == "__main__":
    main()
