import numpy as np
import readData
import matplotlib.pyplot as plt
from part2_claim_classifier import ClaimClassifier
from sklearn import metrics


def plot_AUC_num_epochs_chosen(num_epochs_arr, hidden_layers=[4, 5, 3]):
    # Read in the data
    data = readData.Dataset("part2_training_data.csv")

    # store batch_size and num_epochs
    num_epochs_choices = []
    auc_result = []

    # params passed into the constructor of ClaimClassifier()
    # hidden_layers, batch_size, num_epochs, learning_rate

    for num_epochs in num_epochs_arr:
        # Splitting off a training set and a test set from the data after
        # randomisation
        # 90% training set, 10% test set
        indices = np.random.permutation(data.attributes.shape[0])
        split_point = (data.attributes.shape[0] * 9) // 10
        training_idx, test_idx = indices[:split_point], indices[
                                                        split_point:]
        training_attributes, test_attributes = data.attributes[
                                                   training_idx], \
                                               data.attributes[test_idx]
        training_labels, test_labels = data.labels[training_idx], \
                                       data.labels[
                                           test_idx]

        # Create an instance of a classifier
        # Pass in the hidden layers as a list
        classifier = ClaimClassifier(hidden_layers=hidden_layers,
                                     batch_size=100,
                                     num_epochs=num_epochs, learning_rate=0.001)

        # Fit the model to the training data
        classifier.fit(training_attributes, training_labels)

        # Test the model on the test data
        test_predicted_labels = classifier.predict(test_attributes)

        # Debug
        print(test_predicted_labels)

        # Evaluate the performance of the architecture
        auc, [fpr, tpr] = classifier.evaluate_architecture(
            test_predicted_labels,
            test_labels)
        num_epochs_choices.append(num_epochs)
        auc_result.append(auc)

    print(auc_result)
    print(num_epochs_choices)
    # plot graph of batches vs auc
    plt.title("AUC vs nr of epochs chosen")
    plt.xlabel("epochs nr")
    plt.ylabel("AUC")
    plt.plot(num_epochs_choices, auc_result)
    plt.savefig('./images/auc_vs_nr_of_epochs.png')
    plt.show()


def plot_AUC_batch_size(batch_size_arr, hidden_layers=[4, 5, 3]):
    # Read in the data
    data = readData.Dataset("part2_training_data.csv")

    # store batch_size and num_epochs
    batch_size_choices = []
    auc_result = []

    # params passed into the constructor of ClaimClassifier()
    # hidden_layers, batch_size, num_epochs, learning_rate

    for batch_size in batch_size_arr:
        # Splitting off a training set and a test set from the data after
        # randomisation
        # 90% training set, 10% test set
        indices = np.random.permutation(data.attributes.shape[0])
        split_point = (data.attributes.shape[0] * 9) // 10
        training_idx, test_idx = indices[:split_point], indices[
                                                        split_point:]
        training_attributes, test_attributes = data.attributes[
                                                   training_idx], \
                                               data.attributes[test_idx]
        training_labels, test_labels = data.labels[training_idx], \
                                       data.labels[
                                           test_idx]

        # Create an instance of a classifier
        # Pass in the hidden layers as a list
        classifier = ClaimClassifier(hidden_layers=hidden_layers,
                                     batch_size=batch_size,
                                     num_epochs=50, learning_rate=0.001)

        # Fit the model to the training data
        classifier.fit(training_attributes, training_labels)

        # Test the model on the test data
        test_predicted_labels = classifier.predict(test_attributes)

        # Debug
        print(test_predicted_labels)

        # Evaluate the performance of the architecture
        auc, [fpr, tpr] = classifier.evaluate_architecture(
            test_predicted_labels,
            test_labels)
        batch_size_choices.append(batch_size)
        auc_result.append(auc)

    print(auc_result)
    print(batch_size_choices)
    # plot graph of batches vs auc
    plt.title("AUC vs nr of batches")
    plt.xlabel("batches nr")
    plt.ylabel("AUC")
    plt.plot(batch_size_choices, auc_result)
    plt.savefig('./images/AUC_vs_nr_of_batches.png')
    plt.show()

print_confusion_matrix([88,13],[100,50])
