#!/usr/bin/env python3

"""
This network tries to predict if each nutrient is below (0)
or above (1) the mean, given as input the most relevant ingredients.
"""

import time
import pickle
import collections
import click
import json
import os

import numpy as np
import theano
import theano.tensor as T
import lasagne

# noinspection PyUnresolvedReferences
#from db import Database

from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import get_output, get_all_params
from lasagne.layers import get_all_param_values, set_all_param_values
from sklearn.utils import check_random_state

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

# named tuple for the callback during the training phase
Callback = collections.namedtuple('Callback', ['epoch', 'train_loss', 'train_accuracy', 'train_accuracies',
                                               'validation_loss', 'validation_accuracy', 'validation_accuracies'])


class NeuralNetworkClassification(object):
    """
    Implementation of a neural network for multi-value classification.
    """

    ACTIVATION_FUNCTIONS = {
        'relu': lasagne.nonlinearities.rectify,
        'sigmoid': lasagne.nonlinearities.sigmoid,
        'tanh': lasagne.nonlinearities.tanh
    }

    UPDATES = {
        "sgd": lasagne.updates.sgd,
        "momentum": lasagne.updates.momentum,
        "nesterov": lasagne.updates.nesterov_momentum,
        "adagrad": lasagne.updates.adagrad,
        "adadelta": lasagne.updates.adadelta,
        "adam": lasagne.updates.adam,
        "adamax": lasagne.updates.adamax,
        "rmsprop": lasagne.updates.rmsprop
    }

    def __init__(self, num_inputs, num_outputs, hidden_layers, hidden_layer_size, activation, seed):
        """
        Construct the network.
        """

        # validate parameters
        assert hidden_layers >= 0
        assert hidden_layer_size > 0
        assert activation is not None
        assert seed is not None

        # seed
        np.random.seed(seed)
        rng = check_random_state(seed)
        lasagne.random.set_rng(rng)

        # store the dimension for the input
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        # in our case, the input is one-hot-encoded
        self._x = T.lmatrix()

        # the output is binary
        self._y = T.lmatrix()

        # input layer
        layer = InputLayer(shape=(None, num_inputs), input_var=self._x)

        # hidden layers
        nonlinearity = self.ACTIVATION_FUNCTIONS[activation]
        for _ in range(hidden_layers):
            layer = DenseLayer(layer, hidden_layer_size, nonlinearity=nonlinearity, W=lasagne.init.GlorotUniform())

        # output layers (NB: this can not be softmax!!!)
        layer = DenseLayer(layer, num_outputs, nonlinearity=lasagne.nonlinearities.sigmoid,
                           W=lasagne.init.GlorotUniform())

        # store the entire network
        self._model = layer

        # prediction function
        self._predict = theano.function([self._x], get_output(self._model))

    def fit(self, xs_train, ys_train, xs_validation, ys_validation, epochs, update, learning_rate, batch_size,
            callback=None):
        """
        Train the neural network.
        :param xs_train: Train input matrix.
        :param ys_train: Train output matrix.
        :param xs_validation: Validation input matrix.
        :param ys_validation: Validation output matrix.
        :param epochs: Number of epochs of training.
        :param update: Optimization method.
        :param learning_rate: Learning rate for the optimizer.
        :param batch_size: Size of the batches.
        :param callback: Function to call after each epoch of training.
        """

        # check that the provided data are ok
        assert self._num_inputs == xs_train.shape[1]
        assert self._num_outputs == ys_train.shape[1]
        assert self._num_inputs == xs_validation.shape[1]
        assert self._num_outputs == ys_validation.shape[1]

        # the last layer contains the prediction
        prediction = get_output(self._model)

        # get all weights of the network
        params = get_all_params(self._model, trainable=True)

        # use BINARY cross entropy... the outputs are not categories, but binary values
        loss = lasagne.objectives.binary_crossentropy(prediction, self._y).mean()
        accuracy = lasagne.objectives.binary_accuracy(prediction, self._y).mean()

        # measure accuracy on single features
        accuracies = []
        for i in range(self._num_outputs):
            single_accuracy = lasagne.objectives.binary_accuracy(prediction[:, i], self._y[:, i]).mean()
            accuracies.append(single_accuracy)

        # define the updates, based on the loss function and optimization algorithm
        updates = self.UPDATES[update](loss, params, learning_rate=learning_rate)
        train_fn = theano.function([self._x, self._y], loss, updates=updates)
        loss_fn = theano.function([self._x, self._y], loss)
        accuracy_fn = theano.function([self._x, self._y], accuracy)
        accuracies_fn = theano.function([self._x, self._y], accuracies)

        # print the options
        print('Going to train the network with the following parameters:')
        print('  * %s: %s' % ('epochs', epochs))
        print('  * %s: %s' % ('updates_method', update))
        print('  * %s: %s' % ('learning_rate', learning_rate))
        print('  * %s: %s' % ('batch_size', batch_size))
        print('')

        # keep track of the parameters with the best loss
        best_validation_loss, best_weights = np.inf, None

        # start the training loop
        print('Start training...')
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_accs = np.zeros(self._num_outputs)
            train_batches = 0
            for inputs, targets in self.iterate_mini_batches(xs_train, ys_train, batch_size=batch_size):
                err = train_fn(inputs, targets)
                acc = accuracy_fn(inputs, targets)
                accs = accuracies_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_accs += accs
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_accs = np.zeros(self._num_outputs)
            val_batches = 0
            for inputs, targets in self.iterate_mini_batches(xs_validation, ys_validation, batch_size=batch_size):
                err = loss_fn(inputs, targets)
                acc = accuracy_fn(inputs, targets)
                accs = accuracies_fn(inputs, targets)
                acc2 = sum(accs) / len(accs)
                val_err += err
                val_acc += acc
                val_accs += accs
                val_batches += 1

            # compute the total losses
            train_loss = train_err / train_batches
            train_accuracy = train_acc / train_batches
            train_accuracies = train_accs / train_batches
            validation_loss = val_err / val_batches
            validation_accuracy = val_acc / val_batches
            validation_accuracies = val_accs / val_batches

            # callback
            callback(Callback(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                train_accuracies=train_accuracies,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                validation_accuracies=validation_accuracies
            ))

            # measure the time needed for this epoch
            duration = time.time() - start_time

            # print the progress
            print(("{epoch:4d}: train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} "
                   "train_accuracy={train_accuracy:.4f} validation_accuracy={validation_accuracy:.4f} "
                   "time={duration:.2f}s").format(**locals()))

            # track the best value
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_weights = get_all_param_values(self._model)
                print("^^^ best validation loss so far")

        # after the training, keep the best parameters so far
        set_all_param_values(self._model, best_weights)

    def predict(self, xs):
        """
        Predict the values of some input.
        :param xs: Inputs.
        :return: Prediction.
        """
        return self._predict(xs)

    def dump(self, path):
        """
        Saves the current network parameters as a file.
        :param path: Path to the file.
        """
        with open(path, "wb") as fp:
            pickle.dump(get_all_param_values(self._model), fp)

    def load(self, path):
        """
        Load the network parameters from a file.
        :param path: Path to the file.
        """
        with open(path, "rb") as fp:
            set_all_param_values(self._model, pickle.load(fp))

    @staticmethod
    def iterate_mini_batches(inputs, targets, batch_size):
        """
        Utility method to iterate over the training data in batches.
        At each call of the function, data are randomly shuffled and split in batches.
        An iterator over the batches is returned.
        :param inputs: X.
        :param targets: y.
        :param batch_size: Number of training examples for each batch.
        :return: Iterator over the batches, returning a couple (X, y) at each step.
        """
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield inputs[excerpt], targets[excerpt]


def binarize(ys):
    """
    Convert a matrix of continuous valued to a binary one.
    The output is 0 if the feature is below the mean, 1 if it is above.
    :param ys: Matrix, examples as rows, features as columns.
    :return: Converted matrix.
    """
    nutrient_means = ys.mean(0)
    nutrients_diffs = ys - nutrient_means
    nutrients_diffs[nutrients_diffs <= 0] = 0
    nutrients_diffs[nutrients_diffs > 0] = 1
    return nutrients_diffs.astype(int)


def load_dataset(path):
    """
    Load the dataset from a pickle file.
    For each nutrient, compute if it is below or above the mean.
    :param path: Path for the pickle file.
    :return: Dataset + mapping for xs.
    """
    with open(path, "rb") as fp:
        dataset = pickle.load(fp)
    xs, ys_original = dataset["xs"], dataset["ys"]
    assert xs.shape[0] == ys_original.shape[0]
    ys = binarize(ys_original)
    return xs, ys, dataset["nutrients_mapping"]


def split_dataset(xs, ys, faction_valid=0.2, seed=0):
    """
    Split the dataset into train and validation.
    :param xs: Input matrix.
    :param ys: Output matrix.
    :param faction_valid: Fraction of the training example to use for the validation.
    :param seed: Seed used for the random split.
    :return: X_train, y_train, X_validation, y_validation.
    """

    # check the size
    num_examples = xs.shape[0]
    assert num_examples == ys.shape[0]

    # fix the random generator
    rng = check_random_state(seed)

    # random split
    num_valid_examples = int(np.rint(num_examples * faction_valid))
    indexes = rng.permutation(num_examples)

    # train
    train_indices = indexes[num_valid_examples:]
    train_xs, train_ys = xs[train_indices], ys[train_indices]

    # validation
    valid_indices = indexes[:num_valid_examples]
    valid_xs, valid_ys = xs[valid_indices], ys[valid_indices]

    # return the split
    return train_xs, train_ys, valid_xs, valid_ys

'''
def make_plot_callback(directory, n_epochs, db_file, nutrients_mapping):
    """
    Create a callback that can handle the callback during the training phase.
    The callback function will plot loss and accuracy of the network.
    :param directory: Directory where to store the plots.
    :param n_epochs: Total epochs of training.
    :param db_file: File of the database.
    :param nutrients_mapping: List of nutrients (in the same order as the y matrix).
    :return: Callback function.
    """

    # database (to resolve the nutrients names)
    db = Database(db_file)
    db.open()
    nutrients = db.get_all_nutrients()
    db.close()

    # compute the map: nutrient id -> name
    nutrients_names_map = dict(nutrients)

    # store the history
    history = []

    # callback function (called after each training epoch)
    def callback(row):
        history.append(row)

        # optimize resources
        if row.epoch % 20 == 0 or row.epoch == n_epochs:

            # dump the history
            with open(directory + os.sep + 'history.pickle', "wb") as fp:
                pickle.dump(history, fp)

            # plot
            epochs = list(map(lambda item: item.epoch, history))
            train_loss = list(map(lambda item: item.train_loss, history))
            validation_loss = list(map(lambda item: item.validation_loss, history))
            train_accuracy = list(map(lambda item: item.train_accuracy, history))
            validation_accuracy = list(map(lambda item: item.validation_accuracy, history))

            # plot the loss
            figure_losses = plt.figure()
            ax_losses = figure_losses.add_subplot(111)
            ax_losses.plot(epochs, train_loss, label='train')
            ax_losses.plot(epochs, validation_loss, label='validation')
            ax_losses.set_title('Model Loss (categorical cross entropy)')
            ax_losses.set_ylabel('loss')
            ax_losses.set_xlabel('epoch')
            ax_losses.legend()
            figure_losses.savefig(directory + os.sep + 'loss.pdf')
            plt.close(figure_losses)

            # plot the accuracy
            figure_acc = plt.figure()
            ax_acc = figure_acc.add_subplot(111)
            ax_acc.plot(epochs, train_accuracy, label='train')
            ax_acc.plot(epochs, validation_accuracy, label='validation')
            ax_acc.set_title('Model Accuracy (average over all features)')
            ax_acc.set_ylabel('accuracy')
            ax_acc.set_xlabel('epoch')
            ax_acc.legend()
            figure_acc.savefig(directory + os.sep + 'accuracy_average_of_features.pdf')
            plt.close(figure_acc)

            # plot the single accuracies
            for i in range(len(row.validation_accuracies)):
                nutrient_id = nutrients_mapping[i]
                nutrient_name = nutrients_names_map[nutrient_id]
                train_single_accuracy = list(map(lambda item: item.train_accuracies[i], history))
                validation_single_accuracy = list(map(lambda item: item.validation_accuracies[i], history))
                figure_acc = plt.figure()
                ax_acc = figure_acc.add_subplot(111)
                ax_acc.plot(epochs, train_single_accuracy, label='train')
                ax_acc.plot(epochs, validation_single_accuracy, label='validation')
                ax_acc.set_title('Model Accuracy for feature %s -> %s [%s]' % (i, nutrient_name, nutrient_id))
                ax_acc.set_ylabel('accuracy')
                ax_acc.set_xlabel('epoch')
                ax_acc.legend()
                figure_acc.savefig(directory + os.sep + 'accuracy_feature_%s.pdf' % i)
                plt.close(figure_acc)

    # the callback will have access to the history
    return callback
'''

@click.command()
@click.option('--hidden-layers', help='Number of hidden layers for the network.', prompt=True)
@click.option('--hidden-layer-size', help='Number of units for each hidden layer.', prompt=True)
@click.option('--activation',
              help='Activation function to use for the network (except the output layer) [relu, sigmoid, tanh].',
              prompt=True)
@click.option('--seed', help='Random seed to use for the split of the data and network initialization.', default=0)
@click.option('--epochs', help='Number of epochs of training.', default=100)
@click.option('--update',
              help='Optimization method to use [sgd, momentum, nesterov, adagrad, adadelta, adam, adamax, rmsprop].',
              default='momentum')
@click.option('--learning-rate', help='Learning rate for the optimizer.', default=0.1)
@click.option('--batch-size', help='Number of samples to use for each mini-batch.', default=300)
#@click.option('--db-file', help='Database file where to resolve the nutrient names.', default='../data.db')
@click.option('--input-file', help='Pickle file with the data for the training.', default='../data_cleaned.pickle')
@click.option('--output-directory', help='Directory where to store the results of the training.', default='../results')
def main(hidden_layers, hidden_layer_size, activation, seed, epochs, update, learning_rate, batch_size, db_file,
         input_file, output_directory):
    """
    Train a neural network to perform classification: given the ingredients of a product,
    tell whether its nutrients are below or above the mean.
    """

    # dump the configuration
    arguments = locals()
    directory = output_directory + os.sep + str(int(time.time() * 1000))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + os.sep + 'config.json', 'w') as f:
        json.dump(arguments, fp=f, indent=4)

    # load the data
    xs, ys, nutrients_mapping = load_dataset(input_file)
    xs_train, ys_train, xs_validation, ys_validation = split_dataset(xs, ys)

    # recompute the size of the network
    num_inputs = xs.shape[1]
    num_outputs = ys.shape[1]

    # build the network
    nn = NeuralNetworkClassification(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_layers=int(hidden_layers),
        hidden_layer_size=int(hidden_layer_size),
        activation=activation,
        seed=int(seed)
    )

    # callback -> make plots
    #callback = make_plot_callback(directory, n_epochs=int(epochs), db_file=db_file, nutrients_mapping=nutrients_mapping)

    # train the network
    nn.fit(
        xs_train=xs_train,
        ys_train=ys_train,
        xs_validation=xs_validation,
        ys_validation=ys_validation,
        epochs=int(epochs),
        update=update,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        callback=callback
    )

    # dump the final weights
    nn.dump(directory + os.sep + "weights.pickle")


# entry point for the script
if __name__ == '__main__':
    main()
