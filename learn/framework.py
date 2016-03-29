from __future__ import division
from preprocess import stream_examples
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


class Model(object):

    def __init__(self, features=None, n=50, hidden_layers=3, split=.5, label=None):
        """ Creates a runnable/evaluate-able model object. Pardon my spelling.
        :param features: iterable of functions that take in one parameter, a training example,
            and return some values based on that specific example
        :param n: number of Examples to read in from stream
        :param hidden_layers: number of hidden layers in neural net
        :param split: float in (0, 1) to represent proportion of train data to the whole dataset
        :param label: function that takes in a labeled example and returns the specific lable for
            this model (i.e usefulness)
        """
        self.features = features
        if not self.features:
            print 'Warning: You should probably give your model some features.'
        if not label:
            raise ValueError('Label function is needed to classify Examples.')
        self.n = n
        self.hidden_layers = hidden_layers
        self.split = split
        self.label = label
        self.DataSet = SupervisedDataSet(len(self.features), 1)
        self.trained = False

    def train(self, training):
        """ Train the model
        :param training: An iterable of Examples for training purposes
        """
        for example in training:
            self.DataSet.addSample(tuple([func(example) for func in self.features]), (self.label(example),))
        self.network = buildNetwork(len(self.features), self.hidden_layers, 1)
        self.trainer = BackpropTrainer(self.network, self.DataSet)
        self.trainer.train()
        self.trainer.trainUntilConvergence()
        self.trained = True

    def test(self, test):
        """
        :param test: An iterable of Examples for testing purposes
        :return: The list of results based on the model's decisions
        """
        if self.trained:
            return [self.network.activate([func(ex) for func in self.features])[0] > 0 for ex in test]
        else:
            raise ReferenceError('Cannot test an untrained model. Use Framework.run() instead')


class Framework(object):

    def __init__(self, model=None, additional_preprocessing=None):
        """ A small framework for training, and evaluating models
        :param model: An untrained Model object to train, test, and evaluate
        :param additional_preprocessing: A function that takes in an Example object and returns
            a preprocessed Example object
        """
        self.model = model
        if not self.model:
            raise ValueError('Need a model to engage')
        self.stream = stream_examples('../data/reviews.json', additional_preprocessing)
        self.result = None
        self.actual = None

    def run(self):
        """ Runs the framework, calculates true positive, true negative, false positive,
            and false negative values from testing the trained model
        """
        data = [next(self.stream) for _ in range(50)]
        idx = int(self.model.n * self.model.split)
        # TODO k-fold validation, use mean values for evaluation metrics
        self.model.train(data[:idx])
        self.result = self.model.test(data[idx:])
        self.actual = map(self.model.label, data[idx:])
        pairs = zip(self.result, self.actual)
        self.tp = [a and b for a, b in pairs]
        self.tn = [not a and not b for a, b in pairs]
        self.fp = [a and not b for a, b in pairs]
        self.fn = [not a and b for a, b in pairs]

    @property
    def accuracy(self):
        correct = [a == b for a, b in zip(self.result, self.actual)]
        return sum(correct) / len(correct)

    @property
    def precision(self):
        return sum(self.tp) / sum(self.tp + self.fp)

    @property
    def recall(self):
        return sum(self.tp) / sum(self.tp + self.fn)

    @property
    def f1_score(self):
        return 2 * ((self.precision * self.recall) / (self.precision + self.recall))
