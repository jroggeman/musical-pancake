from main import engage
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


def neuralnet(features, sample_size=20, hidden_neurons=3):
    """
    This function takes in a list of features (functions that take in an example and return some value based on it)
    and creates and engages a neural network model, returning a dictionary with results from testing.
    :param features: list of functions to act as features in a NN model
    :param sample_size:
    :param hidden_neurons:
    :return: result dictionary, keys are which model was being tested (0-5),
            values are tuples of form (number correct, number tested, percent correct)
    """
    def initialize_models():
        return [NNModel(features) for _ in range(5)]

    def train_model(model, examples):
        print 'adding examples'
        for example in examples:
            model.dataset.addSample(tuple([func(example) for func in model.features]), (example.votes['useful'] > 0,))
        print 'building network'
        model.network = buildNetwork(len(model.features), hidden_neurons, 1)
        model.trainer = BackpropTrainer(model.network, model.dataset)
        print 'training network'
        model.trainer.train()
        model.trainer.trainUntilConvergence()
        print 'done'
        return model

    def model_test(model, example):
        model_result = model.network.activate([func(example) for func in model.features])[0] > 0
        return model_result == (example.votes['useful'] > 0)

    jole = Jole(initialize_models, train_model, model_test)
    return engage(jole, filename='../data/smaller_reviews.json', stochastic=False, sample_size=sample_size)


class NNModel(object):
    def __init__(self, features):
        self.features = features
        self.dataset = SupervisedDataSet(len(self.features), 1)
        self.trainer, self.network = None, None


class Jole(object):
    def __init__(self, initialize_models, train_model, model_test):
        self.initialize_models = initialize_models
        self.train_model = train_model
        self.model_test = model_test


if __name__ == '__main__':
    result = neuralnet([
        lambda x: len(x.review)
    ], 30, 3)
    print result
