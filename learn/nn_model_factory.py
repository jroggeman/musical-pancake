from main import engage
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from preprocess import preprocess_file


def neuralnet(features, sample_size=20, hidden_neurons=3):
    """
    This function takes in a list of features (functions that take in an example and return some value based on it)
    and creates and engages a neural network model, returning a dictionary with results from testing.
    :param features: list of functions to act as features in a NN model
    :param hidden_neurons:
    :return: result dictionary, keys are which model was being tested (0-5),
            values are tuples of form (number correct, number tested, percent correct)
    """

    def call_all_features(features, examples, train):
        final = []
        for feat in features:
            f = feat(examples,train)
            if type(f[0]) == list:
                final += f
            else:
                final.append(f)

        result = zip(*final)
        return zip(list(result), [ex.votes['useful'] > 0 for ex in examples])  # [((f1, f2, ..), actual), ..]

    def initialize_models():
        return [NNModel(features) for _ in range(5)]

    def train_model(model, examples):
        matrix = call_all_features(features, examples, True)
        for vector, result in matrix:
            model.dataset.addSample(tuple(vector), (result,))
        model.network = buildNetwork(len(model.features), hidden_neurons, 1)
        model.trainer = BackpropTrainer(model.network, model.dataset)
        model.trainer.train()
        model.trainer.trainUntilConvergence()
        return model

    def model_test(model, example):
        input = test_all_features(model.features, example)
        return model.network.activate(input)[0] > 0.5

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


def make_matrix(features, n):
    examples = preprocess_file('../data/smaller_reviews.json')[:n]
    return [[feat(ex) for feat in features] for ex in examples]


if __name__ == '__main__':

    def feat1(examples, is_training):
        return map(lambda x: len(x.review), examples)

    def feat2(examples, is_training):
        return map(lambda x: 'good' in x.review, examples)

    features = [feat1, feat2]
    result = neuralnet(features, 10, 3)
    print result
