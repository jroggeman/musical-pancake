import preprocess
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle


def shelley(features):
    """
    This function takes in a list of features (functions that take in an example and return some value based on it)
    and creates and engages a neural network model, returning a dictionary with results from testing.
    :param features: list of functions to act as features in a NN model
    :return: result dictionary, keys are which model was being tested (0-5),
            values are tuples of form (number correct, number tested, percent correct)
    """
    def initialize_models():
        return NNModel(features)

    def train_model(model, examples):
        for example in examples:

            model.dataset.addSample(tuple([func(example) for func in model.features]), (example.votes['useful'] > 0,))

        model.network = buildNetwork(len(model.features), 3, 1)  # 3 hidden layers for now. can change at any time
        model.trainer = BackpropTrainer(model.network, model.dataset)
        model.trainer.train()
        model.trainer.trainUntilConvergence()
        return model

    def model_test(model, example):
        model_result = model.network.activate([func(example) for func in model.features])[0] > 0
        return model_result == (example.votes['useful'] > 0)

    jole = Jole(initialize_models, train_model, model_test)
    return engage(jole, filename='../data/smaller_reviews.json', stochastic=False, sample_size=100)


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

def engage(
        model,
        filename='../data/reviews.json',
        extract_features=None,
        stochastic=True,
        sample_size=15714):

    # TODO Magic number; better way to handle this in future? Don't want to
    # read whole file
    length_of_examples = 15714

    example_stream = preprocess.stream_examples(filename, extract_features)

    training = model.train_model

    models, testing_examples = train_models(
    example_stream,  model.initialize_models,
    training, stochastic, sample_size)

    number_correct = evaluate(models, model.model_test, testing_examples)
    print( '. Results:')
    print(str(number_correct) +
          '/' +
          str(len(testing_examples)) +
          ' = ' +
          str(number_correct /
              float(len(testing_examples)) * 100) +
          '%')
    overall_accuracy=(number_correct / float(len(testing_examples)))
    result = (number_correct, len(testing_examples), number_correct / float(len(testing_examples)))

    print('Overall accuracy: ' + str(overall_accuracy))
    return result


def train_models(
        example_stream,
        initialize_models,
        train_with_example,
        stochastic,
        sample_size):

    models = initialize_models()
    training_examples = []
    testing_examples = []
    count = 0

    for ex in example_stream:
        if count >= sample_size:
            break
        if count%4 == 0:
            testing_examples.append(ex)
        else:
            training_examples.append(ex)
        count += 1


    # Run our global training function if we aren't doing stochastic learning
    models = train_with_example(models, training_examples)

    return models, testing_examples



def evaluate(model, model_test, testing_examples):
    number_correct = 0
    for example in testing_examples:
        result = model_test(model, example)
        print result
        if example.is_correct(result):
            number_correct += 1

    return number_correct




if __name__ == '__main__':
    result = joal([
        lambda x: 'good' in x.review,
    ])
    print result
