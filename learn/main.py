from random import shuffle
import preprocess
import sys
import pdb

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

    if stochastic:
        training = model.train_model_stochastic
    else:
        training = model.train_model

    models, testing_examples = train_models(
        example_stream, length_of_examples, model.initialize_models,
        training, stochastic, sample_size)

    # Now run tests for each fold:
    accuracies = []
    for k in range(5):
        number_correct = evaluate(models[k], model.model_test, testing_examples[k])

        print(str(k + 1) + '. Results:')

        print(str(number_correct) +
              '/' +
              str(len(testing_examples[k])) +
              ' = ' +
              str(number_correct /
                  float(len(testing_examples[k]))) +
              '%')

        accuracies.append(number_correct / float(len(testing_examples[k])))

    overall_accuracy = mean(accuracies)

    print('Overall accuracy: ' + str(overall_accuracy))


def train_models(
        example_stream,
        length_of_examples,
        initialize_models,
        train_with_example,
        stochastic,
        sample_size):
    models = initialize_models()
    training_examples = [[], [], [], [], []]
    testing_examples = [[], [], [], [], []]
    training_sets, testing_sets = generate_k_fold_indices(length_of_examples, sample_size)

    for index, example in enumerate(example_stream):
        for k in range(5):
            if index in training_sets[k]:
                if stochastic:
                    models[k] = train_with_example(models[k], example)
                else:
                    training_examples[k].append(example)
            # TODO Handle potential memory issues
            elif index in testing_sets[k]:
                testing_examples[k].append(example)

    # Run our global training function if we aren't doing stochastic learning
    if not stochastic:
        for k in range(5):
            models[k] = train_with_example(models[k], training_examples[k])

    return models, testing_examples


def generate_k_fold_indices(length_of_examples, sample_size):
    training_sets = []
    testing_sets = []

    # By default, choose from a random array of all possible indices
    stream_indices = range(length_of_examples)
    shuffle(stream_indices)

    # If we specify a sample size, limit the indices that we'll look at by removing
    # elements from our training indices until we meet the right size
    if sample_size > 0:
        stream_indices = stream_indices[:sample_size]
        length_of_examples = sample_size

    chunk_size = length_of_examples / 5

    # For every fold:
    for i in range(5):
        # e.g. 1000, 2000 for N = 5000 and k = 1
        left_boundary, right_boundary = get_boundaries_of_fold(chunk_size, i)

        # e.g. training_indices = [0, 1, ..., 999, 2000, 2001, ..., 4999] for N
        # = 5000 and k = 1
        testing_indices, training_indices = calculate_fold_indices(
            length_of_examples, left_boundary, right_boundary)

        # Build a set for the actual indices we'll use for training and testing
        training_set = {stream_indices[x] for x in training_indices}
        testing_set = {stream_indices[x] for x in testing_indices}

        # Throw them into the list of sets.  Now:
        #
        #     index in training_sets[k]
        #
        # Will tell us if the index from the stream is in the training set
        # of fold k
        training_sets.append(training_set)
        testing_sets.append(testing_set)

    return training_sets, testing_sets


def calculate_fold_indices(length_of_examples, left_boundary, right_boundary):
    training_indices = range(left_boundary) + \
        range(right_boundary, length_of_examples)
    testing_indices = range(left_boundary, right_boundary)
    return testing_indices, training_indices


def get_boundaries_of_fold(chunk_size, i):
    left_boundary = i * chunk_size
    right_boundary = left_boundary + chunk_size
    return left_boundary, right_boundary


def evaluate(model, model_test, testing_examples):
    number_correct = 0
    for example in testing_examples:
        result = model_test(model, example)
        if example.is_correct(result):
            number_correct += 1

    return number_correct


def progress_bar(progress):
    sys.stdout.write('\r[{0}] {1}%'.format(
        '#' * progress + ' ' * (100 - progress), progress))
    sys.stdout.flush()


def mean(numbers):
    return sum(numbers) / float(len(numbers))
