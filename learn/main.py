from random import shuffle
import preprocess
import pdb

def engage(initialize_models, train_model_stochastic, model_test, file = '../data/reviews.json', extract_features = None):

    # TODO Magic number; better way to handle this in future?
    length_of_examples = 15714
    chunk_size = length_of_examples / 5
    accuracies = []

    example_stream = preprocess.stream_examples(file, extract_features)

    # Generate a list of indices for each example and shuffle them
    indices = range(length_of_examples)
    shuffle(indices)

    chunk_size = length_of_examples / 5

    training_sets = []
    testing_sets = []

    # For each of the possible five folds:
    for i in range(5):

        # We iterate over each fifth of the indices to be used as the test set.
        # Build a left and right boundary of this region:
        left_boundary = i * chunk_size
        right_boundary = left_boundary + chunk_size

        # Generate a list of the indices into the indices array
        # e.g. training_indices = [0, 1, 4, 5, 6, 7, 8, 9]
        #      testing_indices  = [2, 3]
        #      indices          = [2, 4 ,3, 7, 6, 5 ,8, 1, 9, 0]
        #      So that indices[2] and indices[3] (namely, 3 and 7) are the indices
        #      into the example_stream that we'll use for testing
        training_indices = range(left_boundary) + range(right_boundary, length_of_examples)
        testing_indices = range(left_boundary, right_boundary)

        # Build a set for the actual indices we'll use for training and testing
        training_set = {indices[x] for x in training_indices}
        testing_set = {indices[x] for x in testing_indices}

        # Throw them into the list of sets.  Now:
        #
        #     index in training_sets[k]
        #
        # Will tell us if the index from the stream is in the training set
        # of fold k
        training_sets.append(training_set)
        testing_sets.append(testing_set)

    testing_examples = [[], [], [], [], []]

    # Should return list of five empty models
    models = initialize_models()

    # For each example in the stream:
    for index, example in enumerate(example_stream):
        # Look at each fold
        for k in range(5):
            # And if the example we're looking at is part of training for that
            # fold, add it stochastically for the model
            if index in training_sets[k]:
                models[k] = train_model_stochastic(models[k], example)

            # or add it to our list of test examples if it's for testing.
            # TODO: This will cause memory issues again, since we'll end
            # up with the entire dataset in memory again.  Will have to modify
            # to re-stream it.
            elif index in testing_sets[k]:
                testing_examples[k].append(example)

    pdb.set_trace()

    # Now run tests for each fold:
    for k in range(5):
        number_correct = evaluate(models[k], model_test, testing_examples[k])

        print(str(k + 1) + '. Results:')

        print(str(number_correct) + '/' + str(len(testing_examples[k])) + ' = ' +
                str(number_correct / float(len(testing_examples[k]))) + '%')

        accuracies.append(number_correct / float(len(testing_examples[k])))

    overall_accuracy = mean(accuracies)

    print('Overall accuracy: ' + str(overall_accuracy))

def evaluate(model, model_test, testing_examples):
    number_correct = 0
    for example in testing_examples:
        result = model_test(model, example)
        if example.is_correct(result):
            number_correct += 1

    return number_correct

def progress_bar(progress):
    sys.stdout.write('\r[{0}] {1}%'.format('#'*progress + ' '*(100-progress), progress))
    sys.stdout.flush()

def mean(numbers):
    return sum(numbers) / float(len(numbers))

