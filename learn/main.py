from random import shuffle
import preprocess

def engage(train_model, model_test, file = '../data/reviews.json', extract_features = None):

    # TODO Magic number; better way to handle this in future?
    length_of_examples = 15714
    chunk_size = length_of_examples / 5
    accuracies = []
    for i in range(5):
        # TODO Waste lots of work at the moment
        example_stream = preprocess.stream_examples(file, extract_features)

        print('Running fold ' + str(i))
        # Calculate indices for boundaries of test fold
        left_boundary = i * chunk_size
        right_boundary = left_boundary + chunk_size

        # Calculate the indices of the examples to use
        training_indices = range(left_boundary) + range(right_boundary, length_of_examples)
        testing_indices = range(left_boundary, right_boundary)

        model, testing_examples = train_model(example_stream, training_indices, testing_indices)

        number_correct = evaluate(model, model_test, testing_examples)

        print(str(i) + '. Results:')

        print(str(number_correct) + '/' + str(len(testing_examples)) + ' = ' +
                str(number_correct / float(len(testing_examples))) + '%')

        accuracies.append(number_correct / float(len(testing_examples)))

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

