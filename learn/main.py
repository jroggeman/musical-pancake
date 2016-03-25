from random import shuffle
import preprocess

def engage(train_model, model_test, file = '../data/reviews.json', additional_preprocess = None):
    example_stream = preprocess.stream_examples(file, additional_preprocess)

    # TODO Modify the rest to handle the new stream
    chunk_size = len(all_examples) / 5
    accuracies = []
    for i in range(5):
        # Calculate indices for boundaries of test fold
        left_boundary = i * chunk_size
        right_boundary = left_boundary + chunk_size

        # Extract training and testing lists
        training_examples = all_examples[:left_boundary] + all_examples[right_boundary:]
        testing_examples = all_examples[left_boundary:right_boundary]

        model = train_model(training_examples)

        number_correct = evaluate(model, model_test, testing_examples)

        print(i + '. Results:')

        print(number_correct + '/' + len(testing_examples) + ' = ' + (number_correct / float(len(testing_examples))) + '%')

        accuracies.append(number_correct / float(len(testing_examples)))

    overall_accuracy = mean(accuracies)

    print('Overall accuracy: ' + overall_accuracy)

def evaluate(model, model_test, testing_examples):
    number_correct = 0
    for example in testing_examples:
        result = model_test(model, example)
        if example.is_correct(result):
            number_correct += 1

    return number_correct
