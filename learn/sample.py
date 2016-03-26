import main
import sys

def progress_bar(current_example):
    progress = int(current_example / 157.14)
    sys.stdout.write('\r[{0}] {1}% ({2}/1569264)'.format('#'*progress + ' '*(100-progress), progress, current_example))
    sys.stdout.flush()

    # Each model has .review and .votes
def train_model(stream, train_indices, test_indices):
    print('Training model...')
    sum_useful = 0
    total_sum = 0
    testing = []
    for i, example in enumerate(stream):
        total_sum += 1
        if i in train_indices:
            if example.votes['useful'] > 0:
                sum_useful += 1

        else:
            testing.append(example)

        progress_bar(i)


    percent_useful = float(sum_useful) / total_sum

    return percent_useful > 0.5, testing

def model_test(model, example):
    return model

main.engage(train_model, model_test, file='../data/smaller_reviews.json')
