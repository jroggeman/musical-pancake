from svm_factory import supportvector
from features import sentiment
from features.feature_group import Group
from nn_model_factory import neuralnet
from bow import bow


def mean(iter):
    return sum(iter) / float(len(iter))


def slee(seq, group_features, features, model, n):
    """
    Super Long Execution Engine.py (slee.py)
        - Name Credit: Joel Roggeman
    Recommend running overnight.
    """
    g = Group(group_features)
    all_features = [g.call_all_features] + features
    models = {
        'nn': ('nn', lambda features, num: neuralnet(features, num, len(g.feature_list) + 5)),
        'svm': ('svm', lambda features, num: supportvector(features, num))
    }
    if model in models:
        functions = [models[model]]
    else:
        functions = [models['nn'], models['svm']]

    total_results = []
    for name, model in functions:
        print ' {} '.format(name).center(30, '#')
        for num in seq:
            print ' Size = {} '.format(num).center(30, '-')
            result = model(all_features, num)
            avg_accuracy = mean([result[key][2] for key in result])
            total_results.append((avg_accuracy, num, name, all_features))
    sorted_results = sorted(total_results, key=lambda x: x[0], reverse=True)  # sorts on accuracy high-low
    print ' RESULTS '.center(30, '*')
    for i, res in enumerate(sorted_results[:n]):
        print '{}: accuracy: {}, model: {}, functions: {}'.format(i + 1, *res)
    print ''.center(30, '*')


def main():
    """Runs slee(). Change these 5 parameters as you see fit to test whatever you'd like."""
    # 1) sequence: training example counts for each run
    seq = [10, 20]

    # 2) group_features: the functions that need examples passed in one at a time
    group_features = [sentiment.sentiment_variance, sentiment.raw_sentiment]

    # 3) features: the functions that can take all examples in at the same time
    features = [bow]

    # 4) model: which model to use. Only 'svm', 'nn', and 'both' work right now.
    # model = 'nn'
    # model = 'svm'
    model = 'both'

    # 5) n: the number of top results to display after executing. if n > the number of results, it prints all of them
    n = 2

    slee(seq, group_features, features, model, n)

if __name__ == '__main__':
    main()