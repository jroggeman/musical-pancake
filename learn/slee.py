from svm_factory import supportvector
from nb_factory import naivebayes
from features import sentiment
from features import in_out
from features.feature_group import Group
from nn_model_factory import neuralnet
import itertools
from bow import bow
from scores import smog, readability


def mean(iter):
    return sum(iter) / float(len(iter))


def slee(seq, group_features, features, model, n):
    """
    Super Long Execution Engine.py (slee.py)
        - Name Credit: Joel Roggeman
    Recommend running overnight.
    """
    total_results = []
    for gf_i, _ in enumerate(group_features):
        for gfs in itertools.combinations(group_features, gf_i + 1):
            g = Group(list(gfs))
            for f_i, _ in enumerate(features):
                for fs in itertools.combinations(features, f_i + 1):
                    all_features = [g.call_all_features] + list(fs)
                    models = {
                        'nn': ('nn', lambda features, num: neuralnet(features, num, len(g.feature_list) + 5)),
                        'svm': ('svm', lambda features, num: supportvector(features, num)),
                        'nb': ('nb', lambda features, num: naivebayes(features, num))
                    }
                    if model in models.keys():
                        functions = [models[model]]
                    elif model == 'all':
                        functions = models.values()
                    else:
                        raise ValueError('Must supply a valid model name or "all"')

                    for name, mod in functions:
                        print ' {} '.format(name).center(30, '#')
                        for num in seq:
                            print ' Size = {} '.format(num).center(30, '-')
                            result = mod(all_features, num)
                            avg_accuracy = mean([result[key][2] for key in result])
                            total_results.append((avg_accuracy, num, name, list(gfs) + all_features[1:]))

    sorted_results = sorted(total_results, key=lambda x: x[0], reverse=True)  # sorts on accuracy high-low
    print ' RESULTS '.center(30, '*')
    for i, res in enumerate(sorted_results[:n]):
        print '{}: accuracy: {}, number of samples: {}, model: {}, functions: {}'.format(i + 1, *res)
    print ''.center(30, '*')


def main():
    """Change these 5 parameters as you see fit to test whatever you'd like.
       It will iterate through permutations of features and models (if you pick 'both') and returns
       the top n parameter settings
    """
    # 1) sequence: training example counts for each run. I'd recommend keeping this kinda low to see which parameter settings do best, then test higher numbers on those parameters
    seq = [2500]

    # 2) group_features: the functions that need examples passed in one at a time
    group_features = [sentiment.sentiment_variance, sentiment.raw_sentiment, smog, readability]

    # 3) features: the functions that can take all examples in at the same time
    features = [bow]

    # 4) model: which model to use. Only 'svm', 'nn', and 'both' work right now.
    # model = 'nn'
    # model = 'both'
    model = 'svm'

    # 5) n: the number of top results to display after executing. if n > the number of results, it prints all of them
    n = 5

    slee(seq, group_features, features, model, n)

if __name__ == '__main__':
    main()
