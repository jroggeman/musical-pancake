from preprocess import preprocess_file
from sklearn import svm


def feature_matrix(features, examples):
    return [[feat(ex) for feat in features] for ex in examples]


def answer_list(examples):
    return [ex.votes['useful'] > 0 for ex in examples]


def supportvector(features):
    examples = preprocess_file('../data/smaller_reviews.json')
    idx = int(len(examples) * .8)
    train_ex = examples[:idx]
    test_ex = examples[idx:]
    training, testing = feature_matrix(features, train_ex), feature_matrix(features, test_ex)
    train_ans, test_ans = answer_list(train_ex), answer_list(test_ex)
    classifier = svm.SVC()
    classifier.fit(training, train_ans)
    guesses = classifier.predict(testing)
    result = [a == b for a, b in zip(guesses, train_ans)]
    print sum(result) / float(len(result))


if __name__ == '__main__':
    supportvector([
        lambda x: len(x.review),
        lambda x: 1 if 'good' in x.review else 0,
        lambda x: 1 if 'not' in x.review else 0
    ])
