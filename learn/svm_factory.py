from sklearn import svm
from main import engage


def supportvector(features, sample_size=500):

    def call_all_features(features, examples, train):
        final = []
        for feat in features:
            f = feat(examples,train)
            if type(f[0]) == list:
                final += f
            else:
                final.append(f)

        return zip(*final)


    def answer_list(examples):
        return [ex.votes['useful'] > 0 for ex in examples]

    def initialize_models():
        return [SVMModel() for _ in range(5)]

    def train_model(model, examples):
        train = call_all_features(features, examples, True)
        ans = answer_list(examples)
        model.classifier.fit(train, ans)
        return model

    def model_test(model, example):  # TODO modify so it takes multiple examples at once? use call_all_features()
        return model.classifier.predict(call_all_features(features, [example], False))

    jole = Jole(initialize_models, train_model, model_test)
    return engage(jole, filename='../data/smaller_reviews.json', stochastic=False, sample_size=sample_size)


class SVMModel(object):
    def __init__(self):
        self.classifier = svm.SVC()


class Jole(object):
    def __init__(self, initialize_models, train_model, model_test):
        self.initialize_models = initialize_models
        self.train_model = train_model
        self.model_test = model_test


if __name__ == '__main__':
    def length(examples, train):
        return [len(x.review) for x in examples]

    def isgoodinit(examples, train):
        return ['good' in x.review for x in examples]

    supportvector([
        length,
        isgoodinit,
    ], sample_size=3000)
