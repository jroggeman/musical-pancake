import sentiment

class Group:

    def __init__(self, list):
        self.feature_list = list

    """def make_calls(self, examples, train = None):
        /"/"/"here we'll put the features that we want to call/"/"/"
        return self.call_all_features([sentiment.sentiment_variance],examples)"""

    def call_all_features(self, examples, train = None):
        final = []
        for feat in self.feature_list:
            final.append([])

        for ex in examples:

            for ind,feat in enumerate(self.feature_list):
                final[ind].append(feat(ex))

        return final

    def fake_feature(selfself,examples, train = None):
        return [[3.5 for ex in examples]]
