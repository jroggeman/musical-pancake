import sentiment

def make_calls(examples):
    """here we'll put the features that we want to call"""
    return call_all_features([sentiment.sentiment_variance],examples)

def call_all_features(features,examples):
    return [[feat(ex) for feat in features] for ex in examples]
