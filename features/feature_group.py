
def call_all_features(features,examples):
     return [[feat(ex) for feat in features] for ex in examples]
