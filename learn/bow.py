from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

vectorizer = None
selector = None

def bow(examples, is_training, number_of_features=5):
    """Calculates the bag of words features.

    Arguments:
    examples -- A list of Example objects to get features from
    is_training -- A boolean representing whether or not this is a training or
    testing call

    Keyword arguments:
    number_of_features -- The number of features to extract, the default being
    5
    """

    examples, votes = zip(*[(e.review, e.votes['useful'] > 0) for e in examples])

    # Store ugly global state
    global vectorizer
    global selector

    # Initialize if first (training) call
    if is_training:
        vectorizer = TfidfVectorizer()
        counts = vectorizer.fit_transform(examples).toarray()

        selector = SelectKBest(chi2, k=number_of_features)
        return selector.fit_transform(counts, votes)
    else:
        counts = vectorizer.transform(examples).toarray()
        return selector.transform(counts)

