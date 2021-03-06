from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

vectorizer = None
decomposer = None
#normalizer = None
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
    #global normalizer
    global decomposer
    global selector

    # Initialize if first (training) call
    if is_training:
        vectorizer = TfidfVectorizer()
        counts = vectorizer.fit_transform(examples).toarray()

        #normalizer = Normalizer()
        #counts = normalizer.fit_transform(counts)

        # decompose to 100 for LSA
        decomposer = TruncatedSVD(n_components=100)
        counts = decomposer.fit_transform(counts)

        k = min(number_of_features, counts.shape[1])

        selector = SelectKBest(f_classif, k=k)
        array_result = selector.fit_transform(counts, votes)

        final = []
        for feat in range(len(array_result[0])):
            final.append([])

        for ex in array_result:
            for ind,feat_ex in enumerate(ex):
                final[ind].append(feat_ex)
        return final

    else:
        counts = vectorizer.transform(examples).toarray()
        #counts = normalizer.transform(counts)
        counts = decomposer.transform(counts)
        array_result =  selector.transform(counts)

        final = []
        for feat in range(len(array_result[0])):
            final.append([])

        for ex in array_result:
            for ind,feat_ex in enumerate(ex):
                final[ind].append(feat_ex)
        return final


