import nltk
import utils
import numpy


def sentiment_variance(review):
    """variance of compound sentiment polarities between sentences in review"""
    polarities = utils.sentiment(review.review)
    return numpy.var([x['compound'] for x in polarities],dtype=numpy.float64)


