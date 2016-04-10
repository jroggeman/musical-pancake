import nltk
import utils
import numpy


def sentiment_variance(review):
    """variance of compound sentiment polarities between sentences in review"""
    polarities = utils.sentiment(review)
    return numpy.var([x['compound'] for x in polarities],dtype=numpy.float64)



print sentiment_variance("The movie was bad. I like pizza. The movie was bad. The movie was very BAD!")