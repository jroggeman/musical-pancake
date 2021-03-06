import nltk
import utils
import numpy


def sentiment_variance(review):
    """variance of compound sentiment polarities between sentences in review"""
    polarities = utils.sentiment(review.review)
    return numpy.var([x['compound'] for x in polarities],dtype=numpy.float64)

def raw_sentiment(review):
    return sum(x['compound'] for x in utils.sentiment(review.review))

def strong_sentiment(review):
    count = 0
    for x in utils.sentiment(review.review):
        if x['neg'] > 0.5:
            count -=1
        elif x['pos'] > 0.5:
            count +=1

    return count


