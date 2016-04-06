import nltk
import utils
import numpy

#as I understand it, the sentiment analysis will be cached, so that getting individual sentences
#will work?
#need to decide if I want to keep anything but the compound scores from the dict

def sentiment_variance(review):
    """variance of compound sentiment polarities between sentences in review"""
    polarities = utils.sentiment(review)
    """x =  [y['compound'] for y in polarities]
    print x
    def mean(z):
        return sum(z)/len(z)
    avg = mean(x)
    print avg
    print [(y-avg)**2 for y in x]
    var = mean([(y-avg)**2 for y in x])
    print var"""



    return numpy.var([x['compound'] for x in polarities],dtype=numpy.float64)



print sentiment_variance("The movie was bad. The movie was bad. The movie was bad. The movie was bad. ")