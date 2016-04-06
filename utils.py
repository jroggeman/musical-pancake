import nltk
import nltk.sentiment.util

from collections import namedtuple

class CacheData:
    def __init__(self):
        self.hash, self.text, self.data = None, None, None

    def matches(self, text):
        return hash(text) == self.hash and text == self.text

    def update(self, text, data):
        self.hash = hash(text)
        self.text = text
        self.data = data


TOKEN_CACHE = CacheData()
TAG_CACHE = CacheData()
SENT_TOKEN_CACHE = CacheData()
SENTIMENT_CACHE = CacheData()

def text_cached(cache):
    def dec(func):
        def new_func(text):
            if cache.matches(text):
                result = cache.data
            else:
                result = func(text)
                cache.update(text, result)
            return result
        return new_func
    return dec

@text_cached(TOKEN_CACHE)
def tokenize(text):
    return nltk.word_tokenize(text)

@text_cached(TAG_CACHE)
def pos_tag(text):
    return nltk.pos_tag(tokenize(text))

@text_cached(SENT_TOKEN_CACHE)
def sentence_tokenize(text):
    return nltk.tokenize.sent_tokenize(text)


@text_cached(SENTIMENT_CACHE)
def sentiment(text):
    sentences=sentence_tokenize(text)
    vader_analyzer = nltk.sentiment.SentimentIntensityAnalyzer()
    results = [vader_analyzer.polarity_scores(sentence) for sentence in sentences]
    return results

