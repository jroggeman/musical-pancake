import nltk

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

