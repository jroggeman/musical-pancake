import nltk
import utils
import math
import collections

ADJ_SYM = 'JJ'

def pos_count(pos, review):
    """Count of tokens tagged with `pos` in the review"""
    tags = zip(utils.pos_tag(review.review))[1]
    return sum(1 for token, tag in tags if tag == pos)

def adj_count(review):
    return pos_count(ADJ_SYM, review)

def adj_entropy(review):
    tokens = utils.pos_tag(review.review)
    adjectives = [word for word, tag in tokens if tag == ADJ_SYM]
    total = float(len(adjectives))
    entropy = 0.
    counts = collections.Counter(adjectives)
    for word in counts:
        prob = float(counts[word]) / total
        entropy -= prob * math.log(prob)
    return entropy

