import nltk
import utils

def pos_count(pos, review):
    """Count of tokens tagged with `pos` in the review"""
    tags = zip(utils.pos_tag(review))[1]
    return sum(1 for token, tag in tags if tag == pos)

