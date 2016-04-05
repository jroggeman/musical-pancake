import nltk

def tokenize(review):
    """Breaks review into tokens"""
    return nltk.word_tokenize(review)

def tag_pos(review):
    """Tags review tokens with their parts of speech"""
    return nltk.pos_tag(tokenize(review))

def pos_count(pos, review):
    """Count of tokens tagged with `pos` in the review"""
    return sum(1 for token, tag in tag_pos(review) if tag == pos)

