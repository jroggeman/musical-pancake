import enchant
import utils
import re

DICTIONARY = enchant.Dict('en_US')
has_char = re.compile(r".*\w.*").match

def in_out(review):
    tokens = [token for token in utils.tokenize(review) if has_char(token)]  # filter out tokens that are only punctuation
    if tokens:
        return sum(1.0 for t in tokens if DICTIONARY.check(t.lower())) / len(tokens)
    else:
        return 0.0  # default behavior for no tokens

