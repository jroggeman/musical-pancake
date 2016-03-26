from stream_json import stream_json

class Example:
    def __init__(self, review, votes):
        self.review = review
        self.votes = votes

    def __str__(self):
        return "Review: \"" + self.review[0:100] + "...\""

    def is_correct(self, result):
        return (self.votes['useful'] > 0) == result

    __repr__ = __str__


def preprocess_file(filename):
    stream = stream_json(filename)

    examples = [Example(ex['text'], ex['votes']) for ex in stream]

    return examples

def stream_examples(filename, extract_features = None):
    stream = stream_json(filename)

    for ex in stream:
        e = Example(ex['text'], ex['votes'])
        if extract_features is not None:
            e = extract_features(e)

        yield e
