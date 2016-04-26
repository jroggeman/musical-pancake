from textstat.textstat import textstat


def readability(example):
    if example is None:
        return 100
    return textstat.flesch_reading_ease(example.review)
