from textstat.textstat import textstat


def readability(example):
    if not example:
        return 100
    return textstat.flesch_reading_ease(example.review)

def smog(example):
    if not example:
        return 0

    return textstat.smog_index(example.review)