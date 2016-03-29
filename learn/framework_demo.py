from __future__ import division
from framework import Model, Framework


def pr_then_given_and(review):
    """ Calculates the probability of seeing 'then' after the word 'and' in this review
        Case insensitive, but considers 'and' and 'and?' two different words.
    """
    review = review.lower()
    bigrams = zip(review.split(' '), review.split(' ')[1:])
    ands = len(filter(lambda x: x[0] == 'and', bigrams))
    and_thens = len(filter(lambda x: x == ('and', 'then'), bigrams))
    try:
        return and_thens / ands
    except ZeroDivisionError:
        return 0.


def main():
    # First, we can declare a list of features
    feats = [
        lambda x: "delicious" in x.review,
        lambda x: len(x.review),
        lambda x: pr_then_given_and(x.review)
    ]
    # We also need to declare a label function to determine what the Examples classify as
    # in our case, if they are useful or not
    lab = lambda x: x.votes['useful'] > 0
    # Pass these in as keyword arguments along with the number of samples to use for training and testing,
    # the number of hidden layers in the neural net, and the train/test split ratio
    my_smart_model = Model(features=feats, label=lab, n=50, hidden_layers=3, split=0.5)
    # Now we can give our model to the framework
    fw = Framework(model=my_smart_model)
    # Running the framework gets the data, trains the model, and calculates some evaluation metrics
    fw.run()
    # We can access the metrics as properties of the framework.
    print 'Accuracy:', fw.accuracy
    print 'F1 score:', fw.f1_score


if __name__ == '__main__':
    main()
