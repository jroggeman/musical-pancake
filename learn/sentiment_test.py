from svm_factory import supportvector
from features import sentiment
from features.feature_group import Group
from nn_model_factory import neuralnet
from bow import bow
from scores import readability, smog
from features import pos


seq = [15]

g = Group([sentiment.sentiment_variance, sentiment.raw_sentiment, sentiment.strong_sentiment, pos.adj_count, pos.adj_entropy, readability, smog])



print "###############################all svm#####################################"
for num in seq:
    print "----------------------------------Size = %s---------------------------------------" %num
    supportvector([g.call_all_features,bow],num)

