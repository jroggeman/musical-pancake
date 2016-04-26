from svm_factory import supportvector
from features import sentiment
from features.feature_group import Group
from nn_model_factory import neuralnet
from bow import bow
from scores import readability, smog


seq = [50,75,100,150,200,300,400,500,600,700,800,900,1000,1500]

g = Group([sentiment.sentiment_variance,sentiment.raw_sentiment, readability, smog])

print "NEURAL NET"
num_grouped = len(g.feature_list)
print num_grouped
neuralnet([g.call_all_features,bow],10,num_grouped+5)

print "###############################Sentiment Variance and Raw#####################################"
for num in seq:
    print "----------------------------------Size = %s---------------------------------------" %num
    supportvector([g.call_all_features,bow],num)



g = Group([sentiment.sentiment_variance])
print "###############################Sentiment Variance#####################################"
for num in seq:
    print "----------------------------------Size = %s---------------------------------------" %num
    supportvector([g.call_all_features],num)



g = Group([sentiment.raw_sentiment])
print "###############################Sentiment Raw#####################################"
for num in seq:
    print "----------------------------------Size = %s---------------------------------------" %num
    supportvector([g.call_all_features],num)