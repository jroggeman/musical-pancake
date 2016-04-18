from svm_factory import supportvector
from features import sentiment
from features.feature_group import Group


seq = [10,20,50,75,100,150,200,300,400,500,600,700,800,900,1000,1500]

g = Group([sentiment.sentiment_variance, sentiment.raw_sentiment])
print "###############################Sentiment Variance and Raw#####################################"
for num in seq:
    print "----------------------------------Size = %s---------------------------------------" %num
    supportvector([g.call_all_features],num)



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