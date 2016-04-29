from svm_factory import supportvector
from features import sentiment
from features.feature_group import Group
from nn_model_factory import neuralnet
from bow import bow
from scores import readability, smog
from features import pos
import traceback

num = 5000

g = Group([sentiment.sentiment_variance])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()

g = Group([sentiment.sentiment_variance, sentiment.raw_sentiment, sentiment.strong_sentiment, smog])
print "svm".center(30,"#")
print str(g.feature_list)+"bow"
try:
    supportvector([g.call_all_features,bow],num)
except:
    traceback.print_exc()



print "svm".center(30,"#")
print "bow"
try:
    supportvector([bow],num)
except:
    traceback.print_exc()


g = Group([smog])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()


g = Group([sentiment.sentiment_variance])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()


g = Group([sentiment.raw_sentiment])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()


g = Group([sentiment.sentiment_variance, sentiment.strong_sentiment])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()


g = Group([sentiment.sentiment_variance, sentiment.strong_sentiment, smog])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()


g = Group([sentiment.sentiment_variance, sentiment.raw_sentiment, sentiment.strong_sentiment])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()

g = Group([sentiment.sentiment_variance,smog])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()

g = Group([sentiment.raw_sentiment, smog])
print "svm".center(30,"#")
print str(g.feature_list)
try:
    supportvector([g.call_all_features],num)
except:
    traceback.print_exc()


g = Group([sentiment.raw_sentiment, smog])
print "svm".center(30,"#")
print str(g.feature_list)+"bow"
try:
    supportvector([g.call_all_features,bow],num)
except:
    traceback.print_exc()