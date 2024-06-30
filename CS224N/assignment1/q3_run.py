import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from q3_word2vec import *
from q3_sgd import *

# Data loading
# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset=StanfordSentiment()
tokens=dataset.tokens()
nWords=len(tokens) # 19539 the number of different words
# We are going to train 10-dimensional vector for this assignment
dimVecctors=10

# Context size
C=5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

# Randomly generate word vectors and continuously train using skip-gram.
startTime=time.time()
wordVectors=np.concatenate(
    ((np.random.rand(nWords,dimVecctors)-0.5)/
     dimVecctors,np.zeros((nWords,dimVecctors))),
    axis=0) # (2*19539, 10) = (39078, 10)
wordVectors=sgd(
    lambda vec:word2vec_sgd_wrapper(skipgram,tokens,vec,dataset,C,
        negSamplingCostAndGradient),
    wordVectors,0.3,40000,None,True,PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time()-startTime))

# concatenate the input and output word vectors
wordVectors=np.concatenate(
    (wordVectors[:nWords,:],wordVectors[nWords:,:]),
    axis=0)
# wordVectors=wordVectors[:nWords,:]+wordVectors[nWords:,:]

# Data visualization: dimensionality reduction, using singular value decomposition to reduce word vectors from 25 dimensions to 2 dimensions, thereby achieving 2D visualization on a plane
visualizeWords=[
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"] # 25

visualizeIdx=[tokens[word] for word in visualizeWords]
visualizeVecs=wordVectors[visualizeIdx,:]
temp=(visualizeVecs-np.mean(visualizeVecs,axis=0)) # 25*10
covariance=1.0/len(visualizeIdx)*temp.T.dot(temp)
U,S,V=np.linalg.svd(covariance) # U-(10, 10) S-(10,) V-(10, 10)
coord=temp.dot(U[:,0:2]) # (25, 2)

for i in range(len(visualizeWords)):
    plt.text(coord[i,0],coord[i,1],visualizeWords[i],
        bbox=dict(facecolor='green',alpha=0.1))

plt.xlim((np.min(coord[:,0]),np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]),np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')
