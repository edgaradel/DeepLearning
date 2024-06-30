import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid,sigmoid_grad

def normalizeRows(x):

    ### YOUR CODE HERE
    # Transform each element in the array, axis=0 means summing columns, axis=1 means summing rows."
    denom=np.apply_along_axis(lambda x:np.sqrt(x.T.dot(x)),1,x)
    x/=denom[:,None] # denom (2,)->(2,1)
    ### END YOUR CODE
    return x

def test_normalize_rows():
    print("Testing normalizationRows...")
    x=normalizeRows(np.array([[3.0,4.0],[1,2]]))
    print(x)
    ans=np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x,ans,rtol=1e-05,atol=1e-06)
    print("")

def softmaxCostAndGradient(predicted,target,outputVectors,dataset):

    ### YOUR CODE HERE

    ## Gradient for $\hat{v}$
    # Calculate the prediction:
    vhat=predicted # (3,)
    z=np.dot(outputVectors,vhat) # (5,)
    preds=softmax(z) # the column vector of the softmax prediction of words

    # Calculate the cost: the cross entropy function
    cost=-np.log(preds[target])

    # Gradients
    z=preds.copy()
    z[target]-=1.0

    grad=np.outer(z,vhat) # (5, 3) gradients for the "output" word vectors U (outputVectors)
    gradPred=np.dot(outputVectors.T,z) # (3,) gradients for the "input" word vectors v (predicted)
    ### END YOUR CODE
    return cost,gradPred,grad

def getNegativeSamples(target,dataset,K):
    """Sample K indexes which are not the target"""

    indices=[None] * K
    for k in range(K):
        newidx=dataset.sampleTokenIdx()
        while newidx==target:
            newidx=dataset.sampleTokenIdx()
        indices[k]=newidx
    return indices

def negSamplingCostAndGradient(predicted,target,outputVectors,dataset,K=10):

    # Generate the K negative samples (words), which aren't the expected output
    indices=[target]
    indices.extend(getNegativeSamples(target,dataset,K))

    ### YOUR CODE HERE
    grad=np.zeros(outputVectors.shape) # (5,3) the gradients for the "output" word vectors U (outputVectors)
    gradPred=np.zeros(predicted.shape) # (3,) the gradient for the predicted vector v_c (predicted)
    cost=0
    z=sigmoid(np.dot(outputVectors[target],predicted))
    cost-=np.log(z)
    grad[target]+=predicted*(z-1.0) # (3,) the gradients for u_o
    gradPred+=outputVectors[target]*(z-1.0)

    for k in range(K):
        samp=indices[k+1]
        z=sigmoid(np.dot(outputVectors[samp],predicted))
        cost-=np.log(1.0-z)
        grad[samp]+=predicted*z # (3,) the gradients for u_k
        gradPred+=outputVectors[samp]*z # (3,) the gradients for v_c
    ### END YOUR CODE

    return cost,gradPred,grad

def skipgram(currentWord,C,contextWords,tokens,inputVectors,outputVectors,
             dataset,word2vecCostAndGradient=softmaxCostAndGradient):
    cost=0.0
    gradIn=np.zeros(inputVectors.shape) # (5,3)
    gradOut=np.zeros(outputVectors.shape) # (5,3)

    ### YOUR CODE HERE
    cword_idx=tokens[currentWord]
    vhat=inputVectors[cword_idx] # the input word vector for the center word_c

    for j in contextWords: # 2*5=10
        u_idx=tokens[j] # the output word vector for the word u_o
        c_cost,c_grad_in,c_grad_out=\
            word2vecCostAndGradient(vhat,u_idx,outputVectors,dataset)
        cost+=c_cost
        gradIn[cword_idx]+=c_grad_in # (5,3) the gradients for the input vector
        gradOut+=c_grad_out # (5,3) the gradients for the output vector
    ### END YOUR CODE

    return cost,gradIn,gradOut

def cbow(currentWord,C,contextWords,tokens,inputVectors,outputVectors,
         dataset,word2vecCostAndGradient=softmaxCostAndGradient):
    cost=0.0
    gradIn=np.zeros(inputVectors.shape)
    gradOut=np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    predicted_indices=[tokens[word] for  word in contextWords]
    predicted_vectors=inputVectors[predicted_indices]
    predicted=np.sum(predicted_vectors,axis=0)
    target=tokens[currentWord]
    cost,gradIn_predicted,gradOut=word2vecCostAndGradient(predicted,target,outputVectors,dataset)
    for i in predicted_indices:
        gradIn[i]+=gradIn_predicted
    ### END YOUR CODE
    return cost,gradIn,gradOut

########################################
# Testing functions below. DO NOT MODIFY! #
########################################

def word2vec_sgd_wrapper(word2vecModel,tokens,wordVectors,dataset,C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize=50
    cost=0.0
    grad=np.zeros(wordVectors.shape)
    N=wordVectors.shape[0]
    inputVectors=wordVectors[:N//2,:]
    outputVectors=wordVectors[N//2:,:]
    for i in range(batchsize):
        C1=random.randint(1,C)
        centerword,context=dataset.getRandomContext(C1) # 得到词序列1+2C1

        if word2vecModel==skipgram:
            denom=1
        else:
            denom=1

        c,gin,gout=word2vecModel(
            centerword,C1,context,tokens,inputVectors,outputVectors,
            dataset,word2vecCostAndGradient)
        cost+=c/batchsize/denom
        grad[:N//2,:]+=gin/batchsize/denom
        grad[N//2:,:]+=gout/batchsize/denom
    return cost,grad

def test_word2vec():
    """Interface to the dataset for negative sampling"""
    dataset=type('dummy',(),{})()

    def dummySampleTokenIdx(): # 生成[0,4]之间的随机整数
        return random.randint(0,4)

    def getRandomContext(C): # 生成长度为1+2C的随机数列
        tokens=["a","b","c","d","e"]
        return tokens[random.randint(0,4)],\
                [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx=dummySampleTokenIdx
    dataset.getRandomContext=getRandomContext

    random.seed(31415)
    np.random.seed(92665)
    dummy_vectors=normalizeRows(np.random.randn(10,3))
    dummy_tokens=dict([("a",0),("b",1),("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    """
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram,dummy_tokens,vec,dataset,5,softmaxCostAndGradient),
                    dummy_vectors)

    """    
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
"""
    print("\n==== Gradient check for CBOW ====")
    gradcheck_naive(lambda vec:word2vec_sgd_wrapper(
        cbow,dummy_tokens,vec,dataset,5,softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c",3,["a","b","e","d","b","c"],
                   dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset))
    print(skipgram("c",1,["a","b"],
                   dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset,
                   negSamplingCostAndGradient))
    print(cbow("a",2,["a","b","c","a"],
               dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset))
    print(cbow("a",2,["a","b","a","c"],
               dummy_tokens,dummy_vectors[:5,:],dummy_vectors[5:,:],dataset,
               negSamplingCostAndGradient))        
    """

if __name__ == '__main__':
    #test_normalize_rows()
    test_word2vec()
















