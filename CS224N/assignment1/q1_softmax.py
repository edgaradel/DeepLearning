import numpy as np

def softmax(x):
    orig_shape=x.shape

    if len(x.shape)>1:
        #Matrix
        ###Your Code Here
        #函数定义
        exp_minmax=lambda x:np.exp(x-np.max(x))
        denom=lambda x:1.0/np.sum(x)
        #利用定义的函数对矩阵的每一个元素进行处理
        x=np.apply_along_axis(exp_minmax, 1, x)
        denominator=np.apply_along_axis(denom, 1, x)

        if len(denominator.shape)==1:
            denominator=denominator.reshape((denominator.shape[0],1))

        x=x*denominator
        ###END YOUR CODE
    else:
        #Vecctor
        ###YOUR CODE HERE
        x_max=np.max(x)
        x=x-x_max #常量不变性
        numerator=np.exp(x)
        denominator=1.0/np.sum(numerator)
        x=numerator.dot(denominator) #softmax
        ##END YOUR CODE

    #check the shape
    assert x.shape==orig_shape
    return x

def test_softmax_basic():
    print("Running basic tests...")
    test1=softmax(np.array([1,2]))
    print(test1)
    ans1=np.array([0.26894142, 0.73105858])
    assert np.allclose(test1,ans1,rtol=1e-05,atol=1e-06)

    test2=softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2=np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2,ans2,rtol=1e-05,atol=1e-06)

    test3=softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3=np.array([0.73105858, 0.26894142])
    assert np.allclose(test3,ans3,rtol=1e-05,atol=1e-06)
    print("You should be able to verify these results by hand!\n")

def tese_softmax():
    print("Running your tests...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == '__main__':
    test_softmax_basic()
    tese_softmax()


