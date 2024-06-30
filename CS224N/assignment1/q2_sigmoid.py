import numpy as np
def sigmoid(x):
    ### YOUR CODE HERE
    s=1.0/(1+np.exp(-x))
    ### END YOUR CODE

    return s

def sigmoid_grad(s):
    ### YOUR CODE HERE
    ds=s*(1-s)
    ### END YOUR CODE
    return ds

def test_sigmoid_basic():
    print("Running basic tests...")
    x=np.array([[1,2],[-1,-2]])
    f=sigmoid(x)
    g=sigmoid_grad(f)
    print(f)
    f_ans=np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f,f_ans,rtol=1e-05,atol=1e-06)
    print("You should verify these results by hand!\n")

def test_sigmoid():
    print("Runnning your tests...")
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == '__main__':
    test_sigmoid_basic()
    test_sigmoid()