import numpy as np
import random

def gradcheck_naive(f,x):
    rndstate=random.getstate()
    random.setstate(rndstate)
    fx,grad=f(x) # Evaluate function value at orginal point
    h=1e-4 # Do not chang this!

    # Iterate over all indexes in x
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        ix=it.multi_index

        # Try modifying x[ix] with h defined above to compute numerical gradient.
        ### YOUR CODE HERE
        x[ix]+=h
        random.setstate(rndstate)
        new_f1=f(x)[0]

        x[ix]-=2*h
        random.setstate(rndstate)
        new_f2=f(x)[0]

        x[ix]+=h
        numgrad=(new_f1-new_f2)/(2*h)
        ### END YOUR CODE

        # Compare gradients
        reldiff=abs(numgrad-grad[ix])/max(1,abs(numgrad),abs(grad[ix]))
        if reldiff >1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" %(grad[ix],numgrad))
            return
        it.iternext()
    print("Gradient check passed!")

def sanity_check():
    quad=lambda x:(np.sum(x**2),x*2)

    print("Running sanity checks...")
    gradcheck_naive(quad,np.array(123.456)) # scalar test
    gradcheck_naive(quad,np.random.randn(3,)) # 1-D test
    gradcheck_naive(quad,np.random.randn(4,5)) # 2-D test
    print("")

def your_sanity_checks():
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

if __name__ == '__main__':
    sanity_check()
    your_sanity_checks()