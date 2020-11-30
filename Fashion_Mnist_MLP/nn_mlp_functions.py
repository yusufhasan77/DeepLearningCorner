"""This file contains all functions for a multi layer perceptron model"""
import numpy as np



def initialize_parameters_nn_HE(nn_dimen):
    """
    Function to initialize the parameters of the neural network using HE initialization
    
    Inputs: nn_dimen-> The dimensions of the neural network. e.g (12288,3,2,1)
    Outputs: parameters-> python dictionary containing parameters 'W' and 'b'
    """
    
    #We will use 'HE' initialization
    parameters = {} #parameters dictionary
    L = len(nn_dimen) #number of layers in our network
    
    #start of for loop to initialze the parameters
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(nn_dimen[l],nn_dimen[l-1]) * np.sqrt(2./nn_dimen[l-1])
        parameters["b"+str(l)] = np.zeros((nn_dimen[l],1))
    #end of for loop to initialze the parameters
    
    return parameters



"""Forward propagation and backward propagation implementation for Relu"""
def Relu(Z):
    """
    Relu activation function

    Inputs: Z-> output of matrix operation W[l]*A[l-1] + b

    Outputs : A[l] -> The activation of the current layer
              cache-> Value of A[l] stored in a python dictionary
    """

    Act_current = np.maximum(0,Z) #Relu
    assert(Act_current.shape == Z.shape) #Make sure that activation matrix is of correct shape
    Act_cache = Z #Useful when computing backpropogation
    
    return Act_current, Act_cache

def Relu_backprop(dJdA,Act_cache):
    """
    Function to compute the backward step for hidden units with Relu activation function

    Inputs: dJdA[l]-> Derivative of the cost with respect to Activation function for current layer

    Outputs: dJdZ[l]-> Derivative of the cost with respect to Z function for current layer
    """

    Z = Act_cache
    dJdZ = np.array(dJdA, copy=True) # To convert dJdz to a correct object.
    
    # When z <= 0, dJdz = 0 as well. The derivative is undefined here
    dJdZ[Z <= 0] = 0
    
    assert (dJdZ.shape == Z.shape)
    
    return dJdZ

def activation_forwardprop_Relu(Act_prev, W, b):
    """
    Implementation of activation function of forward propagation for hidden units that have Relu
    
    Inputs: Act_prev-> The activation matrix of previous layer, Act[l-1]
            W-> The weight matrix for the current layer W[l]
            b-> The bias matrix for the current layer b[l]
    
    Outputs: Act-> Activation value for the current layer Act[l]
             cache -> Values of Act_prev(Act[l-1]), W[l], b[l], used to compute Z[l] and the value of Z[l] as well
    """
    
    #Compute Z and save parameter values in cache
    Z = np.dot(W,Act_prev) + b # Z computation
    
    #Obtain Activation matrix by applying Relu
    Act, Act_cache = Relu(Z)
    
    cache = (Act_prev, W, b, Act_cache) #Merge into a single cache, cache is used in backpropagation
   
    return Act, cache


def activation_backwardprop_Relu(dJdA, cache, m):
    """
    Implementation of backward propagation for hidden units that have Relu
    
    Inputs: dJdA -> The derivative matrix of cost with respect to Activation in current layer, dJdA[l]
            cache -> Containing saved values of Act[l-1], W, b and Z
    
    Outputs: dJdW -> Derivatives of cost with respect to Weights in current layer
             dJdb -> Derivatives of cost with respect to biases in current layer
             dAdJ_prev -> Derivatives of cost with respect to Activations in previous layer  
    """
    
    Act_prev, W, b, Z = cache #Retrieve saved values from cache
    
    dJdZ = Relu_backprop(dJdA,Z) 
    assert(dJdZ.shape == Z.shape) #To Make sure dJdZ is of correct dimensions
    
    dJdW = np.dot(dJdZ,Act_prev.T) * (1./m) 
    assert(dJdW.shape == W.shape) #To Make sure dJdW is of correct dimensions
    
    dJdb = np.sum(dJdZ, axis=1, keepdims=True) * (1./m)
    assert(dJdb.shape == b.shape) #To Make sure dJdb is of correct dimensions
    
    dJdA_prev = np.dot(W.T,dJdZ) 
    assert(dJdA_prev.shape == Act_prev.shape) #To Make sure dJdA_prev is of correct dimensions
    
    return dJdW, dJdb, dJdA_prev




"""Softmax regression implementation"""
def softmax(Act_prev, W, b):
    """
    Implementation of softmax regression
    
    Inputs: Act_prev -> Activations of the previous layer (from Relu, tanh, etc.)
            W -> Weights matrix for the current layer W[l]
            b -> Biases matrix for the current layer b[l]
            
    Outputs: Act -> Softmax probabilities
             cache -> Values of Act_prev(Act[l-1]), W[l], b[l] and Z[l]
    """
    
    Z = np.dot(W,Act_prev) + b # Z computation
    
    Act = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    
    cache = (Act_prev, W, b, Z) #useful for backpropagation
    
    return Act, cache

def softmax_backward(Act_L, Y,cache):
    """
    Implementation of backward step of softmax
    
    Inputs: Act_L -> Activation of the last layer (softmax)
            cache -> Saved values of Act_prev(Act[l-1]), W[l], b[l] and Z[l]
            Y -> The ground truth
            m -> Number of training examples
    
    Outputs: dJdW -> Derivatives of cost with respect to Weights in current layer
             dJdb -> Derivatives of cost with respect to biases in current layer
             dAdJ_prev -> Derivatives of cost with respect to Activations in previous layer 
    """
    m = Y.shape[1]
    
    Act_prev, W, b, Z = cache #Retrieve saved values from cache
    
    dJdZ = Act_L - Y #Explained in notebook
    assert(dJdZ.shape == Z.shape) #To Make sure dJdZ is of correct dimensions
    
    dJdW = np.dot(dJdZ,Act_prev.T) * (1./m) 
    assert(dJdW.shape == W.shape) #To Make sure dJdW is of correct dimensions
    
    dJdb = np.sum(dJdZ, axis=1, keepdims=True) * (1./m)
    assert(dJdb.shape == b.shape) #To Make sure dJdb is of correct dimensions
    
    dJdA_prev = np.dot(W.T,dJdZ) 
    assert(dJdA_prev.shape == Act_prev.shape) #To Make sure dJdA_prev is of correct dimensions
    
    return dJdW, dJdb, dJdA_prev



"""Compiling all of the above in forward propagation -> cost -> backward propagation"""

def nn_mlp_forwardpropagation(X, parameters, L):
    """
    Complete forward propagation step for a multi-layer perceptron for multi-class classification
    
    Inputs: X -> The input features
            parameters -> The parameters of the model for all layers, W[l] and b[l]
    """
    
    Act_prev = X
    caches = [] #Python list to store all cached values
    
    #start of for loop to forward propagate through Relu layers of the network
    for l in range(1,L):
        
        #Retrieve parameters from the parameters dictionary
        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]
        #Apply the activation function
        Act_prev, cache = activation_forwardprop_Relu(Act_prev, W, b)
        caches.append(cache)
    #end of for loop to forward propagate through Relu layers of the network
        
    #Computing the softmax layer activation
    Act_L, cache = softmax(Act_prev,parameters["W"+str(L)], parameters["b"+str(L)])
    caches.append(cache)
    
    return Act_L, caches


def cross_entropy_loss(Act_L,Y):
    """
    Cross entropy loss function
    
    Inputs: Act_L -> The activation from the last layer (Softmax)
            Y -> The ground truth
            
    Outputs: cost -> The cost for current iteration
    """
    
    m = Y.shape[1]
    
    #Compute the cost
    cost = -(1/m) * np.sum(np.multiply(Y, np.log(Act_L)))
    #cost = np.squeeze(cost)
    
    #assert(cost.shape == ())
    
    return cost


def nn_mlp_backwardpropagation(derivatives, Act_L, Y, caches, L):
    """
    Complete backward propagation step for a multi-layer perceptron for multi-class classification
    
    Inputs: derivatives -> Python dictionary containing derivatives
            Act_L -> Activation of the last layer (Softmax)
            Y -> The ground truth
            caches -> List of cache containing the saved values A[l-1], W[l], b[l] and Z[l]
    """
    m = Y.shape[1]
    
    #Computation for the softmax layer
    derivatives["dJdW"+str(L)], derivatives["dJdb"+str(L)], derivatives["dJdA"+str(L-1)] = softmax_backward(Act_L, Y,caches[L-1])
    
    #Start of for loop to backpropagate through L-2 to 0th layers which have Relu activation function
    for l in reversed(range(L-1)):
        derivatives["dJdW"+str(l+1)], derivatives["dJdb"+str(l+1)], derivatives["dJdA"+str(l)] = activation_backwardprop_Relu(derivatives["dJdA"+str(l+1)],caches[l],m)
    #End of for loop to backpropagate through L-2 to 0th layers which have Relu activation function
        
    return derivatives