"""
This file contains the ADAM optimization algorithm. Two functions are defined its initialization
and its implementation to update parameters
"""

import numpy as np

def initialize_parameters_adam(parameters):
    """
    Function to initialize parameters for the ADAM optimization algorithm

    Inputs: parameters-> python dictionary containing our parameters
    Ouputs: v-> python dictionary containing 'v', moving average of the first gradient
            s-> python dictionary containing 's', moving average of the squared gradient
    """
    
    L = len(parameters) // 2 #number of layers in neural network, we are initializing 2 parameters 'W' and 'b' for each layer
    v = {} #Dictionary for 'v' moving average of the first gradient
    s = {} #Dictionary for 's' moving average of the squared gradient
    
    for l in range(L):
        #initialize 'v'
        v["dJdW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["dJdb"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
        #initialize 's'
        s["dJdW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["dJdb"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
    return v, s


def update_parameters_ADAM(parameters, derivatives, v, s, t, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """
    Algorithm to update parameters using ADAM

    Inputs: parameters -> python dictionary containing parameters that need to be optimized
            derivatives -> python dictionary containing derivatives of parameters that need to be optimized
            v -> Python dictionary containing 'v', moving average of the first gradient
            s -> Python dictionary containing 's', moving average of the squared gradient
            beta_1 -> Exponential decay hyperparameter for the first moment estimates 
            beta_2 -> Exponential decay hyperparameter for the second moment estimates 
            epsilon -> hyperparameter to avoid division by zero in Adam updates
            learing_rate -> The learning rate
    
    Outputs: parameters -> Python dictionary containing our model paramaters
             v -> Python dictionary containing 'v', moving average of the first gradient, updated
             s -> Python dictionary containing 's', moving average of the squared gradient, updated
    """

    L = len(parameters) //2 ##number of layers in neural network, we are initializing 2 parameters 'W' and 'b' for each layer
    v_corr = {} #bias corrected 'v'
    s_corr = {} #bias corrected 's'
    
    #Start of for loop to update all parameters
    for l in range(L):

        # Moving average of the first gradients
        v["dJdW"+str(l+1)] = (beta_1*v["dJdW"+str(l+1)]) + ((1-beta_1) * derivatives["dJdW"+str(l+1)])
        v["dJdb"+str(l+1)] = (beta_1*v["dJdb"+str(l+1)]) + ((1-beta_1) * derivatives["dJdb"+str(l+1)])
        # Moving average of the squared gradients
        s["dJdW"+str(l+1)] = (beta_2*s["dJdW"+str(l+1)]) + ((1-beta_2) * np.power(derivatives["dJdW"+str(l+1)],2))
        s["dJdb"+str(l+1)] = (beta_2*s["dJdb"+str(l+1)]) + ((1-beta_2) * np.power(derivatives["dJdb"+str(l+1)],2))

        # Compute bias-corrected first moment estimate
        v_corr["dJdW"+str(l+1)] = v["dJdW"+str(l+1)]/ (1- (np.power(beta_1,t)))
        v_corr["dJdb"+str(l+1)] = v["dJdb"+str(l+1)]/ (1- (np.power(beta_1,t)))
        # Computation for bias-corrected second moment estimate
        s_corr["dJdW"+str(l+1)] = s["dJdW"+str(l+1)]/ (1- (np.power(beta_2,t)))
        s_corr["dJdb"+str(l+1)] = s["dJdb"+str(l+1)]/ (1- (np.power(beta_2,t)))
        
        # Update parameters
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] -(learning_rate * (v_corr["dJdW"+str(l+1)] /np.sqrt(s_corr["dJdW"+str(l+1)] + epsilon)))
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] -(learning_rate * (v_corr["dJdb"+str(l+1)] /np.sqrt(s_corr["dJdb"+str(l+1)] + epsilon)))

    #End of for loop to update all parameters
    
    return parameters, v, s