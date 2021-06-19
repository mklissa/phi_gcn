import gym
import argparse
import numpy as np
from fourrooms import Fourrooms


import pdb
from scipy.special import expit
from scipy.special import logsumexp



def make_interpolater(left_min, left_max, right_min, right_max): 
    if left_min == left_max:
        return lambda x:x
    # Figure out how 'wide' each range is  
    leftSpan = left_max - left_min  
    rightSpan = right_max - right_min  

    # Compute the scale factor between left and right values 
    scaleFactor = float(rightSpan) / float(leftSpan) 

    # create interpolation function using pre-calculated scaleFactor
    def interp_fn(value):
        return right_min + (value-left_min)*scaleFactor

    return interp_fn



class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.,lr=0.1):
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp
        self.lr = lr

    def value(self, phi, action=None):
        if action is None:
            return self.weights[phi,:].squeeze()
        return self.weights[phi,action].squeeze()

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi,step):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))

    def add(self,weight):
        # pdb.set_trace()
        self.weights = np.vstack((self.weights,weight))

    def save(self,phi,action):
        self.last_phi = phi
        self.last_action = action

    def update(self, critic):
        phi =self.last_phi
        action = self.last_action

        actions_pmf = self.pmf(phi)
        self.weights[phi, :] -= self.lr*critic*actions_pmf
        self.weights[phi, action] += self.lr*critic






class StateValue:
    def __init__(self, discount, lr, weights):
        self.lr = lr
        self.discount = discount
        # self.terminations = terminations
        self.weights = weights
        self.elig_w = np.zeros_like(self.weights)

    def save(self, phi):
        self.last_phi = phi
        self.last_value = self.value(phi)

    def value(self, phi):
        return self.weights[phi]

    def init_elig(self,):
        self.elig_w = np.zeros_like(self.weights)

    def update_lambda(self, phi, reward, done):
        
        self.elig_w[self.last_phi] += 1.

        update_target = reward + 1.*self.discount *(1-done) * self.value(phi)

        tderror = update_target - self.last_value

        # pdb.set_trace()

        self.weights += self.lr * tderror * self.elig_w    

        self.elig_w  *= (self.discount * .95)
        
        # print(self.elig_w)
        # self.weights[self.last_phi] += self.lr*tderror


    def update(self, phi, reward, done):

        update_target = reward + 1.*self.discount *(1-done) * self.value(phi)

        tderror = update_target - self.last_value

        self.weights[self.last_phi] += self.lr*tderror


    def update2(self, update_target, phi):
        # pdb.set_trace()
        # update_target = reward + 1.*self.discount *(1-done) * val

        tderror = update_target - self.weights[phi]

        self.weights[phi] += self.lr*tderror



    def add(self,weight):
        # pdb.set_trace()
        self.weights = np.hstack((self.weights,weight))
        self.elig_w = np.hstack((self.elig_w,[0.]))



class ActionValue:
    def __init__(self, discount, lr, weights):
        self.lr = lr
        self.discount = discount
        # self.terminations = terminations
        self.weights = weights
        # self.qbigomega = qbigomega

    def value(self, phi, action):

        return self.weights[phi,action]
        # return np.sum(self.weights[phi, option, action], axis=0)

    def save(self, phi, action):
        self.last_phi = phi
        self.last_action = action

    def update(self, phi, reward, done, critic_value):
        # One-step update target
        update_target = reward + self.discount* (1-done) * critic_value

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_action)
        self.weights[self.last_phi, self.last_action] += self.lr*tderror

    def add(self,weight):
        # pdb.set_trace()
        self.weights = np.vstack((self.weights,weight))
























class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies

    def add_option(self,option_policies):
        self.option_policies = option_policies

    def update(self, phi, option, action, critic):
        # pdb.set_trace()
        actions_pmf = self.option_policies[option].pmf(phi)
        self.option_policies[option].weights[phi, :] -= self.lr*critic*actions_pmf
        self.option_policies[option].weights[phi, action] += self.lr*critic





class SigmoidTermination:
    def __init__(self, rng, nfeatures):
        self.rng = rng
        self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, option):
        magnitude, direction = self.terminations[option].grad(phi)
        self.terminations[option].weights[direction] -= \
                self.lr*magnitude*(self.critic.advantage(phi, option))

class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.



class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs




class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))
