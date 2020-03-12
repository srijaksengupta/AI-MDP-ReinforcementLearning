from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # As per previous hint, we can make use of util.Counter()
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # Done as instructed above
        if (state, action) not in self.q_values:
            q_node_val = 0.0
        else:
            q_node_val = self.q_values[(state, action)]
        return q_node_val

        # util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
        # best_val will store value max_actions as specified which is the max over legal actions
        best_val = -9999999999
        if len(legalActions) != 0:
            best_action = None
            for action in legalActions:
                q_val = self.getQValue(state, action)
                if best_val <= q_val:
                    best_val, best_action = q_val, action
        else:
            best_val = 0.0
        return best_val

        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
        # best_action stores the best action to take in a state, it is initialised to None
        best_action = None
        if len(legalActions) != 0:
            best_val = -9999999999
            for action in legalActions:
                q_val = self.getQValue(state, action)
                if best_val <= q_val:
                    best_val, best_action = q_val, action
        else:
            best_action = None
        return best_action

        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        if len(legalActions) != 0:
            # Here, self.epsilon is exploration probability
            exploration_prob = self.epsilon
            if util.flipCoin(exploration_prob):
                # As instructed, we pick randomly from a list using random.choice(legalActions)
                action = random.choice(legalActions)
            else:
                # We consider the best action to take in a state in this case
                action = self.getPolicy(state)
        else:
            action = None
        return action

        # util.raiseNotDefined()
        # return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # Here, we store discount and alpha of self in variables
        discount_rate, learning_rate = self.discount, self.alpha
        # We resolve the q-value of the current state and the q-value of the next state
        curr_qval, next_qval = self.getQValue(state, action), self.computeValueFromQValues(nextState)

        x = reward + ((discount_rate * next_qval) - curr_qval)
        # This is the calculated q-value
        cal_val = curr_qval + (learning_rate * x)

        self.q_values[(state, action)] = cal_val

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        qval = 0.0
        # Getting the feature vectors for each state action pair
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            # Calculating sum of weight * featureVector where * is dotProduct operator
            w = self.weights[feature]
            featureVector = features[feature]
            qval = qval + (w * featureVector)
        return qval

        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # Getting the feature vectors for each state-action pair
        features = self.featExtractor.getFeatures(state, action)
        # Here, we store discount and alpha of self in variables
        discount_rate, learning_rate = self.discount, self.alpha
        # We resolve the q-value of the current state and the q-value of the next state
        curr_qval, next_qval = self.getQValue(state, action), self.computeValueFromQValues(nextState)

        x = reward + ((discount_rate * next_qval) - curr_qval)
        for feature in features:
            cal_val = features[feature] * (learning_rate * x)
            # Updating weights based on transition
            self.weights[feature] += cal_val

        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
