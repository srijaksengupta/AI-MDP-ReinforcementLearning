import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        x = 0
        while x < self.iterations:
            # Here, we create a copy of values
            v = self.values.copy()
            for state in states:
                # Checks of state is terminal state or not and if the best action in the given state is valid
                if not mdp.isTerminal(state) and self.getAction(state):
                    # To get the Q-value of action in state from the value function stored in self.values
                    q = self.getQValue(state, self.getAction(state))
                    v[state] = q
                else:
                    continue
            x += 1
            self.values = v


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        q_value = 0
        # Getting the transition states and probabilities
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        for n_state, prob in transitionStatesAndProbs:
            reward, val, discount = self.mdp.getReward(state, action, n_state), self.getValue(n_state), self.discount
            # Computing the Q-value of action in state from the value function stored in self.values
            q_value += prob * (reward + (discount * val))
        return q_value

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        possible_actions = self.mdp.getPossibleActions(state)
        # best_action will store the best action in the given state to be returned, it is initialised to None
        best_action = None
        if not self.mdp.isTerminal(state) and (len(possible_actions) != 0):
            best_val = -9999999999
            for action in possible_actions:
                q_val = self.getQValue(state, action)
                if best_val <= q_val:
                    best_val, best_action = q_val, action
        return best_action

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
