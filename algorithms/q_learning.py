import numpy as np

class q_learning:

    def __init__(self, q_table, alpha=0.2, gamma=0.99, epsilon=0.002):
        """
        Parameters
        ----------
        alpha : double
            learning rate
        gamma : double
            discount factor
        epsilon : double
        """
        self.q_table=q_table
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        self.action_num=self.q_table.shape[-1]

    def get_state(self, _observation):
        # this function should return a tuple which indicates state's indices of q-table
        return (_observation)

    def update_q_table(self, _action,  _observation, _next_observation, _reward):
        # the best action value which will be taken at the next observation
        next_state = self.get_state(_next_observation)
        next_max_q_value = max(self.q_table[next_state])

        # the action value at the current observation
        state = self.get_state(_observation)
        q_value = self.q_table[state][_action]

        # update the action value function
        self.q_table[state][_action] = q_value + self.alpha * (_reward + self.gamma * next_max_q_value - q_value)

    def get_action(self, _observation):
        if np.random.uniform(0, 1) > self.epsilon:
            state = self.get_state(_observation)
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(self.action_num)
        return  action
