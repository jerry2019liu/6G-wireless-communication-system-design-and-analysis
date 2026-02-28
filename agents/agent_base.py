# Agent Base Class

class Agent:
    def __init__(self):
        pass

    def act(self, state):
        """Given the current state, return an action."""
        pass

    def learn(self, state, action, reward, next_state):
        """Update the agent based on the action taken and the reward received."""
        pass

    def reset(self):
        """Reset the agent to its initial state or configuration."""
        pass
