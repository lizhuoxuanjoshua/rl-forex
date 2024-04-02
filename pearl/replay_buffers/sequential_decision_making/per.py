import random
import numpy as np

class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.index = 0
        self.max_priority = 1.0

    def push(self, transition):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition

        self.priorities[self.index] = max_priority
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        prob = priorities ** self.alpha / np.sum(priorities ** self.alpha)
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)
        samples = [self.buffer[i] for i in indices]
        weights = (1 / (len(self.buffer) * prob[indices])) ** self.beta
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)