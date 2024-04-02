import numpy as np
import torch
from collections import deque

class OffPolicyReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, cost=None):
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'cost': cost
        }
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        return {
            'states': torch.stack([torch.tensor(sample['state']) for sample in samples]),
            'actions': torch.stack([torch.tensor(sample['action']) for sample in samples]),
            'rewards': torch.tensor([sample['reward'] for sample in samples]),
            'next_states': torch.stack([torch.tensor(sample['next_state']) for sample in samples]),
            'dones': torch.tensor([sample['done'] for sample in samples]),
            'costs': torch.tensor([sample['cost'] for sample in samples]) if self.buffer[0]['cost'] is not None else None
        }

    def __len__(self):
        return len(self.buffer)
