# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# Importing necessary modules from the PEARL framework, a reinforcement learning library.
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import UCBExploration
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
# Importing additional policy learners and replay buffers for sequential decision making.
from pearl.replay_buffers.sequential_decision_making.fifo_on_policy_replay_buffer import FIFOOnPolicyReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from Env_Forex import ForexTradingEnv

# Importing a custom environment for Forex Trading.


# Setting display options for numpy and pandas to ensure all data is visible in outputs.
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)  # Adjust display width for complete row visibility
pd.set_option('display.max_colwidth', None)  # Adjust display width for individual cell content


indicator_column_names=[ '二分高_Close_1', '二分高_Close_2', '二分高_Close_3', '二分高_Close_4', '二分高_Close_5', '二分高_Close_6', '二分高_Close_7', '二分高_Close_8', '二分高_Close_9', '二分高_Close_10', '二分高_Close_11', '二分高_Close_12', '二分高_Close_13', '二分高_Close_14', '二分高_Close_15', '二分高_Close_16', '二分高_Close_17', '二分高_Close_18', '二分高_Close_19', '二分高_Close_20', '二分低_Close_1', '二分低_Close_2', '二分低_Close_3', '二分低_Close_4', '二分低_Close_5', '二分低_Close_6', '二分低_Close_7', '二分低_Close_8', '二分低_Close_9', '二分低_Close_10', '二分低_Close_11', '二分低_Close_12', '二分低_Close_13', '二分低_Close_14', '二分低_Close_15', '二分低_Close_16', '二分低_Close_17', '二分低_Close_18', '二分低_Close_19', '二分低_Close_20', '二分高simi_Close_1', '二分高simi_Close_2', '二分高simi_Close_3', '二分高simi_Close_4', '二分高simi_Close_5', '二分高simi_Close_6', '二分高simi_Close_7', '二分高simi_Close_8', '二分高simi_Close_9', '二分高simi_Close_10', '二分高simi_Close_11', '二分高simi_Close_12', '二分高simi_Close_13', '二分高simi_Close_14', '二分高simi_Close_15', '二分高simi_Close_16', '二分高simi_Close_17', '二分高simi_Close_18', '二分高simi_Close_19', '二分高simi_Close_20', '二分低simi_Close_1', '二分低simi_Close_2', '二分低simi_Close_3', '二分低simi_Close_4', '二分低simi_Close_5', '二分低simi_Close_6', '二分低simi_Close_7', '二分低simi_Close_8', '二分低simi_Close_9', '二分低simi_Close_10', '二分低simi_Close_11', '二分低simi_Close_12', '二分低simi_Close_13', '二分低simi_Close_14', '二分低simi_Close_15', '二分低simi_Close_16', '二分低simi_Close_17', '二分低simi_Close_18', '二分低simi_Close_19', '二分低simi_Close_20']

# Define the environment with a custom Forex trading environment, initializing it with a specific CSV file for data.
env = ForexTradingEnv(csv_file_path="Data_module/data.csv",
                      indicator_column_names=indicator_column_names)

# Initialize the PEARL agent with a policy learner, a history summarization module, and a replay buffer.
agent = PearlAgent(
    policy_learner=ContinuousSoftActorCritic(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        training_rounds=400000,
        exploration_module=UCBExploration(alpha=0.5)
    ),
    history_summarization_module=LSTMHistorySummarizationModule(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=env.observation_space.shape[0],
        history_length=120*24, # steps per day
    ),
    replay_buffer=FIFOOnPolicyReplayBuffer(250), # Initialize a First-In-First-Out On-Policy Replay Buffer with capacity 250
)

# Setting the recording period for tracking the agent's performance.
record_period=1
# Perform online learning with the agent in the environment, collecting performance data.
info = online_learning(
    agent=agent,
    env=env,
    print_every_x_steps=100,
    number_of_episodes=80000,
    record_period=record_period,
    learn_after_episode=True,
    reload_model=False,
    model_path=r"pearl\外汇交易手动保存.pt",
)

# Output the performance data for review.
print(info)

# Save the recorded data for later analysis.
torch.save(info["return"], "保存记录.pt")

# Plot the returns for each episode, providing a visual representation of the agent's learning progress over time.
plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="return of every episode")
plt.legend()
plt.show()



