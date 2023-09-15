import random
random.seed(2)

#############################
# Experiments configuration #
#############################
'''
Training-phase parameters:
train_num_episodes - the number of episodes for the training
train_steps_cutoff - maximal steps allowed per episode
'''
train_num_episodes = 100
train_steps_cutoff = 500

'''
Evaluation-phase parameters
test_num_episodes - the number of episodes for the evaluation
test_steps_cutoff - maximal steps allowed per episode
'''
test_num_episodes = 10
test_steps_cutoff = 500

'''
Default hyper-parameters values (the tested hyperparameter will override its value in the relevant test)
'''

env_params = {'compliance': 1, 'num_boxes': 1}
algorithm_params = {'learning_rate': 0.001, 'gamma': 0.9, 'epsilon': 0.99, 'ep_decay': 0.9,
                    'num_episodes': train_num_episodes, 'steps_cutoff': train_steps_cutoff,
                    'batch_size': 32, 'target_freq_update': 10, 'memory_buffer_size': 10000,
                    'layers_sizes': [16, 32], 'fixed_board': True, 'train_action_value_freq_update': 1}

'''
Add to the list numeric hyper-parameter that you want to force to plot as catergorical (using bar plot)
instead of numeric (using lineplot) in the final report
'''
force_categoric_plot = ['batch_size', 'target_freq_update', 'memory_buffer_size']

#######################################
# Tested hyperparameter configuration #
#######################################
'''
remove dictionary keys to test less hyperparameters
Change the values in the entries to test different hyper-parameters values
'''
tested_parameters = {'layers_sizes': [[16], [32], [64], [16,16], [32,32], [16,32], [32, 16]],
                     'learning_rate': [0.001, 0.005, 0.01, 0.05],
                     'batch_size': [16, 32, 64, 128, 256],
                     'target_freq_update': [4, 10, 8, 16, 32],
                     'memory_buffer_size': [100, 1000, 10000, 100000],
                     'compliance': [0.8, 0.9, 1],
                     'learning_rate': [0.01, 0.05, 0.001, 0.005, 0.0001],
                     'gamma': [0.8, 0.9, 0.99],
                     'ep_decay': [0.5, 0.75, 0.9, 0.99],
                     'train_action_value_freq_update': [1, 4, 8, 16,32]
                     }

tested_parameters = { 'gamma': [0.5, 0.75, 0.9, 0.99] }