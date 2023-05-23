import random
random.seed(2)

#############################
# Experiments configuration #
#############################

DEBUG = True

'''
This is the number of time a given hyper-parameter value will be evaluated (i.e. "sample size").
At the end, the mean of the resulting metrices will be calculated and presented.
'''
num_of_experiments_per_value = 2 if DEBUG else 10

'''
Training-phase parameters:
train_num_episodes - the number of episodes for the training
train_steps_cutoff - maximal steps allowed per episode
'''
train_num_episodes = 4 if DEBUG else 500
train_steps_cutoff = 3 if DEBUG else 5000

'''
Evaluation-phase parameters
test_num_episodes - the number of episodes for the evaluation
test_steps_cutoff - maximal steps allowed per episode
'''
test_num_episodes = 4 if DEBUG else 10
test_steps_cutoff = 3 if DEBUG else 500

'''
Default hyper-parameters values (the tested hyperparameter will override its value in the relevant test)
'''

batch_size = 2 if DEBUG else 32

env_params = {'compliance': 0.9, 'num_boxes': 1}
algorithm_params = {'learning_rate': 0.9, 'gamma': 0.99, 'epsilon': 1.0, 'ep_decay': 0.99,
                    'num_episodes': train_num_episodes, 'steps_cutoff': train_steps_cutoff,
                    'batch_size': batch_size, 'target_freq_update': 8, 'memory_buffer_size':1000,
                    'layers_sizes': [32], 'fixed_board': True}

#######################################
# Tested hyperparameter configuration #
#######################################
'''
remove dictionary keys to test less hyperparameters
Change the values in the entries to test different hyper-parameters values
'''
tested_parameters = {'layers_sizes': [[16], [32], [64], [16,16], [32,32], [16,32], [32, 16]],
                     'batch_size': [16, 32, 64, 128, 256],
                     'target_freq_update': [4, 8, 16, 32],
                     'memory_buffer_size': [32, 64, 128, 256],
                     'fixed_board': [True, False],
                     'compliance': [0.8, 0.9, 1],
                     'learning_rate': [0.8, 0.9, 0.99],
                     'gamma': [0.8, 0.9, 0.99],
                     'ep_decay': [0.9, 0.95, 0.99],
                     }