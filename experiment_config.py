#############################
# Experiments configuration #
#############################
'''
Training-phase and testing-phase parameters
'''
train_num_episodes = 100 # the number of episodes for the training
test_num_episodes = 10 # the number of episodes for the evaluation
steps_cutoff = 500 # maximal steps allowed per episode

'''
Environment hyperparameters for sokoban environment
'''
fix_board = True # Set to True to use the predefined board, and False to randomly generate a new game board
compliance = 1 # when the agent takes a certain action, this is the probability that the environment will comply
num_boxes = 1 # number of boxes in the sokoban puzzle
dim_room = (7, 7) # width and length of sokoban puzzle
use_distance_reward = True # set to True to use distance-based reward. Set to False to use the default sokoban setting

'''
Default hyper-parameters values (the tested hyperparameter will override its value in the relevant test)
'''
algorithm_params = {'learning_rate': 0.0001, 'gamma': 0.999, 'epsilon': 1, 'ep_decay': 0.999,
                    'num_episodes': train_num_episodes, 'steps_cutoff': steps_cutoff,
                    'batch_size': 32, 'target_freq_update': 10, 'memory_buffer_size': 10000,
                    'train_action_value_freq_update': 1}

#######################################
# Tested hyperparameter configuration #
#######################################
'''
remove dictionary keys to test less hyperparameters
Change the values in the entries to test different hyper-parameters values
'''
tested_parameters = {
                     'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
                     'batch_size': [16, 32, 64, 128, 256],
                     'target_freq_update': [4, 10, 16, 32],
                     'memory_buffer_size': [100, 1000, 10000, 100000],
                     'gamma': [0.9, 0.95, 0.999],
                     'ep_decay': [0.9, 0.95, 0.999],
                     'epsilon': [0.7, 0.9, 1.0],
                     'train_action_value_freq_update': [1, 4, 8, 16]
                     }