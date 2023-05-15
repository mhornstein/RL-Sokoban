DEBUG = False

# Hyper parameters
num_of_experiments_per_value = 2 if DEBUG else 10

# train hyper parameters
train_num_episodes = 4 if DEBUG else 500
train_steps_cutoff = 3 if DEBUG else 10000

# test hyper parameters
test_num_episodes = 4 if DEBUG else 10
test_steps_cutoff = 3 if DEBUG else 500

batch_size = 2 if DEBUG else 32

# Experiment hyper parameters (both env-related and algorithm-related)
env_params = {'compliance': 1}
algorithm_params = {'learning_rate': 0.9, 'gamma': 0.99, 'epsilon': 1.0, 'ep_decay': 0.99,
                    'num_episodes': train_num_episodes, 'steps_cutoff': train_steps_cutoff,
                    'batch_size': batch_size, 'target_freq_update': 10, 'memory_buffer_size':1000}
