import os
import numpy as np

from DQN import dqn
from EnvWrapper import EnvWrapper
from constants import *
from experiment_config import tested_parameters, num_boxes
from soko_pap import PushAndPullSokobanEnv

def init_results_files(tested_parameter, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Test and train stats csv files
    test_result_file = f'{result_path}/test_result_{tested_parameter}.csv'
    f = open(test_result_file, 'w')
    f.write(f'{tested_parameter},experiment_number,done_episodes_count,undone_episodes_count,done_episodes_avg_steps,total_steps_avg\n')
    f.close()

    train_result_file = f'{result_path}/train_result_{tested_parameter}.csv'
    f = open(train_result_file, 'w')
    f.write(f'{tested_parameter},experiment_number,done_episodes_count,total_episodes_count,total_steps_avg,rewards_avg\n')
    f.close()

    return train_result_file, test_result_file

def run_experiment(env_params, algorithm_params, tested_parameter, tested_values, num_of_experiments_per_value):
    algorithm_params_cpy = algorithm_params.copy()
    env_params_cpy = env_params.copy()

    env_params_cpy['sok'] = PushAndPullSokobanEnv(dim_room=(7, 7), num_boxes=1, max_steps=500) # todo UPDATE

    result_path = f'./results_{tested_parameter}'
    train_result_file, test_result_file = init_results_files(tested_parameter, result_path)

    train_log_path = f'{result_path}/train_log'  # Create training log path
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)

    # Running test for parameter_value
    for parameter_value in tested_values:

        if tested_parameter in env_params_cpy:
            env_params_cpy[tested_parameter] = parameter_value
        else:
            algorithm_params_cpy[tested_parameter] = parameter_value

        env = EnvWrapper(**env_params_cpy)
        algorithm_params_cpy['env'] = env

        print(f'running experiment: Algo params: {algorithm_params_cpy}, Env params: {env_params_cpy}')
        parameter_train_log_path = f'{train_log_path}/{tested_parameter}_{parameter_value}'  # Create training log path
        if not os.path.exists(parameter_train_log_path):
            os.makedirs(parameter_train_log_path)

        for experiment in range(1, num_of_experiments_per_value + 1):
            #print(f'Evaluating parameter: {tested_parameter}={parameter_value}. Algo: {algorithm_name}. Experiment number: {experiment}.')
            #################
            # Step 1: Train #
            #################
            #print("Start training")
            policy, done_count, episodes_steps, episodes_rewards = dqn(**algorithm_params_cpy)

            # First - log training process
            experiment_log_path = f'{parameter_train_log_path}/{experiment}'
            # TODO
            # log_training_process(experiment_log_path, start_states_count, states_visits_mean, episodes_steps, episodes_rewards)

            # Then - log training results
            '''
            f = open(train_result_file, 'a')
            total_steps_avg = np.mean(episodes_steps)
            rewards_avg = np.mean(episodes_rewards)
            f.write(f'{parameter_value},{experiment},{done_count},{algorithm_params_cpy["num_episodes"]},{total_steps_avg},{rewards_avg}\n')
            f.close()
            '''

            ################
            # Step 2: Test #
            ################
            '''
            print('Start testing')
            done_episodes_count, undone_episodes_count, done_episodes_avg_steps, total_steps_avg = evaluate_policy(env, policy, num_episodes=test_num_episodes, steps_cutoff=test_steps_cutoff)

            f = open(test_result_file, 'a')
            f.write(f'{parameter_value},{experiment},{done_episodes_count},{undone_episodes_count},{done_episodes_avg_steps},{total_steps_avg}\n')
            f.close()
            '''
    # create_report(result_path, tested_parameter, train_result_file, test_result_file)

if __name__ == '__main__':
    for tested_parameter, tested_values in tested_parameters.items():
        run_experiment(env_params, algorithm_params, tested_parameter, tested_values, num_of_experiments_per_value)