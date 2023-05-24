import os
import numpy as np
from DQN import dqn
from EnvWrapper import EnvWrapper
from experiment_config import *
from reports_util import log_training_process, create_report


def escape(value):
    '''
    Escapes the given string so it can be written to a csv
    '''
    es_value = str(value)
    es_value = es_value.replace(',','')
    return es_value

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

def evaluate_policy(env, policy, fix_board, num_episodes, steps_cutoff):
    '''
    Tests the given policy on the given env episode_count times.
    :return: statistics of the conducted test: successful_finish_count, unsuccessful_finish_count, successful_finish_steps_avg, total_steps_avg
    '''
    successful_finish_count = 0
    successful_finish_steps = []
    unsuccessful_finish_count = 0
    total_steps = []

    for ep in range(1, num_episodes + 1):
        # print(f'start ep: {ep}. ', end='')
        if not fix_board:
            env.change_board()
        state = env.reset()
        done = False
        for t in range(1, steps_cutoff + 1):
            env.render()
            action = policy(state)
            state, reward, done, info = env.step(action)
            if done:
                # print(f'episode finished successfully after {t} timesteps')
                successful_finish_count += 1
                successful_finish_steps.append(t)
                break
        if not done:
            # print(f'episode finished due to timeout')
            unsuccessful_finish_count += 1

        total_steps.append(t)

    successful_finish_steps_avg = 0 if len(successful_finish_steps) == 0 else np.mean(successful_finish_steps)
    total_steps_avg = np.mean(total_steps)

    return successful_finish_count, unsuccessful_finish_count, successful_finish_steps_avg, total_steps_avg

def run_experiment(env_params, algorithm_params, tested_parameter, tested_values, num_of_experiments_per_value):
    algorithm_params_cpy = algorithm_params.copy()
    env_params_cpy = env_params.copy()

    result_path = f'./results_{tested_parameter}'
    train_result_file, test_result_file = init_results_files(tested_parameter, result_path)

    train_log_path = f'{result_path}/train_log'  # Create training log path
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)

    # Running test for parameter_value
    for parameter_value in tested_values:
        esc_parameter_value = escape(parameter_value)

        if tested_parameter in env_params_cpy:
            env_params_cpy[tested_parameter] = parameter_value
        else:
            algorithm_params_cpy[tested_parameter] = parameter_value

        env = EnvWrapper(**env_params_cpy)
        algorithm_params_cpy['env'] = env

        print(f'running experiment: Algo params: {algorithm_params_cpy}, Env params: {env_params_cpy}')
        parameter_train_log_path = f'{train_log_path}/{tested_parameter}_{esc_parameter_value}'  # Create training log path
        if not os.path.exists(parameter_train_log_path):
            os.makedirs(parameter_train_log_path)

        for experiment in range(1, num_of_experiments_per_value + 1):
            #################
            # Step 1: Train #
            #################
            print("Start training")
            policy, done_count, episodes_steps, episodes_rewards = dqn(**algorithm_params_cpy)

            # First - log training process
            experiment_log_path = f'{parameter_train_log_path}/{experiment}'
            log_training_process(experiment_log_path, episodes_steps, episodes_rewards)

            # Then - log training results
            f = open(train_result_file, 'a')
            total_steps_avg = np.mean(episodes_steps)
            rewards_avg = np.mean(episodes_rewards)
            f.write(f'{esc_parameter_value},{experiment},{done_count},{algorithm_params_cpy["num_episodes"]},{total_steps_avg},{rewards_avg}\n')
            f.close()

            ################
            # Step 2: Test #
            ################
            print('Start testing')
            done_episodes_count, undone_episodes_count, done_episodes_avg_steps, total_steps_avg = evaluate_policy(env, policy, fix_board=algorithm_params_cpy['fixed_board'], num_episodes=test_num_episodes, steps_cutoff=test_steps_cutoff)

            f = open(test_result_file, 'a')
            f.write(f'{esc_parameter_value},{experiment},{done_episodes_count},{undone_episodes_count},{done_episodes_avg_steps},{total_steps_avg}\n')
            f.close()
        create_report(result_path, tested_parameter, train_result_file, test_result_file)

if __name__ == '__main__':
    for tested_parameter, tested_values in tested_parameters.items():
        print(f'Testing: {tested_parameter} with values: {tested_values}')
        run_experiment(env_params, algorithm_params, tested_parameter, tested_values, num_of_experiments_per_value)