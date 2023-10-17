import os
import numpy as np
from DQN import dqn
from EnvWrapper import EnvWrapper
from experiment_config import *
from reports_util import log_training_process, create_report
import time
import torch
import random
from soko_pap import PushAndPullSokobanEnv

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
    f.write(f'{tested_parameter},solved,steps_count\n')
    f.close()

    train_result_file = f'{result_path}/train_result_{tested_parameter}.csv'
    f = open(train_result_file, 'w')
    f.write(f'{tested_parameter},done_episodes_count,total_episodes_count,total_steps_avg,rewards_avg\n')
    f.close()

    return train_result_file, test_result_file

def evaluate_policy(env, policy, steps_cutoff):
    '''
    Evaluates a given policy on the given env for a limited steps_cutoff number of steps or until the environment is solved.
    :return: A tuple containing two values:
        - steps_count: The number of steps taken during the evaluation, capped at steps_cutoff.
        - done: A boolean indicating whether the environment was solved during the evaluation.
    :rtype: tuple
    '''
    state = env.reset()
    done = False
    steps_count = 0

    while not done and steps_count < steps_cutoff:
        action = policy(state)
        state, reward, done, info = env.step(action)
        steps_count += 1

    return steps_count, done

def run_experiment(env, algorithm_params, tested_parameter, tested_values):
    algorithm_params_cpy = algorithm_params.copy()

    result_path = f'./results_{tested_parameter}'
    train_result_file, test_result_file = init_results_files(tested_parameter, result_path)

    train_log_path = f'{result_path}/train_log'  # Create training log path
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)

    # Running test for parameter_value
    for parameter_value in tested_values:
        print(f'Testing param: {tested_parameter}={parameter_value}')
        start_time = time.time()

        esc_parameter_value = escape(parameter_value)

        algorithm_params_cpy[tested_parameter] = parameter_value

        algorithm_params_cpy['env'] = env

        print(f'running experiment: Algo params: {algorithm_params_cpy}, Env: {env}')
        parameter_train_log_path = f'{train_log_path}/{tested_parameter}_{esc_parameter_value}'  # Create training log path
        if not os.path.exists(parameter_train_log_path):
            os.makedirs(parameter_train_log_path)

        #################
        # Step 1: Train #
        #################
        print("Start training")
        mid_train_policy, policy, policy_net, done_count, episodes_steps, episodes_rewards, episodes_loss = dqn(**algorithm_params_cpy)

        # First - log training process
        log_training_process(parameter_train_log_path, episodes_loss, episodes_steps, episodes_rewards)

        # Then - log training results
        f = open(train_result_file, 'a')
        total_steps_avg = np.mean(episodes_steps)
        rewards_avg = np.mean(episodes_rewards)
        f.write(f'{esc_parameter_value},{done_count},{algorithm_params_cpy["num_episodes"]},{total_steps_avg},{rewards_avg}\n')
        f.close()

        # Last - save the policy network's parameters for future usage
        torch.save(policy_net.state_dict(), f'{parameter_train_log_path}/policy_net_params.pth')

        ################
        # Step 2: Test #
        ################
        print('Start testing')
        steps_count, done = evaluate_policy(env, policy, steps_cutoff=steps_cutoff)

        f = open(test_result_file, 'a')
        f.write(f'{esc_parameter_value},{done},{steps_count}\n')
        f.close()

        create_report(result_path, tested_parameter, train_result_file, test_result_file, train_log_path)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Done. Total time to run for testing param: {execution_time} seconds.\n')

if __name__ == '__main__':
    start_time = time.time()

    if fix_board: # keeping seed as in the assigment notebook
        random.seed(2)

    sok = PushAndPullSokobanEnv(dim_room=dim_room, num_boxes=num_boxes ,max_steps=steps_cutoff)

    env = EnvWrapper(sok=sok, use_distance_reward=use_distance_reward)

    for tested_parameter, tested_values in tested_parameters.items():
        print(f'Testing: {tested_parameter}. values: {tested_values}\n')
        run_experiment(env, algorithm_params, tested_parameter, tested_values)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Total time to run: {execution_time} seconds.')