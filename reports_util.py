import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import csv

palette = 'Set2'

def save_heatmap(data, path, fmt='.2g', annot_kws=None):
    fig = sns.heatmap(data, annot=True, fmt=fmt, annot_kws=annot_kws)
    plt.savefig(path)
    plt.clf()
    plt.close()

def save_lineplot(data, path, title, xlabel, ylabel):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.clf()
    plt.close()

def log_training_process(log_dir, episodes_steps, episodes_rewards):
    plt.clf()
    save_lineplot(data=episodes_steps, path=f'{log_dir}/Convergence_Graph__Episodes_steps.png',
                  title='Convergence Graph: Episodes steps', xlabel='Episode number', ylabel='Steps')
    save_lineplot(data=episodes_rewards, path=f'{log_dir}/Convergence_Graph__Episodes_reward.png',
                  title='Convergence Graph: Episodes reward', xlabel='Episode number', ylabel='Reward')

def log_train_loss(log_dir, train_loss):
    save_lineplot(data=train_loss, path=f'{log_dir}/Convergence_Graph__network_loss.png',
                  title='Convergence Graph: Loss', xlabel='Episode number', ylabel='Loss')

#########################
## full report generation
def create_header(subplot, header):
    subplot.set_title(header)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines.clear()

def plot_mean_steps(df, tested_parameter, ax):
    df = df[[tested_parameter, 'total_steps_avg']]
    sns.barplot(data=df, x=tested_parameter, y='total_steps_avg', palette=palette, ax=ax)
    ax.set(xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set(ylabel=ax.get_ylabel().replace('_', ' '))

def plot_done_episodes_count(df, tested_parameter, total_episodes_count, ax):
    df = df[[tested_parameter, 'done_episodes_count']]
    sns.barplot(data=df, x=tested_parameter, y='done_episodes_count', palette=palette, ax=ax)
    ax.set(xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set(ylabel=f"avg {ax.get_ylabel().replace('_', ' ')} [in {total_episodes_count} episodes]")

def create_report(plot_path, tested_parameter, train_result_file, test_result_file):
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(6, 2, height_ratios=[0.05, 0.05, 1, 0.1, 0.05, 1], hspace=0.4, wspace=0.4)

    # add headers
    parameter_header_subplot = fig.add_subplot(gs[0, :])
    create_header(parameter_header_subplot, tested_parameter.replace('_', ' '))

    train_header_subplot = fig.add_subplot(gs[1, :])
    create_header(train_header_subplot, 'train results')

    test_header_subplot = fig.add_subplot(gs[4, :])
    create_header(test_header_subplot, 'test results')

    # add train graphs
    df = pd.read_csv(train_result_file)

    ax = fig.add_subplot(gs[2, 0])
    plot_mean_steps(df, tested_parameter, ax)

    ax = fig.add_subplot(gs[2, 1])
    total_episodes_count = df['total_episodes_count'].iloc[0]
    plot_done_episodes_count(df, tested_parameter, total_episodes_count, ax)

    # add test graphs
    df = pd.read_csv(test_result_file)

    ax = fig.add_subplot(gs[5, 0])
    plot_mean_steps(df, tested_parameter, ax)

    ax = fig.add_subplot(gs[5, 1])
    total_episodes_count = df.iloc[0].done_episodes_count + df.iloc[0].undone_episodes_count
    plot_done_episodes_count(df, tested_parameter, total_episodes_count, ax)

    plt.savefig(f'{plot_path}/results_plot.png')
    plt.close()
    plt.clf()

if __name__ == '__main__':
    tested_param = 'fixed_board'
    # tested_param = 'layers_sizes'
    #tested_param = 'compliance'
    train_results_path = f'./results_{tested_param}/train_result_{tested_param}.csv'
    test_results_path = f'./results_{tested_param}/test_result_{tested_param}.csv'
    create_report(f'./results_{tested_param}', tested_param, train_results_path, test_results_path)