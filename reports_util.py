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

def log_training_process(log_dir, episodes_loss, episodes_steps, episodes_rewards):
    # First - save the lineplot
    plt.clf()
    save_lineplot(data=episodes_loss, path=f'{log_dir}/Convergence_Graph__Episodes_loss.png',
                  title='Convergence Graph: Episodes Loss', xlabel='Episode number', ylabel='Loss')
    save_lineplot(data=episodes_steps, path=f'{log_dir}/Convergence_Graph__Episodes_steps.png',
                  title='Convergence Graph: Episodes steps', xlabel='Episode number', ylabel='Steps')
    save_lineplot(data=episodes_rewards, path=f'{log_dir}/Convergence_Graph__Episodes_reward.png',
                  title='Convergence Graph: Episodes reward', xlabel='Episode number', ylabel='Reward')

    # Then - save the raw data
    raw_data = {
        'episode': list(range(1, len(episodes_loss) + 1)),
        'loss': episodes_loss,
        'steps': episodes_steps,
        'rewards': episodes_rewards
    }
    df = pd.DataFrame(raw_data)
    df.to_csv(f'{log_dir}/Convergence_logs.csv', index=False)

#########################
## full report generation
def create_header(subplot, header):
    subplot.set_title(header)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines.clear()

def create_table(subplot, header, df):
    table_data = [df.columns] + df.values.tolist()

    table = subplot.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    subplot.set_title(header)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines.clear()

def plot_mean_steps(df, tested_parameter, ax):
    df = df[[tested_parameter, 'total_steps_avg']]
    sns.barplot(data=df, x=tested_parameter, y='total_steps_avg', palette=palette, ax=ax)
    ax.set(xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set(ylabel='')

def plot_done_episodes_count(df, tested_parameter, ax):
    df = df[[tested_parameter, 'done_episodes_count']]
    sns.barplot(data=df, x=tested_parameter, y='done_episodes_count', palette=palette, ax=ax)
    ax.set(xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set(ylabel='')

def plot_train_metric(train_log_path, metric, ax):
    # First - load train logs
    res_df = pd.DataFrame()
    for dir_name in os.listdir(train_log_path):
        value_header = dir_name.split('_')[-1]
        dir_path = os.path.join(train_log_path, dir_name)
        log_file_path = os.path.join(dir_path, "Convergence_logs.csv")
        df = pd.read_csv(log_file_path).set_index('episode')
        res_df[value_header] = df[metric]
    # Then - plot
    ax.set_title(metric)
    sns.lineplot(data=res_df, palette=palette, ax=ax, dashes=False)

def create_report(plot_path, tested_parameter, train_result_file, test_result_file, train_log_path):
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(7, 6, height_ratios=[0.05,
                                       1,
                                       0.05, 1,
                                       0.1,
                                       0.05, 1], hspace=0.4, wspace=0.4)

    # add headers
    parameter_header_subplot = fig.add_subplot(gs[0, :])
    create_header(parameter_header_subplot, tested_parameter.replace('_', ' '))

    train_mean_header_subplot = fig.add_subplot(gs[2, :])
    create_header(train_mean_header_subplot, 'Train metrics')

    train_graph_header_subplot = fig.add_subplot(gs[5, :])
    create_header(train_graph_header_subplot, 'Train graphs')

    # add evaluation results
    df = pd.read_csv(test_result_file)
    policy_evaluation_subplot = fig.add_subplot(gs[1, :])
    create_table(policy_evaluation_subplot, 'Policy evaluation results', df)

    # add train metrics
    df = pd.read_csv(train_result_file)
    total_episodes_count = df['total_episodes_count'].iloc[0]

    ax = fig.add_subplot(gs[3, 0:3])
    plot_done_episodes_count(df, tested_parameter, ax)
    ax.set_title(f'Total episodes done out of {total_episodes_count}')

    ax = fig.add_subplot(gs[3, 3:])
    plot_mean_steps(df, tested_parameter, ax)
    ax.set_title(f'Avg episode steps per {total_episodes_count} episodes')

    # Add steps, reward and loss graphs
    plot_train_metric(train_log_path, 'loss', ax=fig.add_subplot(gs[6, 0:2]))
    plot_train_metric(train_log_path, 'steps', ax=fig.add_subplot(gs[6, 2:4]))
    plot_train_metric(train_log_path, 'rewards', ax=fig.add_subplot(gs[6, 4:]))

    plt.savefig(f'{plot_path}/results_plot.png')
    plt.close()
    plt.clf()

if __name__ == '__main__':
    tested_param = 'learning_rate'
    # tested_param = 'layers_sizes'
    #tested_param = 'compliance'
    result_path = f'results_{tested_param}'
    train_results_path = f'./{result_path}/train_result_{tested_param}.csv'
    test_results_path = f'./{result_path}/test_result_{tested_param}.csv'
    train_log_path = f'{result_path}/train_log'
    create_report(f'./results_{tested_param}', tested_param, train_results_path, test_results_path, train_log_path)