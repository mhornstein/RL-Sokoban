
# Sokoban Solver using Deep Q Learning (DQN)

This project presents the use of Reinforcement Learning to solve the Sokoban game, a complex puzzle involving the movement of boxes to target locations, and uses Deep Q Learning (DQN) for this purpose.

## Project Structure

The project is divided into three main parts:

### Experiments

In this section, one can explore how different hyperparameter values affect the performance of the DQN model. The script for running experiments is `experimenter.py`. To execute the experiments, use the following command:

```bash
python experimenter.py
```

To customize the configuration for the experiments, adjust the settings in the `experiment_config.py` file. It is well-documented to assist in this task. Note that for experimenting with various hyperparameter values, simply update the `tested_parameters` dictionary as needed.

#### Experiment Results

During the experiments, you can expect the following:

* Progress updates will be printed to the console.
* For each hyperparameter, a dictionary of the form "result_<hyperparameter name>" will be generated. For example, for the learning rate, you will find the "results_learning_rate" dictionary.
* Inside each hyperparameter's dictionary, you will find the following:
  1. `results_plot.png`: An illustration comparing the results of different values of the hyperparameter.
  2. `test_result_<hyperparameter name>.csv` and `train_result_<hyperparameter name>.csv`: CSV files containing the raw data used to generate the `results_plot.png`.
  3. `train_log`: A directory containing training results for each hyperparameter. For each tested value of the hyperparameter, graphs showing loss, reward, and steps convergence are plotted. The learned network is also saved as a .pth file.

You can access the results used in this project under the "results" directory committed in this repository.

### Results Presentation

This section presents videos and graphs demonstrating the performance of the DQN model with the best hyperparameter configuration for three different Sokoban puzzle scenarios. These materials can be found in the `Final_Project_sokoban.ipynb` notebook, along with the accompanying videos.

### Report

For a comprehensive summary of the project's work, please refer to the `report.pdf` file.

## Tested Environment

All code within this repository has been tested in the following environment:

- Operating System: Windows 10
- Python Version: 3.10.8
