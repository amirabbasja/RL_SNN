# Description

Firstly, be sure to fill the .env file with your huggingface access tokens (read and write). Furthermore, add a session name to your app and also enter your huggingface repoId for this project. You cna see the variable names in *.env_EXAMPLE* file.

This repo is for learning reinforcement learning using Deep Q-Network (DQN) algorithm. Each example file is a different approach for training the dqn. After example 1, each run data is saved in **Data** directory in its unique directory. Each file/algorithm/parameter-scheme may be ran multiple times, this is why various runs of the same algorithm are saved in the same directory with incrementing the counter on the end of its name. Enter the parameter configuration in the conf.json file and enter the file's name in the onefile_run.py file to run it. In this file you can specify the time each single run can take and also, how long the entire training can take.

Its noteworthy that as the training progresses, the history of training is uploaded to huggignface as well. YOu can use *data_manager.py* to do necessary operations such as uploading a directory's content to huggingface or deleting the content of a directory on huggingface.

1. *ANN_EXAMPLE_1*: Teaching a simple ANN with DQN scheme using a single agent

2. *ANN_EXAMPLE_2*: Teaching a simple ANN with DQN scheme using multiple agents (Distributed learning)

--

## Files

1. conf.json: Configuration file for training. Not used for Example_1

2. onefile_run.py: Runs a python file with desired parameters. Not used for Example_1

3. utils.py: Utility functions for training the DQN adn saving/loading it

4. data_manager.py: For uploading and downloading data from huggingface

5. models.py: Contains the DQN models

6. visualize_runs: Having a directory of runs, plots their training histories all in one chart and saves it in the *./Data* directory.

7. run_aggregator: Having a multitude of huggingface apiKeys and repoIDs, aggregates all of them to a single machine.
