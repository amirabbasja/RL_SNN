from models import qNetwork_ANN
from collections import deque, namedtuple
import os
import argparse
from utils import *
from IPython.display import clear_output

import sys
from tqdm import tqdm
import pandas as pd
import random, imageio, time, copy
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

# Parse model arguments
parser = modelParamParser()
args, unknown = parser.parse_known_args()

# Define the super parameters
projectName = args.name

# Save/Get weights from persistent storage. Pass empty string for not saving. 
# Pass drive for using google derive (If code is running in colab). If local, 
# pass the location of your desire
savePath = os.path.join(os.path.dirname(__file__), "Data")
continueLastRun = args.continue_run
backUpData = {}

# Make the save directory if it does not exist
os.makedirs(savePath, exist_ok = True)

if __name__ == "__main__":
    runStartTime = time.time() # The time the training begun
    maxRunTime = runStartTime + args.max_run_time
    
    # Making the environments
    NUM_ENVS = args.agents
    env = gym.make("LunarLander-v3") # Use render_mode = "human" to render each episode
    envs = gym.make_vec(
        "LunarLander-v3", 
        num_envs = NUM_ENVS, # Number of environments to create
        vectorization_mode = "async",
        wrappers = (gym.wrappers.TimeAwareObservation,),
    )
    states, info = env.reset() # Get a sample state of the environment
    stateSize = env.observation_space.shape # Number of variables to define current step
    nActions = env.action_space.n # Number of actions
    actionSpace = np.arange(nActions).tolist()

    # Set pytorch parameters: The device (CPU or GPU) and data types
    __device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    __dtype = torch.float

    # Model parameters
    hiddenNodes = args.hidden_layers
    learningRate = args.learning_rate
    eDecay = args.decay
    miniBatchSize = args.batch # The length of mini-batch that is used for training
    gamma = args.gamma # The discount factor
    extraInfo = args.extra_info
    continueLastRun = args.continue_run

    # handle the save location
    modelDetails = f"{'_'.join([str(l) for l in hiddenNodes])}_{learningRate}_{eDecay}_{miniBatchSize}_{gamma}_{NUM_ENVS}_{extraInfo}"
    savePath = os.path.join(savePath, f"{projectName}_{modelDetails}")
    os.makedirs(savePath, exist_ok=True)

    # Get how many times the model has been trained and add it to the file name
    runNumber =  len([f for f in os.listdir(savePath) if f"{modelDetails}" in f]) if savePath != None else ""
    runNumber = runNumber if not continueLastRun else runNumber -1
    modelDetails += f"_{runNumber}"
    
    saveFileName = f"{projectName}_{modelDetails}.pth"
    
    # Make the model objects
    qNetwork_model = qNetwork_ANN([stateSize[0], *hiddenNodes, nActions]).to(__device, dtype = __dtype)
    targetQNetwork_model = qNetwork_ANN([stateSize[0], *hiddenNodes, nActions]).to(__device, dtype = __dtype)

    # Two models should have identical weights initially
    targetQNetwork_model.load_state_dict(qNetwork_model.state_dict())

    # TODO: Add gradient clipping to the optimizer for avoiding exploding gradients
    # Suitable optimizer for gradient descent
    optimizer_main = torch.optim.Adam(qNetwork_model.parameters(), lr=learningRate)
    optimizer_target = torch.optim.Adam(targetQNetwork_model.parameters(), lr=learningRate)

    # Starting episode and ebsilon
    startEpisode = 0
    startEbsilon = None
    lstHistory = None

    # Making the memory buffer object
    memorySize = 100_000 # The length of the entire memory
    mem = ReplayMemory(memorySize, __dtype, __device)

    if continueLastRun and os.path.isfile(os.path.join(savePath, saveFileName)):
        # Load necessary parameters to resume the training from most recent run 
        saveLen = 1
        load_params = {
            "qNetwork_model": qNetwork_model,
            "optimizer_main": optimizer_main,
            "targetQNetwork_model": targetQNetwork_model,
            "trainingParams": [startEpisode, startEbsilon, lstHistory, eDecay, mem]
        }
        # NUM_ENVS is a constant and is defined when running the script for the first time, So we disregard re-loading it
        qNetwork_model, optimizer_main, targetQNetwork_model, startEpisode, startEbsilon, lstHistory, eDecay, _, mem = loadNetwork(os.path.join(savePath, saveFileName), **load_params)
        print("Continuing from episode:", startEpisode)

    print(f"Device is: {__device}")

    # Start the timer
    tstart = time.time()

    # The experience of the agent is saved as a named tuple containing various variables
    agentExp = namedtuple("exp", ["state", "action", "reward", "nextState", "done"])

    # Parameters
    nEpisodes = 6000 # Number of learning episodes
    maxNumTimeSteps = 1000 # The number of time step in each episode
    ebsilon = 1 if startEbsilon == None else startEbsilon # The starting  value of ebsilon
    ebsilonEnd   = .1 # The finishing value of ebsilon
    eDecay = eDecay # The rate at which ebsilon decays
    numUpdateTS = 4 # Frequency of time steps to update the NNs
    numP_Average = 100 # The number of previous episodes for calculating the average episode reward

    # Variables for saving the required data for later analysis
    episodePointHist = [] # For saving each episode's point for later demonstration
    episodeHistDf = None
    lstHistory = [] if lstHistory == None else lstHistory
    initialCond = None # Initial condition (state) of the episode
    epPointAvg = -999999 if len(lstHistory) == 0 else pd.DataFrame(lstHistory).iloc[-numP_Average:]["points"].mean()
    latestCheckpoint = 0
    _lastPrintTime = 0


    initialSeed = random.randint(1,1_000_000_000) # The random seed that determines the episode's I.C.
    states, info = envs.reset(seed = initialSeed)
    points = np.zeros((envs.num_envs, 1))
    initialCond = states
    tempTime = time.time()
    t = 0
    episode = 0

    while True:
        # The last element of each state is the time step, so we slice the tensor
        qValueForActions = qNetwork_model(torch.tensor(states[:,:-1], device = __device, dtype = __dtype))

        # use ebsilon-Greedy algorithm to take the new step
        action = getAction(qValueForActions, ebsilon, actionSpace, __device).cpu().numpy()

        # Take a step
        observation, reward, terminated, truncated, _ = envs.step(action)

        batchExperiences = [agentExp(s, a, r, o, d) for s, a, r, o, d in zip(states[:,:-1], action, reward, observation[:,:-1], terminated | truncated) ]

        # Store the experience of the current step in an experience deque.
        mem.addMultiple(batchExperiences)

        # Check to see if we have to update the networks in the current step
        update = updateNetworks(t, mem, miniBatchSize, numUpdateTS)

        if update:
            # Update the NNs
            experience = mem.sample(miniBatchSize)

            # Update the Q-Network and the target Q-Network
            # Bear in mind that we do not update the target Q-network with direct gradient descent.
            # so there is no optimizer needed for it
            fitQNetworks(experience, gamma, [qNetwork_model, optimizer_main], [targetQNetwork_model, None])

        # Save the necessary data
        points += reward.reshape(-1, 1)
        states = observation.copy()

        # Print the training status. Print only once each second to avoid jitters.
        if 1 < (time.time() - _lastPrintTime):
            print('\033[2J\033[H', end='', flush=True)  # Clear screen and move 
            _lastPrintTime = time.time()
            print(f"ElapsedTime: {int(time.time() - tstart): <5}s | Episode: {episode: <5} | Time step: {t: <5} | The average of the {numP_Average: <5} episodes is: {int(epPointAvg): <5}")
            print(f"Latest checkpoint: {latestCheckpoint} | Speed {t/(time.time()-tempTime+1e-9):.1f} tps | ebsilon: {ebsilon:.3f}")
            print(f"Run number: {runNumber} | Remaining training time: {int(maxRunTime - time.time())}")

        # Handle episode ending
        if (terminated | truncated).any():
            mask = terminated | truncated
            finalPoint = points[mask]
            
            for k in range(finalPoint.shape[0]):
                # Decay ebsilon
                ebsilon = decayEbsilon(ebsilon, eDecay, ebsilonEnd)
                episode += 1
                
                # Save the episode history in dataframe
                if (episode+1) % 3 == 0:
                    # only save every 10 episodes
                    lstHistory.append({
                        "episode": episode,
                        "seed": initialSeed,
                        "points": finalPoint[k]
                    })
                
                # Save model weights and parameters periodically (For later use)
                if (episode + 1) % (20) == 0:
                    _exp = mem.exportExperience()
                    backUpData = {
                        "episode": episode,
                        'qNetwork_state_dict': qNetwork_model.state_dict(),
                        'qNetwork_optimizer_state_dict': optimizer_main.state_dict(),
                        'targetQNetwork_state_dict': targetQNetwork_model.state_dict(),
                        'targetQNetwork_optimizer_state_dict': optimizer_target.state_dict(),
                        'hyperparameters': {"ebsilon": ebsilon, "eDecay": eDecay, "NUM_ENVS": NUM_ENVS},
                        "elapsedTime": int(time.time() - tstart),
                        "train_history": lstHistory,
                        "experiences": {
                            "state": _exp["state"],
                            "action": _exp["action"],
                            "reward": _exp["reward"],
                            "nextState": _exp["nextState"],
                            "done": _exp["done"]
                        }
                    }
                    
                    saveModel(backUpData, os.path.join(savePath, saveFileName))

                    # Save the episode number
                    latestCheckpoint = episode
            
            # Add the points to the history
            episodePointHist.extend(finalPoint.tolist())
            
            # Getting the average of {numP_Average} episodes
            epPointAvg = np.mean(episodePointHist[-numP_Average:])

            # Reset the points of terminated episodes
            points[mask] = 0

        # Stop the learning process if suitable average point is reached
        if 200 < epPointAvg or maxRunTime < time.time():
            Tend = time.time()
            print(f"\nThe learning ended. Elapsed time for learning: {Tend-tstart:.2f}s. \nAVG of latest 100 episodes: {epPointAvg}")
            
            _exp = mem.exportExperience()
            backUpData = {
                "episode": episode,
                'qNetwork_state_dict': qNetwork_model.state_dict(),
                'qNetwork_optimizer_state_dict': optimizer_main.state_dict(),
                'targetQNetwork_state_dict': targetQNetwork_model.state_dict(),
                'targetQNetwork_optimizer_state_dict': optimizer_target.state_dict(),
                'hyperparameters': {"ebsilon": ebsilon, "eDecay":eDecay, "NUM_ENVS": NUM_ENVS},
                "train_history": lstHistory,
                "elapsedTime": int(time.time() - tstart),
                "experiences": {
                    "state": _exp["state"],
                    "action": _exp["action"],
                    "reward": _exp["reward"],
                    "nextState": _exp["nextState"],
                    "done": _exp["done"]
                }
            }
            
            saveModel(backUpData, os.path.join(savePath, saveFileName))

            # Save the episode number
            latestCheckpoint = episode
            
            break
        t += 1

    # Reset the index
    episodeHistDf = pd.DataFrame(lstHistory)
    episodeHistDf.reset_index(drop=True, inplace=True)