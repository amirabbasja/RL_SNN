import torch.nn as nn
import torch
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import pandas as pd
import random, imageio, time, copy

class qNetwork_classic_4layers(nn.Module):
    def __init__(self, inputSize, L1Size, L2Size, L3Size, L4Size, outputSize):
        super().__init__()
        self.layer1 = nn.Linear(inputSize, L1Size)
        self.L1Relu = nn.ReLU()
        self.layer2 = nn.Linear(L1Size, L2Size)
        self.L2Relu = nn.ReLU()
        self.layer3 = nn.Linear(L2Size, L3Size)
        self.L3Relu = nn.ReLU()
        self.layer4 = nn.Linear(L3Size, L4Size)
        self.L4Relu = nn.ReLU()
        self.output = nn.Linear(L4Size, outputSize)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.L1Relu(x)
        x = self.layer2(x)
        x = self.L2Relu(x)
        x = self.layer3(x)
        x = self.L3Relu(x)
        x = self.layer4(x)
        x = self.L4Relu(x)
        x = self.output(x)
        return x

class qNetwork_classic_2layers(nn.Module):
    def __init__(self, inputSize, L1Size, L2Size, outputSize):
        super().__init__()
        self.layer1 = nn.Linear(inputSize, L1Size)
        self.L1Relu = nn.ReLU()
        self.layer2 = nn.Linear(L1Size, L2Size)
        self.L2Relu = nn.ReLU()
        self.output = nn.Linear(L2Size, outputSize)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.L1Relu(x)
        x = self.layer2(x)
        x = self.L2Relu(x)
        x = self.output(x)
        return x

class qNetwork_SNN_4layers(nn.Module):
    def __init__(self, inputSize, L1Size, L2Size, L3Size, L4Size, outputSize, **kwargs):
        super().__init__()

        # Model super parameters
        self.beta = kwargs["beta"]
        self.tSteps = kwargs["tSteps"]

        # Defining the layers
        self.layer1 = nn.Linear(inputSize, L1Size)
        self.L1LIF = snn.Leaky(beta = self.beta)
        self.layer2 = nn.Linear(L1Size, L2Size)
        self.L2LIF = snn.Leaky(beta = self.beta)
        self.layer3 = nn.Linear(L2Size, L3Size)
        self.L3LIF = snn.Leaky(beta = self.beta)
        self.layer4 = nn.Linear(L3Size, L4Size)
        self.L4LIF = snn.Leaky(beta = self.beta)
        self.output = nn.Linear(L4Size, outputSize)
        self.outputLIF = snn.Leaky(beta = self.beta)


    def forward(self, x):

        # Set initial potentials to be zero
        potential1 = self.L1LIF.reset_mem()
        potential2 = self.L2LIF.reset_mem()
        potential3 = self.L3LIF.reset_mem()
        potential4 = self.L4LIF.reset_mem()
        potential5 = self.outputLIF.reset_mem()

        # Save the state of the output layer
        outSpikes = []
        outPotentials = []

        # Iterate through time steps
        for t in range(self.tSteps):
            # First layer
            current1 = self.layer1(x)
            spk1, potential1 = self.L1LIF(current1, potential1)

            # Second layer
            current2 = self.layer2(spk1)
            spk2, potential2 = self.L2LIF(current2, potential2)

            # Third layer
            current3 = self.layer3(spk2)
            spk3, potential3 = self.L3LIF(current3, potential3)

            # Fourth layer
            current4 = self.layer4(spk3)
            spk4, potential4 = self.L4LIF(current4, potential4)

            #Output
            current5 = self.output(spk4)
            spk5, potential5 = self.outputLIF(current5, potential5)

            # Save output
            outSpikes.append(spk5)
            outPotentials.append(potential5)

        return torch.stack(outSpikes, dim = 0).sum(dim = 0)

class qNetwork_SNN_2layers(nn.Module):
    def __init__(self, inputSize, L1Size, L2Size, outputSize, **kwargs):
        super().__init__()

        # Model super parameters
        self.beta = kwargs["beta"]
        self.tSteps = kwargs["tSteps"]

        # Defining the layers
        self.layer1 = nn.Linear(inputSize, L1Size)
        self.L1LIF = snn.Leaky(beta = self.beta)
        self.layer2 = nn.Linear(L1Size, L2Size)
        self.L2LIF = snn.Leaky(beta = self.beta)
        self.output = nn.Linear(L2Size, outputSize)
        self.outputLIF = snn.Leaky(beta = self.beta)


    def forward(self, x):

        # Set initial potentials to be zero
        potential1 = self.L1LIF.reset_mem()
        potential2 = self.L2LIF.reset_mem()
        potential3 = self.outputLIF.reset_mem()

        # Save the state of the output layer
        outSpikes = []
        outPotentials = []

        # Iterate through time steps
        for t in range(self.tSteps):
            # First layer
            current1 = self.layer1(x)
            spk1, potential1 = self.L1LIF(current1, potential1)

            # Second layer
            current2 = self.layer2(spk1)
            spk2, potential2 = self.L2LIF(current2, potential2)

            #Output
            current3 = self.output(spk2)
            spk3, potential3 = self.outputLIF(current3, potential3)

            # Save output
            outSpikes.append(spk3)
            outPotentials.append(potential3)

        return torch.stack(outSpikes, dim = 0).sum(dim = 0)

class qNetwork_SNN(nn.Module):
    """"
    Makes a pyTorch network, utilizing spiking neural networks using snnTorch
    """
    
    def __init__(self, layers, **kwargs):
        """
        Initializes the SNN with passed layer. This network is consisted of linear layers
        followed by LIF (leaky integrate and fire) units with a leak parameter (beta)
        
        Args: 
            layers (list): layers is a list of the form: [inputSize, ... Hidden Layers Sizes ... , outputSize]
            beta (float): LIF neuron's leak parameter.
            tSteps (int): Number of time steps. This indicates the amount of forward passes through the network.
        """
        # Inherit from the main class
        super().__init__()
        
        # Impose initial requirements
        assert  2 < len(layers), "Network must have at least, one hidden layer"
        assert "beta" in kwargs.keys(), "Leak parameter (beta) must be passed as a keyword argument"
        assert "tSteps" in kwargs.keys(), "Number of time steps (tSteps)  must be passed as a keyword argument"
        
        # Model parameters
        self.beta = kwargs["beta"]
        self.tSteps = kwargs["tSteps"]
        self.trackLayers = True if "trackLayers" in kwargs else False
        self.layers = layers
        
        # Define the model layers
        lstLayers = []
        
        for i in range(len(layers)-1):
            lstLayers.append(nn.Linear(layers[i], layers[i+1]))
            lstLayers.append(snn.leaky(beta = self.beta))
        
        self.layers = nn.ModuleList(lstLayers) # Has 2x the number of layers argument

    def forward(self):
        """
        Makes self.tSteps passes through the network. layer output is treated as the sum of all
        the spikes in the final LIF layer throughout the time steps.
        """
        
        # Define layer potentials
        potentials = []
        outSpikes = []
        details = {"spikes": {}, "potentials": {}}
        
        # Set initial potentials to be zero
        for i in range(len(self.layers)):
            if i % 2 == 1:
                potentials.append(self.layers[i].reset_mem())
        
        # Iterate through the time steps
        for t in range(self.tSteps):
            # Iterate through the layers
            for j in range(len(self.layers)):
                if j % 2 == 0:
                    # Simple linear layer, returning the current which will go through LIF
                    x = self.layers[j](x)
                else:
                    # LIF layer, returning the current which will go through the next layer
                    x, potentials[j//2] = self.layers[j](x, potentials[j//2])
                    
                    # Track details of each layer, if asked
                    if self.trackLayers:
                        details["potentials"][f"L{j//2}"].append(potentials[j//2])
                        details["spikes"][f"L{j//2}"].append(x)
            
            # Save the outout spikes
            outSpikes.append(x)
        
        return torch.stack(outSpikes, dim = 0).sum(dim = 0), details