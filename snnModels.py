import torch.nn as nn
import torch
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import pandas as pd
import random, imageio, time, copy


class qNetwork_SNN_dynamic(nn.Module):
    def __init__(self, layers, **kwargs):
        """
        Self.layers = [inputSize, ... Hidden Layers Sizes ... , outputSize]
        """
        super().__init__()

        # Model super parameters
        self.beta = kwargs["beta"]
        self.tSteps = kwargs["tSteps"]
        self.trackDetails = True if "trackDetails" in kwargs else False

        # Defining the layers
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(snn.Leaky(beta = self.beta))
        


    def forward(self, x):

        # Set initial potentials to be zero
        potentials = []
        for i in range(len(self.layers)):
            if i % 2 == 1:
                potentials.append(self.layers[i].reset_mem())

        # Save the state of the output layer
        outSpikes = []
        detailsDict = {"spikes": {}, "potentials": {}}
        if self.trackDetails:
            for i in range(len(self.layers)/2):
                detailsDict["potentials"][f"L{i}"] = []
                detailsDict["spikes"][f"L{i}"] = []

        # Iterate through time steps
        for t in range(self.tSteps):
            # Iterate through layers
            for i in range(len(self.layers)):
                if i % 2 == 0:
                    x = self.layers[i](x)
                else:
                    x, potentials[i] = self.layers[i](x, potentials[i])
                    
                    # Save the output state of each layer, if asked 
                    if self.trackDetails:
                        detailsDict["potentials"][f"L{i//2}"].append(potentials[i])
                        detailsDict["spikes"][f"L{i//2}"].append(x)
                
                outSpikes.append(x)

        return torch.stack(outSpikes, dim = 0).sum(dim = 0), detailsDict

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