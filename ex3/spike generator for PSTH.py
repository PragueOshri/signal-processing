# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:23:12 2022

@author: yalud

General description:
This function generates two spike trains for two electrically stimulated neurons
Neuron A excites neuron B

Input:
    experiment general definitions (number of Trials and trial duration)
    firing rates of each neuron and corresponding durstions of response to stimulus and to excitation
Output:
    CSV files of stimulus onsets and spike trains of each neuron
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

###definitions:

#experiment general definitions:
numOfTrials=500
duration=100000 #in ms

#Stim onsets
stimOnsets=np.round(np.random.uniform(low=100,high=(duration-10000),size=(numOfTrials,))).astype(int)

#Neurons definitions:

#Neuron A
r0Neuron1=60 #baseline firing rate
rStimNeuron1=90 #firing rate after stimulation
stimDurNeuron1=50 #in ms, duration of stimulation
stimDelayNeuron1=0 #delay time until stimulation of the neuron

#Neuron B
r0Neuron2=50
rStimNeuron2=100
stimDurNeuron2=50 
stimDelayNeuron2=10
rExited2addNeuron2=20 #firing rate to add due to excitation from neuron A
excitationDur=25 #in ms, duration of excitation

###end of definitions

#simulation of the spike trains:
trialVecs1=np.zeros([duration,numOfTrials])
for trial_id in range(numOfTrials):
    if trial_id%10==0:
        print(trial_id)
    trialPoissonVecNeuron1=np.zeros([duration,1])
    trialPoissonVecNeuron2=np.zeros([duration,1])
    stimOnset=stimOnsets[trial_id]
    idx=1
    while idx<duration:
        if idx-stimOnset-stimDelayNeuron1<=stimDurNeuron1 and (idx-stimOnset-stimDelayNeuron1>0):
            rateNeuron1=rStimNeuron1
        else:
            rateNeuron1=r0Neuron1
        
        binProb=rateNeuron1/1000;
        randNum=np.random.uniform(low=0,high=1)
        if randNum<=binProb:
            trialPoissonVecNeuron1[idx]=1
        
        if idx-stimOnset-stimDelayNeuron2<=stimDurNeuron2 and (idx-stimOnset-stimDelayNeuron2>0):
            rateNeuron2=rStimNeuron2
        else:
            rateNeuron2=r0Neuron2
        
        if idx>0 and np.isin(1,trialPoissonVecNeuron1[idx-1:idx-excitationDur]):
            rateNeuron2=rateNeuron2+rExited2addNeuron2
        
        binProb=rateNeuron2/1000;
        randNum=np.random.uniform(low=0,high=1)
        if randNum<=binProb:
            trialPoissonVecNeuron2[idx]=1
        
        idx=idx+1;
    trialVecs1[:,trial_id]=trialPoissonVecNeuron1.ravel()
    trialVecs2[:,trial_id]=trialPoissonVecNeuron2.ravel()

#to save as a CSV the data
'''
np.savetxt('stimOnsets.csv', stimOnsets,delimiter=',')
np.savetxt('psthDataNeuronA.csv', trialVecs1,delimiter=',')
np.savetxt('psthDataNeuronB.csv', trialVecs2,delimiter=',')
'''

#basic scripts to check the spike trains visually:
#plt.figure
#plt.scatter(range(100),trialVecs1[range(100),1])