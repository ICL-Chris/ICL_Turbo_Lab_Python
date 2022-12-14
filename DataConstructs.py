#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for containing data created in the lab and read from lab data files


Created on Sat Nov  6 19:21:55 2021

@author: cnoon
"""

import numpy as np
from matplotlib import pyplot
pyplot.ion()

class DataChannel:
    """Contains the data, and associated parameters for a channel"""
    
    type = "dataChannel"
    
    def __init__(self, name :str, data :list, units: str = "", offset: float = 0):
        self.data = np.array(data, dtype=(np.float32))
        self.name = name
        self.units = units

class ScaledDataChannel(DataChannel):
    """Specific class for reading test data from logged datafile"""
    
    type = "scaled data channel"
    
    def __init__(self, name: str, data, units: str = "", scalingFactors = [0,0,0,0,0],
                 offset: float = 0, scalingType = ""):
        self.data = np.array(data, dtype=(np.float32))
        self.name = name
        self.units = units
        self.offset = offset
        self.scalingFactors = scalingFactors
        self.scalingType = scalingType

class DataFile:
    """contains data channels and parameters from logged etst data"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.dataChannels = []
        self.channelCount = 0
        self.dataPoints = 0
        self.frequency = 0
        
    def getChannelNames(self):
        """
        return names of channels contained within the dataFile
        """
        
        channelNames = []
        for channel in self.dataChannels:
            channelNames.append(channel.name)
        
        return channelNames
    
    def showChannels(self):
        """
        return nindices and channel names for all channels in the file
        """
        i = 0
        for channel in self.dataChannels:
            print(i,'\t',channel.name)
            i += 1
        
    
    def getChannelData(self, channelSet = []):
        """
        Return specific channel data
        
        returns all data if no channels are specified
        """
        # check type and length of channelSet, set to all if empty list
        if type(channelSet) == int:
            return self.dataChannels[channelSet].data
        
        else:
            if len(channelSet) == 0:
                channelSet = list(range(self.channelCount))
                    
            data = np.array([self.dataChannels[channelSet[0]].data])
            
            for i in range(1,len(channelSet)):
                data = np.concatenate((data,
                                       [self.dataChannels[channelSet[i]].data]))
            
            return data
    
    def plotData(self, plotChannels):
        """Plot directly channels of data
        
        """
        
        if type(plotChannels) == int:
            pyplot.plot(self.getChannelData(plotChannels))
        else:
            pyplot.plot(self.getChannelData(plotChannels).T)

    def plotTime(self, channelSet):
        """Plots the selected channels against the time channel
        
        """
        pyplot.plot(self.dataChannels[0],self.getChannelData(channelSet))
    
    def showFilename(self):
        """ Print filename to screen
        
        """
        from os.path import split as pathSplit
        path, file = pathSplit(self.filename)
        
        print("\n\tFile: ",file," \n\n on path:\n",path)
    
    def getFilename(self):
        """ return filename and path in the form (path, file)
        
        """
        from os.path import split as pathSplit
        return pathSplit(self.filename)

        

class TextFile(DataFile):
    """Generic class for importing text data of no specific format"""
    
    def __init__(self, filename: str, dataRow: int = 1, namesRow: int = 0,
                 unitsRow: int = -1, delimeter: str = ','):
        #Check if filename argument has been passed
        if not len(filename):
            #If not then launch file selection UI
            from tkinter import filedialog
            filename = filedialog.askopenfilename()
        
        self.filename = filename
        
        #Open file and read rows up to data
        fileheader = []
        with open(filename) as fID:
            for i in range(dataRow):
                fileheader.append(fID.readline().split(','))

            fileData = list(zip(*[row.split(',') for row in fID]))
        
        #read channel names from file header
        channelNames = fileheader[namesRow]
        #get channel count from length of channel names
        self.channelCount = len(channelNames)
        
        #if passed, get units, otherwise create blank list
        if unitsRow == -1:
            units = [''] * self.channelCount
        else:
            units = fileheader[unitsRow]
        

        # create data channels list
        self.dataChannels = []
        for i in range(self.channelCount):
            self.dataChannels.append(DataChannel(channelNames[i], 
                                                 np.float_(fileData[i]), units[i]))
        
                

class CompressorRigFile(DataFile):
    """DataFile type specifically for handling Compressor test rig logged data
    
    CompressorRigFile objects contain an array of DataFileChannel types for
    each channel of data contained within the file.
    
    DataFileChannel objects can be accessed via
        CompressorRigFile.dataChannels
    
    Currently implemented methods:
        getChannelNames
        showChannels
        getChannelData
    """
    
    def __init__(self, filename = ""):
        if not len(filename):
            from tkinter import filedialog
            filename = filedialog.askopenfilename()
        
        self.filename = filename
        self.calculatedChannels = []
        
        with open(filename) as fID:
            firstline = fID.readline().split(',')
            self.channelCount = int(firstline[2])
            self.dataPoints = int(firstline[3])
            
            secondline = fID.readline().split(',')
            self.frequency = int(secondline[2])
            self.motorSpeed = int(secondline[4][:secondline[4].find('k')]) * 1000
            self.pulsator = int(secondline[6][:secondline[6].find(' RPM')])
            self.valveOpening = int(secondline[8])
            
            # Extract channel metadata
            fID.readline()
            scalingTypes = fID.readline().split(',')
            scalingFactors = []
            
            # create empty scaling factors array
            for i in range(self.channelCount):
                scalingFactors.append([0.0] * 5)
                
            
            for row in range(5):
                factors = fID.readline().split(',')
                for channel in range(self.channelCount):
                    scalingFactors[channel][row] = float(factors[channel])
            
            
            tempOffset = [float(x) for x in fID.readline().split(',')]
            
            channelNames = fID.readline().split(',')
            channelUnits = fID.readline().split(',')
            
            # create empty scaling factors array
            data = []
            for i in range(self.channelCount):
                data.append([0.0] * self.dataPoints)
            
            for row in range(self.dataPoints):
                thisData = fID.readline().split(',')
                for channel in range(self.channelCount):
                    data[channel][row] = float(thisData[channel])
            
            # Create data channel objects
            self.dataChannels = []
            
            for i in range(self.channelCount):
                self.dataChannels.append(ScaledDataChannel(channelNames[i], data[i],
                                                         channelUnits[i],scalingFactors[i],scalingType="none"))
    

    