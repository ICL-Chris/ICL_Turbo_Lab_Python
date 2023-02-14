#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for containing data created in the lab and read from lab data files


Created on Sat Nov  6 19:21:55 2021

@author: cnoon
"""

import numpy as np
from tkinter import filedialog as FD
import matplotlib.pyplot as plt

class DataChannel:
    """Contains the data, and associated parameters for a channel

    """

    type = "dataChannel"

    def __init__(self, name: str, data: list, units: str = "", offset: float = 0):
        self.data = np.asarray(data, dtype=(np.float32))
        self.name = name
        self.units = units

class ScaledDataChannel(DataChannel):
    """Specific class for reading test data from logged datafile"""

    type = "scaled data channel"

    def __init__(self, name: str, data, units: str = "", scalingFactors= [0,0,0,0,0],
                 offset: float = 0, scalingType= ""):
        self.data = np.array(data, dtype=(np.float32))
        self.name = name
        self.units = units
        self.offset = offset
        self.scalingFactors = scalingFactors
        self.scalingType = scalingType
    
    def unscale(self, originalRange=[-10,10]):
        """return raw data with scaling factors removed"""
        x = self.scalingFactors

        match self.scalingType:
            case "none":
                return self.data
            case "linear":
                return (self.data - x[0]) / x[1]
            case "polynomial":
                print("not yet implemented")
                return np.zeros(self.data.shape)
    
    def unOffset(self):
        """return data with live offset removed"""

        return self.data - self.offset

    def reScale(self, newFactors, newUnits = "", newName = ""):
        """return new ScaledDataChannel with data rescaled"""

        from numpy.polynomial import Polynomial as poly

        if len(newName) == 0:
            newName = self.name
        if len(newUnits) == 0:
            newUnits = self.units
        
        newPoly = poly(newFactors)

        return ScaledDataChannel(newName,
                newPoly(self.unscale()),
                newUnits,
                scalingFactors=newPoly.coef,
                scalingType="polynomial")

class DataFile:
    """contains data channels and parameters from logged etst data"""
    
    def __init__(self, filename = ""):
        
        if not len(filename):
            from tkinter import filedialog
            filename = filedialog.askopenfilename()
        
        self.filename = filename
        self.dataChannels = []
        self.channelCount = 0
        self.dataPoints = 0
        self.frequency = 0

    def getChannelNames(self, channelSet = []):
        """
        return names of channels contained within the dataFile
        """

        if channelSet == []:
            channelSet = list(range(self.channelCount))
        
        channelSet = self.__ID(channelSet)
        
        channelNames = []
        for channel in channelSet:
            channelNames.append(self.dataChannels[channel].name)
        
        return channelNames
    
        
    def getUnits(self, channelSet = []):
        """
        return units for each channel specified in channelSet
        """

        # if there is no channelSet specified then return all channels
        if channelSet == []:
            channelSet = list(range(self.channelCount))
        
        # convert any channel names to indices
        channelSet = self.__ID(channelSet)
        
        channelUnits = []
        # extract units for each channel in channelSet
        for channel in channelSet:
            channelUnits.append(self.dataChannels[channel].units)
        
        return channelUnits

    
    def showChannels(self):
        """
        return indices and channel names for all channels in the file
        """
        i = 0
        for channel in self.dataChannels:
            print(i,'\t',channel.name)
            i += 1

        print("\t\n")
        
    
    def getChannelData(self, channelSet = []):
        """
        Return specific channel data
        
        returns all data if no channels are specified
        """

        channelSet = self.__ID(channelSet)

        if len(channelSet) == 0:
            channelSet = list(range(self.channelCount))
                
        data = self.dataChannels[channelSet[0]].data
        
        for i in range(1,len(channelSet)):
            data = np.vstack((data,[self.dataChannels[channelSet[i]].data]))
        
        return data

    
    def plotData(self, channelSet, *args):
        """Plot directly channels of data
        
        """
        if type(channelSet) == int: channelSet = [ channelSet ]
        
        # create the plot
        plt.plot(self.getChannelData(channelSet).T, *args)

        # format the plot
        plt.xlabel("Datapoints")
        if len(channelSet) == 1:
            plt.ylabel(self.__plotLabel(channelSet))
        else:
            plt.legend(self.__plotLabels(channelSet))
        
        plt.show()


    def plotTime(self, channelSet, *args):
        """Plots the selected channels against the time channel
        
        """
        if type(channelSet) == int: channelSet = [ channelSet ]

        plt.plot(self.getChannelData(0),self.getChannelData(channelSet).T)

        # format the plot
        plt.xlabel("Time (s)")
        if len(channelSet) == 1:
            plt.ylabel(self.__plotLabel(channelSet))
        else:
            plt.legend(self.__plotLabels(channelSet))

        plt.show()

    def plotXY(self, xChannel, yChannels, *args):
        """
        Plots channels specified in yChannels against xChannel
        """
        # if multiple xChannels are passed then use only the first item
        if type(xChannel) == list: xChannel = xChannel[0]
        if type(yChannels) == int: yChannels = [ yChannels ]

        # plot the data
        plt.plot(self.getChannelData(xChannel), self.getChannelData(yChannels).T, *args)

        # format the plot
        plt.xlabel(self.__plotLabel(xChannel))
        if len(yChannels) == 1:
            plt.ylabel(self.__plotLabel(yChannels))
        else:
            plt.legend(self.__plotLabels(yChannels))
        
        plt.show()
    
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
    
    def indexFromChannelName(self, channelSet):
        """Return the channel index for channel names defined in channelSet
        
        """

        return self.__ID(channelSet)

    def cyclicAverage(self, primary, * ,
                        channelSet = [],
                        cycleRepeats = 1,
                        threshold = 0.5,
                        halfWindowSize = 50,
                        showPlot = True,
                        align = True):
        """returns 3D numpy array containing wrapped data from each cycle
        as identified by the cycle using the primaryChannel to identify
        the cyclic nature of the data
        
        """
        if len(channelSet) == 0:
            channelSet = list(range(self.channelCount))
        else:
            # ensure the time and primary channel are included in the output
            channelSet.append([0,primary])
            channelSet = self.__ID(channelSet)

        # First extract the primary channel from the dataSet
        primaryChannel = self.getChannelData(primary)
        # created a filtered version of the channel
        L = 2 * halfWindowSize + 1
        filteredChannel = []
        for i in range(0, self.dataPoints - L):
            filteredChannel.append(np.mean(primaryChannel[i:i+L]))

        # identify crossing threshold
        peak = max(filteredChannel)
        trough = min(filteredChannel)
        threshold = trough + ((peak - trough) * threshold)

        # identify threshold crossing indices
        above = np.greater(filteredChannel, threshold)
        crossingPoints = []
        for i in range(len(above)-1):
            if above[i+1] > above[i]:
                crossingPoints.append(i)

        periods = np.subtract(crossingPoints[1:], crossingPoints[:-1])

        # calculate the period length
        # median period provides a rough estimate of period length
        medPeriod = np.median(periods)
        # mean of period lengths close to median (i.e. full cycles) provides best approximation
        period = periods[abs(periods - medPeriod) < (0.1 * medPeriod)].mean() * cycleRepeats

        # create the 3D output array
        averageCycle = np.zeros((int(period),len(channelSet),int((self.dataPoints//period)-1)))

        # get the requested channels for the output
        data = self.getChannelData(channelSet)

        position = crossingPoints[0]
        i = 0

        while position < self.dataPoints - period:
            averageCycle[:,:,i] = data[:,int(position):int(position)+int(period)].T
            position += period
            i += 1
        
        # create time data
        time = np.array(range(0,int(period))) * 1/self.frequency

        if showPlot:
            primary = self.__ID(primary)

            for i in range(i-1):
                plt.plot(time,np.squeeze(averageCycle[:,primary,i]),'k.')
            
            plt.plot(time,np.squeeze(np.mean(averageCycle[:,primary,:],axis=2)),'r')
            plt.show()


        return (time, averageCycle)
    
    def delta(self, channelSet):
        """Returns an array of length = dataPoints containing the point-
        by-point difference in each channel of channelSet
        index 0 === 0
        """
        print("not done yet")
        pass

        

    def __ID(self, names, withFound = False):
        """
        converts items in 'names' to the equivalent channel index.
        Any unmatched channels are ommitted
        """
        # create empty lists
        indices = []
        found = []
        channelNames = []

        # get lowercase copy o fevery channel name
        for channel in self.dataChannels:
            channelNames.append(channel.name.casefold())

        # ensure if a single item i spassed it is treated as a list
        if type(names) is not list: names = [ names ]

        # for each item provided...
        for name in names:
            # check whether it is an index (integer) already
            if type(name) != int:
                # check for a match against channel names list
                try:
                    indices.append(channelNames.index(name.casefold()))
                    found.append(True)
                # for a non-match print the un-matched channel and omit from the list
                except ValueError:
                    print("unable to match channel '", name, "'. ")
                    indices.append(0)
                    found.append(False)
            # add integers directly to indices list
            else:
                if name < self.channelCount:
                    indices.append(name)
                    found.append(True)
                else:
                    print("index ", name, " out of range. (>=", self.channelCount,")")
                    indices.append(0)
                    found.append(False)
    
        # remove duplicate channels
        indices = list(set(indices))

        # sort indices and found arrays

        indices, found = zip(*sorted(zip(indices, found)))

        if withFound:
            return list(indices), list(found)
        else:
            return list(indices)

        
    def __plotLabel(self, channel):
        """
        returns a string usable for a legend or axis label
        """

        channel = self.__ID(channel)
    
        return (self.getChannelNames(channel)[0] + " (" + self.getUnits(channel)[0] + ")")
    

    def __plotLabels(self, channelSet):
        """
        returns a list of strings usable for a legend
        """
        channelSet = self.__ID(channelSet)

        labels = []
        for channel in channelSet:
            labels.append(self.getChannelNames(channel)[0] + " (" + self.getUnits(channel)[0] + ")")

        return labels


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
        

class ScaledDataFile(DataFile):
    """DataFile containing channels using the standard ScaledDataChannel class.
    
    This class is designed to work with the 'standard' channel classes created in LabVIEW
    for reading, scaling and writing data to file.

    """

    def __init__(self, filename="", withSuccess = False):
        super().__init__(filename)

        with open(self.filename) as fID:
            fileParameters = fID.readline().split(',')
            
            self.dataFileVersion = fileParameters[1]
            self.customFileVersion = fileParameters[6].rstrip()
            self.channelCount = int(fileParameters[2])
            self.dataPoints = int(fileParameters[3])
            
            match self.dataFileVersion:
                case "1.1" | "1.2":
                    pass
                case "1.3":
                    self = scaledDataFileReader13(self, fID)




class CompressorRigFile(ScaledDataFile):
    """DataFile type specifically for handling Compressor test rig logged data
    
    CompressorRigFile objects contain an array of DataFileChannel types for
    each channel of data contained within the file.
    
    DataFileChannel objects can be accessed via
        CompressorRigFile.dataChannels
    """
    
    def __init__(self, filename = ""):
        super().__init__(filename)

        self.frequency = int(self.customHeader[2])
        self.motorSpeed = int(self.customHeader[4])
        self.pulsator = int(self.customHeader[6])
        self.valvePosition = int(self.customHeader[8])

class TurboRigFile(ScaledDataFile):
    """DataFile type specifically for handling Turbo test rig logged data
    
    TurboRigFile objects contain an array of DataFileChannel types for
    each channel of data contained within the file.
    
    DataFileChannel objects can be accessed via
        TurboRigFile.dataChannels
    """
    
    def __init__(self, filename = ""):
        super().__init__(filename)

        self.frequency = int(self.customHeader[2])
        self.motorSpeed = int(self.customHeader[4])
        self.pulsator = int(self.customHeader[6])
        self.valvePosition = int(self.customHeader[8])

        


def scaledDataFileReader13(dataFile:DataFile, fID):
    """open a .csv file and import the contained data.
    
    This assumes DataFile version 1.3 with the assoiocated header structure
    returns the DataFile with populated content including custom parameters
    """

    # open the specified file
        
    dataFile.customHeader = fID.readline().split(',')
    
    # Extract channel metadata
    fID.readline()
    scalingTypes = fID.readline().split(',')
    scalingFactors = []
    
    # create empty scaling factors array
    for i in range(dataFile.channelCount):
        scalingFactors.append([0.0] * 5)
        
    
    for row in range(5):
        factors = fID.readline().split(',')
        for channel in range(dataFile.channelCount):
            scalingFactors[channel][row] = float(factors[channel])
    
    
    tempOffset = [float(x) for x in fID.readline().split(',')]
    
    channelNames = fID.readline().split(',')
    channelUnits = fID.readline().split(',')
    
    # create empty scaling factors array
    data = []
    for i in range(dataFile.channelCount):
        data.append([0.0] * dataFile.dataPoints)
    
    for row in range(dataFile.dataPoints):
        thisData = fID.readline().split(',')
        for channel in range(dataFile.channelCount):
            data[channel][row] = float(thisData[channel])
    
    # Create data channel objects
    dataFile.dataChannels = []
    
    for i in range(dataFile.channelCount):
        dataFile.dataChannels.append(ScaledDataChannel(
            channelNames[i],
            data[i],
            channelUnits[i],
            scalingFactors[i],
            tempOffset[i],
            scalingTypes[i]))