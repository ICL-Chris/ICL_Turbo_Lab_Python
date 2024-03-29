#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for containing data created in the lab and read from lab data files


Created on Sat Nov  6 19:21:55 2021

@author: cnoon
"""
import os.path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def openDataFile(file: str = ""):
    """Function to open a DataFile using tag in file header"""

    if not len(file):
        from tkinter import filedialog, Tk
        root = Tk()
        root.withdraw()

        file = filedialog.askopenfilename()
        root.destroy()

        if not len(file):
            print("No file selected")
            return

    with open(file) as fID:
        try:
            header = fID.readline().split(',')
        except UnicodeDecodeError:
            print("Error reading file - please check file type")
            return None

    match header[0]:
        case "CompressorRigFile" | "Compressor Rig V2 logfile":
            return CompressorRigFile(file)
        case "LegacyTurboFile":
            print("not yet implemented")
            pass
        case "TurboRigFile":
            return TurboRigFile(file)
        case "Theme5 TurboFile" | "Theme 5 turbo rig datafile":
            return Theme5TurboFile(file)
        case "HeatTrap":
            return HeatTrapFile(file)
        case _:
            print("filetype not matched")
            return ScaledDataFile(file)


def createFileSet(method: str = "selectFiles", reduce: str = "mean",
                  include_subfolders=False, file_types: str = "csv"):
    from tkinter import filedialog, Tk
    from pathlib import Path
    import os
    search_path = "*." + file_types
    if include_subfolders:
        search_path = "*/*." + file_types

    root = Tk()
    root.withdraw()
    filenames = []
    match method:
        case "selectFiles":
            filenames = filedialog.askopenfilenames(defaultextension=search_path)
            if not len(filenames):
                print("no files selected")
                return

        case "selectFolder":
            folder = filedialog.askdirectory(mustexist=True, title="Select folder containing files")
            if not len(folder):
                print("no folder selected")
                return
            current_path = os.getcwd()
            os.chdir(folder)
            for item in Path().glob(search_path):
                if item.is_file():
                    filenames.append(item.name)
            os.chdir(current_path)

        case "currentFolder":
            for item in Path().glob(search_path):
                if item.is_file():
                    filenames.append(item.name)

    root.destroy()

    # check for any files
    if not len(filenames):
        print("no suitable files found")
        return

    skipped_files = []

    for i in range(len(filenames)):
        data_file = openDataFile(filenames[i])
        if data_file == None:
            skipped_files.append(filenames[i])
        else:
            break

    # create a FileSet from the returned object

    if isinstance(data_file, DataFile):
        file_set = FileSet(data_file)
    elif isinstance(data_file, CompressorRigFile):
        file_set = FileSetCR(data_file)
    elif isinstance(data_file, TurboRigFile):
        file_set = FileSetTurbo(data_file)
    else:
        print("unknown data file type ", type(data_file))
        return

    for i in range(i + 1, len(filenames)):
        try:
            file_set.addFiles(openDataFile(filenames[i]))
        except:
            pass

    return file_set


class DataChannel:
    """Contains the data, and associated parameters for a channel

    """

    type = "dataChannel"

    def __init__(self, name: str, data: list, units: str = "", offset: float = 0):
        self.data = np.asarray(data, dtype=(np.float32))
        self.name = name
        self.units = units

    def rename(self, newName):
        """Update the name of the channel"""

        self.name = newName

    def downSample(self, factor=10.0, average=False):
        """Reduce the number of dataPoints in the channel by a factor"""

        limit = self.data.size // factor

        if average:
            selection = np.array(np.arange(limit) * factor, dtype=int)

            self.data = self.data[selection]

    def toPandas(self, timeIndex = []):
        import pandas as pd
        if not timeIndex:
            return pd.Series(self.data, name=self.name)
        else:
            return pd.Series(self.data, index=timeIndex, name=self.name)

    def __add__(self, other):

        return self.data + other

    def __sub__(self, other):

        return self.data - other

    def __mul__(self, other):
        return self.data * other

class ScaledDataChannel(DataChannel):
    """Specific class for reading test data from logged datafile"""

    type = "scaled data channel"

    def __init__(self, name: str, data, units: str = "", scalingFactors=[0, 0, 0, 0, 0],
                 offset: float = 0, scalingType=""):
        self.data = np.array(data, dtype=(np.float32))
        self.name = name
        self.units = units
        self.offset = offset
        self.scalingFactors = scalingFactors
        self.scalingType = scalingType

    def unscale(self, originalRange=[-10, 10]):
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

    def changeOffset(self, newOffset):
        """ update dataChannel to have an alternative offSet """

        self.data = self.unOffset() + newOffset
        self.offset = newOffset

    def reScale(self, newFactors, newUnits="", newName=""):
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
    """contains data channels and parameters from logged test data"""

    def __init__(self, filename="", empty=False):

        self.isValid = False

        if not empty:
            # if there is no filename passed, call the file select UI
            if not len(filename):
                from tkinter.filedialog import askopenfilename
                from tkinter import Tk
                root = Tk()
                root.withdraw()
                filename = askopenfilename()
                root.destroy()

                if not len(filename):
                    print("no file selected")
                    return

            self.filename = filename
            self.dataChannels = []
            self.channelCount = 0
            self.dataPoints = 0
            self.frequency = 0
            self.isValid = True

    def getChannelNames(self, channelSet=[]):
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

    def getUnits(self, channelSet=[]):
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
            print(i, '\t', channel.name)
            i += 1

        print("\t\n")

    def getChannelData(self, channel_set=[]):
        """
        Return specific channel data
        
        returns all data if no channels are specified
        """
        if not channel_set:
            channel_set = list(range(self.channelCount))
        else:
            channel_set = self.__ID(channel_set)

        if len(channel_set) == 0:
            channel_set = list(range(self.channelCount))

        data = self.dataChannels[channel_set[0]].data

        for i in range(1, len(channel_set)):
            data = np.vstack((data, [self.dataChannels[channel_set[i]].data]))

        return data

    def plotData(self, channelSet, show_plot=True, *args):
        """Plot directly channels of data
        
        """
        if type(channelSet) == int: channelSet = [channelSet]

        # create the plot
        plt.plot(self.getChannelData(channelSet).T, *args)

        # format the plot
        plt.xlabel("Datapoints")
        if len(channelSet) == 1:
            plt.ylabel(self.__plotLabel(channelSet))
        else:
            plt.legend(self.__plotLabels(channelSet))

        plt.grid(True)

        if show_plot:
            plt.show()

    def plotTime(self, channelSet, show_plot=True, *args):
        """Plots the selected channels against the time channel
        
        """
        if type(channelSet) == int: channelSet = [channelSet]

        plt.plot(self.getChannelData([0]), self.getChannelData(channelSet).T, *args)

        # format the plot
        plt.xlabel(self.__plotLabel(0))
        if len(channelSet) == 1:
            plt.ylabel(self.__plotLabel(channelSet))
        else:
            plt.legend(self.__plotLabels(channelSet))

        plt.grid(True)
        if show_plot:
            plt.show()

    def plotXY(self, xChannel, yChannels, show_plot=True, *args):
        """
        Plots channels specified in yChannels against xChannel
        """
        # if multiple xChannels are passed then use only the first item
        if type(xChannel) == list: xChannel = xChannel[0]
        if type(yChannels) == int: yChannels = [yChannels]

        # plot the data
        plt.plot(self.getChannelData(xChannel), self.getChannelData(yChannels).T, *args)

        # format the plot
        plt.xlabel(self.__plotLabel(xChannel))
        if len(yChannels) == 1:
            plt.ylabel(self.__plotLabel(yChannels))
        else:
            plt.legend(self.__plotLabels(yChannels))

        plt.grid(True)
        if show_plot:
            plt.show()

    def showFilename(self):
        """ Print filename to screen
        
        """
        from os.path import split as pathSplit
        path, file = pathSplit(self.filename)

        print("\n\tFile: ", file, " \n\n on path:\n", path)

    def getFilename(self):
        """ return filename and path in the form (path, file)
        
        """
        from os.path import split as pathSplit
        return pathSplit(self.filename)

    def indexFromChannelName(self, channelSet):
        """Return the channel index for channel names defined in channelSet
        
        """

        return self.__ID(channelSet)

    def cyclicAverage(self, primary, *,
                      channelSet=[],
                      cycleRepeats=1,
                      threshold=0.5,
                      halfWindowSize=50,
                      showPlot=True,
                      align=True,
                      force_frequency=0):
        """returns 3D numpy array containing wrapped data from each cycle
        as identified by the cycle using the primaryChannel to identify
        the cyclic nature of the data
        
        """
        if len(channelSet) == 0:
            channelSet = list(range(self.channelCount))
        else:
            # ensure the time and primary channel are included in the output
            channelSet.append([0, primary])
            channelSet = self.__ID(channelSet)

        # First extract the primary channel from the dataSet
        primaryChannel = self.getChannelData(primary)
        # created a filtered version of the channel
        L = 2 * halfWindowSize + 1
        filteredChannel = []
        for i in range(0, self.dataPoints - L):
            filteredChannel.append(np.mean(primaryChannel[i:i + L]))

        # identify crossing threshold
        peak = max(filteredChannel)
        trough = min(filteredChannel)
        threshold = trough + ((peak - trough) * abs(threshold))

        # identify threshold crossing indices
        if threshold > 0:
            above = np.greater(filteredChannel, abs(threshold))
        else:
            above = np.less_equal(filteredChannel, abs(threshold))

        crossingPoints = np.nonzero(above[1:] > above[:-1])[0]

        if force_frequency:
            period = self.frequency / force_frequency
        else:
            period = (crossingPoints[-1] - crossingPoints[0]) / (len(crossingPoints) - 1)

        # create the 3D output array
        averageCycle = np.zeros((int(period), len(channelSet), int((self.dataPoints // period))))

        # get the requested channels for the output
        data = self.getChannelData(channelSet)

        position = crossingPoints[0]
        i = 0

        while position < self.dataPoints - period:
            averageCycle[:, :, i] = data[:, int(position):int(position) + int(period)].T
            position += period
            i += 1

        if i <= averageCycle.shape[2]:
            averageCycle = np.delete(averageCycle, range(i, averageCycle.shape[2]), axis=2)

        # create time data
        time = np.array(range(0, int(period))) * 1 / self.frequency

        if showPlot:
            primary = self.__ID(primary)

            fig, ax = plt.subplots()

            for i in range(i - 1):
                ax.scatter(time, np.squeeze(averageCycle[:, primary, i]), marker='.', color="0.8", alpha=0.2)

            ax.plot(time, np.squeeze(np.mean(averageCycle[:, primary, :], axis=2)), color="C3")
            plt.show()

        return (np.squeeze(np.mean(averageCycle, axis=2)), time, self.frequency / period)

    def delta(self, channelSet):
        """Returns an array of length = dataPoints containing the point-
        by-point difference in each channel of channelSet
        index 0 === 0
        """
        print("not done yet")
        pass

    def reduceToMean(self):
        """Creates a deep copy of the DataFile with a single value for each channel as them mean of each channel"""
        import copy

        newDataFile = copy.deepcopy(self)

        for i in range(self.channelCount):
            newDataFile.dataChannels[i].data = np.mean(newDataFile.dataChannels[i].data)

        newDataFile.dataPoints = 1

        return newDataFile

    def reduceToCycle(self, primary, *,
                      cycleRepeats=1,
                      threshold=0.5,
                      halfWindowSize=50,
                      showPlot=False,
                      align=True):
        """Creates a deep copy of the DataFile ith the data replaced by a single-cycle reduction"""

        import copy

        newDataFile = copy.deepcopy(self)

        averageCycle, time = self.cyclicAverage(primary=primary,
                                                cycleRepeats=cycleRepeats,
                                                threshold=threshold,
                                                halfWindowSize=halfWindowSize,
                                                showPlot=showPlot,
                                                align=align)

        newDataFile.dataPoints = len(time)

        for i in range(newDataFile.channelCount):
            newDataFile.dataChannels[i].data = averageCycle[:, i]

        return newDataFile

    def toPandas(self):
        """Produce a panda dataFrame containing data from the DataFile"""
        import pandas as pd
        if self.dataPoints == 1:
            return pd.Series(data=self.getChannelData()[:,0], index=self.getChannelNames(), name="Mean Values")

        else:
            channelDict = {}
            for channel, name in zip(self.dataChannels, self.getChannelNames()):
                channelDict[name] = channel.toPandas()

            return pd.DataFrame(channelDict)

    def channelToPandas(self, channel):

        channel = self.__ID(channel)[0]

        return self.dataChannels[channel].toPandas()

    def FFT(self, channel, plot_graph=True, freq_cutoff=1500):

        if isinstance(channel, list):
            channel = channel[0]

        # perform the FFT, normalize the response and select only the sensible frequency range
        ft = np.fft.fft(self.getChannelData(channel))[1:(self.dataPoints // 2) + 1] / self.dataPoints

        # generate corresponding frequency range
        fr = self.frequency * np.arange(1, (self.dataPoints // 2) + 1) / self.dataPoints

        if plot_graph:
            plt.plot(fr, abs(ft.real))
            plt.xlim(0, freq_cutoff)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(self.getChannelNames(channel)[0] + " frequency response")
            plt.grid(True)
            plt.show()

    def __ID(self, names, withFound=False):
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

        # ensure if a single item is passed it is treated as a list
        if type(names) is not list: names = [names]

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
                    print("index ", name, " out of range. (>=", self.channelCount, ")")
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
        # Check if filename argument has been passed
        if not len(filename):
            # If not then launch file selection UI
            from tkinter import filedialog
            filename = filedialog.askopenfilename()

        self.filename = filename

        # Open file and read rows up to data
        fileheader = []
        with open(filename) as fID:
            for i in range(dataRow):
                fileheader.append(fID.readline().split(','))

            fileData = list(zip(*[row.split(',') for row in fID]))

        # read channel names from file header
        channelNames = fileheader[namesRow]
        # get channel count from length of channel names
        self.channelCount = len(channelNames)

        # if passed, get units, otherwise create blank list
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

    def __init__(self, filename=""):
        super().__init__(filename)

        if self.isValid:
            with open(self.filename) as fID:
                fileParameters = fID.readline().split(',')

                self.dataFileVersion = fileParameters[1]

                self.channelCount = int(fileParameters[2])
                self.dataPoints = int(fileParameters[3])

                match self.dataFileVersion:
                    case "1.1" | "1.2":
                        self = self.scaledDataFileReader13(fID)
                        self.customFileVersion = "1.0"
                        self
                    case "1.3":
                        self = self.scaledDataFileReader13(fID)
                        self.customFileVersion = fileParameters[6].rstrip()
                    case _:
                        print("file version '", self.dataFileVersion, "' not recognised", sep="")
                        self.isValid = False

    def scaledDataFileReader13(dataFile: DataFile, fID):
        """open a .csv file and import the contained data.
        
        This assumes DataFile version 1.3 with the associated header structure
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

        return dataFile

    def showOffsets(self, channelSet=[]):
        """print the selected channels' (or all) offsets to screen"""

        channelSet = self.__ID(channelSet)

        for i in range(len(channelSet)):
            channel = self.dataChannels[channelSet[i]]
            print("\t", i, channel.name, ": offset = ", channel.offset)

    def getUnscaledData(self, channelSet):
        """return the unscaled data for each channel defined in channelSet"""

        channelSet = self.__ID(channelSet)

        data = self.dataChannels[channelSet[0]].unscale()

        for i in range(1, len(channelSet)):
            data = np.vstack((data, [self.dataChannels[channelSet[i]].unscale()]))

        return data

    def getUnOffsetData(self, channelSet):
        """return the unscaled data for each channel defined in channelSet"""

        channelSet = self.__ID(channelSet)

        data = self.dataChannels[channelSet[0]].unOffset()

        for i in range(1, len(channelSet)):
            data = np.vstack((data, [self.dataChannels[channelSet[i]].unPffset()]))

        return data

    def reScale(self, channel, newScale, newUnits="", newName=""):
        """rescales the identified channel with newScale"""

        if type(channel) == list: channel = channel[0]

        channel = self.__ID(channel)[0]

        self.dataChannels[channel].rescale(newScale, newUnits, newName)

    def changeOffset(self, channel, newOffset=0):
        """update the offset from the specified channel"""

        if type(channel) == list: channel = channel[0]
        print("Cannot remove multiple offsets, only channel", channel, "hes been un-Offset")

        channel = self.__ID(channel)[0]

        self.dataChannels[channel].changeOffset(newOffset)


class CompressorRigFile(ScaledDataFile):
    """DataFile type specifically for handling Compressor test rig logged data
    
    CompressorRigFile objects contain an array of DataFileChannel types for
    each channel of data contained within the file.
    
    DataFileChannel objects can be accessed via
        CompressorRigFile.dataChannels
    """

    def __init__(self, filename=""):
        super().__init__(filename)

        if self != None:
            match self.customFileVersion:
                case "1.0":
                    self.timestamp = datetime.strptime(self.customHeader[0], "%d/%m/%Y %H:%M")
                    self.frequency = int(self.customHeader[2])
                    self.motorSpeed = 0
                    self.pulsator = 0
                    self.valvePosition = 0
                case "1.1" | "1.2":
                    self.timestamp = datetime.strptime(self.customHeader[0], "%d/%m/%Y %H:%M")
                    self.frequency = int(self.customHeader[2])
                    self.motorSpeed = int(self.customHeader[4])
                    self.pulsator = int(self.customHeader[6])
                    self.valvePosition = int(self.customHeader[8])
                case "1.3":
                    self.timestamp = datetime.strptime(self.customHeader[0], "%d/%m/%Y %H:%M:%S")
                    self.frequency = int(self.customHeader[2])
                    self.motorSpeed = int(self.customHeader[4])
                    self.pulsator = int(self.customHeader[6])
                    self.valvePosition = int(self.customHeader[8])
                case _:
                    print("unknown custom file version ", self.customFileVersion)


class TurboRigFile(ScaledDataFile):
    """DataFile type specifically for handling Turbo test rig logged data
    
    TurboRigFile objects contain an array of DataFileChannel types for
    each channel of data contained within the file.
    
    DataFileChannel objects can be accessed via
        TurboRigFile.dataChannels
    """

    def __init__(self, filename=""):
        super().__init__(filename)

        self.frequency = int(self.customHeader[2])
        self.motorSpeed = int(self.customHeader[4])
        try:
            self.pulsator = int(self.customHeader[6])
            self.valvePosition = int(self.customHeader[8])
        except:
            self.pulsator = int(float(self.customHeader[6]))
            self.valvePosition = int(float(self.customHeader[8]))


class HeatTrapFile(ScaledDataFile):
    """DataFile type specifically for handling Turbo test rig logged data

    TurboRigFile objects contain an array of DataFileChannel types for
    each channel of data contained within the file.

    DataFileChannel objects can be accessed via
        TurboRigFile.dataChannels
    """

    def __init__(self, filename=""):
        super().__init__(filename)

        self.supplyPressure = float(self.customHeader[2])
        self.outletValve = float(self.customHeader[4])
        self.propValve = float(self.customHeader[6])
        self.AVT_Speed = int(self.customHeader[8])
        self.frequency = int(self.customHeader[10])


class Theme5TurboFile(TurboRigFile):
    """ DataFile type for Theme 5"""

    def __init__(self, filename=""):
        super().__init__(filename)

        if self.isValid:
            customParameters = self.customFileVersion
            try:
                self.timestamp = datetime.strptime(self.customHeader[0], "%d/%m/%Y %H:%M:%S")
            except:
                self.timestamp = datetime.strptime(self.customHeader[0], "%d/%m/%y %H:%M:%S")

            match self.customFileVersion:
                case '0.5':
                    self.frequency = 5000
                    self.speed = float(self.customHeader[2])
                    self.massflow = 0
                    mfr = dict(zip([2.5, 1.0, 0.4, 0], [0.25, 0.5, 0.75, 1]))
                    try:
                        self.flowRatio = mfr[float(self.customHeader[6]) / float(self.customHeader[4])]
                    except:
                        self.flowRatio = 0

                    self.targetPin = float(self.customHeader[8])
                    self.targetPout = float(self.customHeader[8])
                    self.wastegate = float(self.customHeader[10])
                case '0.6':
                    self.frequency = float(self.customHeader[2])
                    self.speed = float(self.customHeader[4])
                    self.massflow = float(self.customHeader[6])
                    self.flowRatio = float(self.customHeader[8])
                    self.targetPin = float(self.customHeader[10])
                    self.targetPout = float(self.customHeader[12])
                    self.wastegate = float(self.customHeader[14])
                    self.pulsator = float(self.customHeader[16])
        else:
            print("unable to create Theme 5 Turbo File")
            return None


class FileSet():
    """the FileSet class is a container for multiple DataFile type objects"""

    def __init__(self, data_files):
        """ initialises the FileSet object with one or more DataFile objects"""

        success = True
        self.dataFiles = []
        self.fileCount = 0
        self.dataPoints = 0
        self.equalLengths = True

        if type(data_files) != list: data_files = [data_files]

        for data_file in data_files:
            if isinstance(data_file, DataFile):
                self.dataFiles.append(data_file)
            else:
                success = False

        self.updateFileSet()

        if success == False:
            print("warning: not all files were added to the FileSet")

    def __len__(self):
        return len(self.dataFiles)

    def addFiles(self, dataFiles):
        """adds a new DataFile to the FileSet"""

        if type(dataFiles) != list: dataFiles = [dataFiles]

        for file in dataFiles:
            if isinstance(file, DataFile):
                self.dataFiles.append(file)

        self.updateFileSet()

    def updateFileSet(self):
        """Updates the parameters of the FileSet"""

        self.fileCount = len(self.dataFiles)
        dataPoints = []
        filenames = []
        for file in self.dataFiles:
            dataPoints.append(file.dataPoints)
            filenames.append(os.path.basename(file.filename))

        self.dataPoints = min(dataPoints)
        self.files = filenames

        if max(dataPoints) != min(dataPoints):
            self.equalLengths = False

    def plot(self, channel, *args):
        for data_file in self.dataFiles:
            data_file.plotData(channel, show_plot=False, *args)
        plt.legend(self.files)
        plt.show()

    def plotTime(self, channel, *args):
        for data_file in self.dataFiles:
            data_file.plotTime(channel, show_plot=False, *args)
        plt.legend(self.files)
        plt.show()

    def plotXY(self, x_channel, y_channel, *args):
        for data_file in self.dataFiles:
            data_file.plotXY(x_channel, y_channel, False, *args)
        plt.legend(self.files)
        plt.show()

    def reduceToMean(self):
        import copy

        newFileSet = copy.deepcopy(self)

        for i in range(self.fileCount):
            newFileSet.dataFiles[i] = newFileSet.dataFiles[i].reduceToMean()

        return newFileSet

    def reduceToCycle(self, primary, *,
                      channelSet=[],
                      cycleRepeats=1,
                      threshold=0.5,
                      halfWindowSize=50,
                      showPlot=False,
                      align=True):

        import copy
        newFileSet = copy.deepcopy(self)

        for i in range(newFileSet.fileCount):
            newFileSet.dataFiles[i] = newFileSet.dataFiles[i].reduceToCycle(primary=primary,
                                                                            cycleRepeats=cycleRepeats,
                                                                            threshold=threshold,
                                                                            halfWindowSize=halfWindowSize,
                                                                            showPlot=showPlot,
                                                                            align=align)

        return newFileSet

    def showChannelNames(self):
        self.dataFiles[0].showChannels()

    def channelToPandas(self, channel):
        channelDict = {}

        for file in self.dataFiles:
            channelDict[file.getFilename()[1]] = file.channelToPandas(channel=channel)

        return pd.DataFrame(channelDict)

    def toPandas(self):
        dataFrames = []

        for file in self.dataFiles:
            dataFrames.append(file.toPandas())

        return dataFrames

class FileSetCR(FileSet):

    def __init__(self, data_file: CompressorRigFile):
        super().__init__(data_file)

    def addFile(self, dataFile: CompressorRigFile):
        self.dataFiles.append(dataFile)

    def addFiles(self, dataFiles: list):
        for dataFile in dataFiles:
            self.addFile(dataFile)


class FileSetTurbo(FileSet):

    def __init__(self, dataFile: TurboRigFile):
        super().__init__(dataFile)
