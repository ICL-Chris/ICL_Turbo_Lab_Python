import DataConstructs as DC

filename = "/Users/cnoon/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Test Data/Compressor Rig Data/22-11-29/900/40000 85-0.csv"

f = DC.ScaledDataFile()

f.getChannelNames(9)

f.plotTime(11)

(t, data) = f.cyclicAverage(9)