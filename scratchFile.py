import DataConstructs as DC

filename = "/Users/cnoon/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Test Data/Theme 5 pulsating trial/research run 2/23-12-11/Turbo File 11-33-09-0.csv"

f = DC.openDataFile(filename)

output = f.cyclicAverage(18)
