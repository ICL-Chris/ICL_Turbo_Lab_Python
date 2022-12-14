import DataConstructs as DC
from tkinter import filedialog

filename = filedialog.askopenfilename()

print(filename)

f = DC.DataFile(filename)