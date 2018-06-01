"""
Central bank of brazil inflation rate
"""

import pandas as pd
import os


os.chdir('..\dataset')
pwd = os.getcwd()
data = pd.read_csv( pwd + os.sep + 'BCB-datasets-codes.csv')

print(data.isnull().T.any())
