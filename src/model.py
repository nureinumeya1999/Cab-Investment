import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cabData = pd.read_csv('DataSets/Cab_Data.csv')
cityData = pd.read_csv('DataSets/Cab_Data.csv')
customerData = pd.read_csv('DataSets/Cab_Data.csv')
transactionData = pd.read_csv('DataSets/Cab_Data.csv')

X = cabData[[]]
