import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from ydata_profiling import ProfileReport

df = pd.read_csv(r"C:\Users\HARSHAL\OneDrive\Desktop\Delivery Pulse\Zomato Dataset.csv")

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

profile = ProfileReport(df,title="Delivery Pulse",explorative=True)
profile.to_file("Delivery_Pulse.html")
