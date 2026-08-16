# import mlflow 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer


df = pd.read_csv(r"C:\Users\HARSHAL\OneDrive\Desktop\Delivery Pulse\Zomato Dataset.csv")
df = df.drop(columns=["ID","Delivery_person_ID","Restaurant_latitude","Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude","Type_of_order","Type_of_vehicle","Festival"])

print(df)