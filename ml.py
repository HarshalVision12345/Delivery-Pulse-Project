import mlflow 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv(r"C:\Users\HARSHAL\OneDrive\Desktop\Delivery Pulse\Zomato Dataset.csv")

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

print("Generating EDA report for Delivery Pulse ......")

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# Writing Experiment Name -----------> 
mlflow.set_experiment("Decision Tree - EXP1")

with mlflow.start_run():
    #  Log Full Interactive HTML Report (ydata-profiling) 
    profile = ProfileReport(
    df,
    title="Delivery Pulse - Exploratory Data Analysis",
    explorative=True
    )

    html_report_path ="Delivery_Pulse.html"
    profile.to_file(html_report_path)

    mlflow.log_artifact(html_report_path,artifact_path="EDA")

    print("EDA Succesfully to MLFlow")