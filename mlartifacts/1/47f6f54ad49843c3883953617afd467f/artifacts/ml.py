import mlflow 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv(r"C:\Users\HARSHAL\OneDrive\Desktop\Delivery Pulse\Zomato Dataset.csv")
df = df.drop(
    columns = [
        "ID",
        "Delivery_person_ID"
        ,"Restaurant_latitude"
        ,"Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival"]
        )

# =========================================================
# STEP 4: DATE & TIME PROCESSING (ADDED HERE)
# =========================================================

# Convert Order_Date string into a Datetime object
df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')

# Extract day of week (0=Monday, 6=Sunday)
df['day_of_week'] = df['Order_Date'].dt.dayofweek

# Create binary flag: 1 if Saturday/Sunday, else 0
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Define function to convert "HH:MM" string to total minutes
def time_to_minutes(time_str):
    if pd.isna(time_str) or str(time_str).strip() in ["NaN", "nan", ""]:
        return np.nan
    try:
        parts = str(time_str).strip().replace("@", "").split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan

# Convert time strings to minutes from midnight
df['order_time_min'] = df['Time_Orderd'].apply(time_to_minutes)
df['picked_time_min'] = df['Time_Order_picked'].apply(time_to_minutes)

# Extract hour of order (0 to 23)
df['order_hour'] = df['order_time_min'] // 60

# Calculate kitchen preparation time (Picked Time - Order Time)
df['prep_time_min'] = df['picked_time_min'] - df['order_time_min']

# Fix negative values for orders crossing midnight
df['prep_time_min'] = np.where(df['prep_time_min'] < 0, df['prep_time_min'] + 1440, df['prep_time_min'])

# Drop original text date/time columns
df = df.drop(columns=['Order_Date', 'Time_Orderd', 'Time_Order_picked', 'picked_time_min'])

# =========================================================
# STEP 5: SPLIT FEATURES AND TARGET
# =========================================================

target_col = "Time_taken (min)"

# Clean target values if they contain text (e.g. "(min) 24")
if target_col in df.columns and df[target_col].dtype == object:
    df[target_col] = df[target_col].astype(str).str.extract(r'(\d+)').astype(float)

x = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_cols = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
    "multiple_deliveries", "day_of_week", "is_weekend", "order_hour", "prep_time_min"
]

cat_cols = [
    "Weather_conditions", "Road_traffic_density", "City"
]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("Scaler", StandardScaler())
])

col_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("OHE", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("col", col_pipeline, cat_cols)
])

num_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("Scaler",StandardScaler())
])

col_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("OHE",OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num",num_pipeline,num_cols),
    ("col",col_pipeline,cat_cols)
])

max_depth = 3
criterion = 'poisson'
splitter = 'random'
random_state = 42


# Writing Experiment Name -----------> 
mlflow.set_experiment("Decision Tree - EXP1")

with mlflow.start_run():
    #  Log Full Interactive HTML Report (ydata-profiling) 
    # profile = ProfileReport(
    # df,
    # title="Delivery Pulse - Exploratory Data Analysis",
    # explorative=True
    # )

    # html_report_path ="Delivery_Pulse.html"
    # profile.to_file(html_report_path)

    # mlflow.log_artifact(html_report_path,artifact_path="EDA")

    # print("EDA Succesfully to MLFlow")

  # Preprocess Training and Testing Data --------------------->

    X_train_preprocess = preprocessor.fit_transform(X_train)
    X_test_preprocess = preprocessor.transform(X_test)

    model = DecisionTreeRegressor(
    max_depth=max_depth,
    criterion=criterion,
    splitter=splitter,
    random_state=random_state
    )

    model.fit(X_train_preprocess,y_train)

    y_pred = model.predict(X_test_preprocess)

    score = r2_score(y_test,y_pred)

    mlflow.log_param("Max_depth",max_depth)
    mlflow.log_param("Criterion",criterion)
    mlflow.log_param("Splitter",splitter)
    mlflow.log_param("Random_State",random_state)

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model,artifact_path="Decision_Tree_Regressor")
    mlflow.log_metric("R2_Score",score)

    print(f"Score = {score}")
    print("Succesfully Completed The Preprocessing and model making and both loged in")