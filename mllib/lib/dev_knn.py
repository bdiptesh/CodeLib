import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

path = "/media/ph33r/Data/Project/mllib/Git/data/input/"

fn_ip = "iris.csv"

df = pd.read_csv(path + fn_ip)

y_var = ["y"]
x_var = ["x1", "x2", "x3", "x4"]

scaler = MinMaxScaler()
scaler.fit(df[x_var])

df_x_var = scaler.transform(df[x_var])
df_y_var = df[y_var].values.ravel()

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(df_x_var, df_y_var)

y_hat = classifier.predict(df_x_var)

tmp = classification_report(y_hat, df_y_var, output_dict=True, zero_division=0)
model_summary = tmp["weighted avg"]
model_summary["accuracy"] = tmp["accuracy"]
model_summary
