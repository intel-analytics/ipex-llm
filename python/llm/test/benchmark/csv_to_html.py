# Python program to convert 
# CSV to HTML Table
 
import os
import pandas as pd

folder_path = "../../dev/benchmark/all-in-one"
csv_files = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".csv"):
        csv_files.append(file_path)

a = pd.read_csv(csv_files[0], index_col=0).to_html(csv_files[0].split("/")[-1].split(".")[0]+".html")