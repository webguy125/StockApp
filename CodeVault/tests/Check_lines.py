import json
import os

file_path = "C:/StockApp/backend/data/lines_FF.json"

if not os.path.exists(file_path):
    print("File not found:", file_path)
else:
    with open(file_path, "r") as f:
        try:
            lines = json.load(f)
            print(f"{len(lines)} lines remaining")
        except json.JSONDecodeError:
            print("File is empty or corrupted.")