import pandas as pd
import numpy as np
import os

# Load the original dataset
df = pd.read_csv("./qdataset/custom/qd-generation/all_tasks.csv")

# 🔴 Filtrer les circuits avec width <= 127
df = df[df["original_width"] <= 127]

# Vérification (optionnelle)
if len(df) == 0:
    raise ValueError("Aucun circuit ne respecte la condition width <= 127")

# Function to create a synthetic sub-dataset
def create_sub_dataset(df, target_depth, tolerance, num_circuits):
    subset = pd.DataFrame()
    current_depth = 0

    while (
        current_depth < target_depth - tolerance
        or current_depth > target_depth + tolerance
    ):
        # 🔴 replace=True pour éviter erreur si dataset petit
        subset = df.sample(n=num_circuits, replace=True)
        current_depth = subset["original_depth"].sum()

    return subset, current_depth


# Adjustable parameters
num_subsets = 1000   # Number of subsets to create
num_circuits = 26   # Number of circuits in each subset

average_depth = (
    df["original_depth"].sum() // len(df) * num_circuits
)

tolerance = average_depth * 0.1  # 10% tolerance

# Generate synthetic sub-datasets
all_subsets = []

for i in range(num_subsets):
    subset, current_depth = create_sub_dataset(
        df, average_depth, tolerance, num_circuits
    )

    subset["subset"] = i + 1
    all_subsets.append(subset)

    print(
        f"Generated subset {i+1}/{num_subsets} | Avg depth: {current_depth/num_circuits:.2f}"
    )

# Combine all subsets
final_df = pd.concat(all_subsets, ignore_index=True)

# Move 'subset' column to first position
cols = ["subset"] + [col for col in final_df.columns if col != "subset"]
final_df = final_df[cols]

# Save file
output_path = "./qdataset/custom/datasets"
os.makedirs(output_path, exist_ok=True)

file_name = f"qdataset_{num_subsets}_sub_{num_circuits}.csv"

final_df.to_csv(
    os.path.join(output_path, file_name),
    index=False,
)

print(f"\nDataset saved successfully: {file_name}")