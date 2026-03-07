# To split raw data into demo and train

import pandas as pd

# 1. Load your original dataset
# Make sure the path matches where your data is actually stored
df = pd.read_csv('data/data.csv')

# 2. Create the Demo Dataset (5 Malignant, 5 Benign)
# We group by diagnosis so we get an equal number of both classes
# random_state=42 ensures you get the exact same 10 patients every time you run this
demo_df = df.groupby('diagnosis').sample(n=5, random_state=42)

# 3. Create the Training/Main Dataset
# This takes the original dataframe and drops the 10 rows we just set aside for the demo
main_df = df.drop(demo_df.index)

# 4. Save them to new CSV files
demo_df.to_csv('data/demo_data.csv', index=False)
main_df.to_csv('data/train_data.csv', index=False)

print(f"Original dataset size: {len(df)} patients")
print(f"Demo dataset size: {len(demo_df)} patients (5 M, 5 B)")
print(f"Main dataset size: {len(main_df)} patients")