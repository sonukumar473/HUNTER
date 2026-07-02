import pandas as pd

# Load the full dataset
print("📂 Loading dataset...")
df = pd.read_csv("fake_job_postings.csv")

# Check current size
print(f"Original size: {len(df)} rows")
print(f"Fake jobs: {df['fraudulent'].sum()} rows")
print(f"Real jobs: {len(df) - df['fraudulent'].sum()} rows")

# Keep ALL fake jobs + equal number of real jobs (Balanced dataset)
fake_jobs = df[df['fraudulent'] == 1]
real_jobs = df[df['fraudulent'] == 0].sample(n=len(fake_jobs), random_state=42)

# Combine them
balanced_df = pd.concat([fake_jobs, real_jobs])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the reduced file
balanced_df.to_csv("fake_job_postings_reduced.csv", index=False)

print(f"\n✅ New size: {len(balanced_df)} rows")
print(f"   Fake: {balanced_df['fraudulent'].sum()}")
print(f"   Real: {len(balanced_df) - balanced_df['fraudulent'].sum()}")
print("\n📁 Saved as: fake_job_postings_reduced.csv")