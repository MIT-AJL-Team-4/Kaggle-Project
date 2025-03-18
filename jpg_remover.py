import pandas as pd

# Load CSV
df = pd.read_csv('submission.csv')

# Remove ".jpg" from md5hash column
df['md5hash'] = df['md5hash'].str.replace('.jpg', '', regex=False)

# Save the fixed CSV
df.to_csv('submission_fixed.csv', index=False)

print("Fixed CSV saved as submission_fixed.csv")
