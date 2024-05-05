import pandas as pd

# Load the CSV file
csv_file = "val.csv"
df = pd.read_csv(csv_file)

# Remove rows containing the specified path
df = df[~df['Image Filename'].str.contains('/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-missing/images/')]



# Edit the filename path
df['Image Filename'] = df['Image Filename'].str.replace(
    '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-patients/images/',
    '/workspace/test-patients/images/'
)

# Edit the filename path
df['Image Filename'] = df['Image Filename'].str.replace(
    '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/train/train1/',
    '/workspace/train/train1/'
)

# Write back to the CSV file
df.to_csv(csv_file, index=False)
