# %% [markdown]
# LOADING DATA INTO THE NOTEBOOK

# %%
import pandas as pd

# Load company information
company_info = pd.read_csv('company_info.csv')

# Load company stock details
company_stock_details = pd.read_csv('company_stock_details.csv')

# Display the first few rows to understand the structure
print(company_info.head())
print(company_stock_details.head())

# %% [markdown]
# CLEAN DATA

# %%
print(company_info.isnull().sum())
print(company_stock_details.isnull().sum())

# %% [markdown]
# company_stock_details.fillna(method='ffill', inplace=True)

# %%
print(company_info.isnull().sum())
print(company_stock_details.isnull().sum())

# %% [markdown]
# MERGE THE TWO PDF TOGETHER

# %%
# Perform an inner join to merge on 'Symbol' (this only keeps rows where there is a match)
merged_data = pd.merge(company_stock_details, company_info, on='Symbol', how='inner')

# Check the first few rows of the merged data
print(merged_data.head())

# Save the merged data into a new CSV file
merged_data.to_csv('merged_company_data.csv', index=False)

# %% [markdown]
# hi i need this update up
# 


