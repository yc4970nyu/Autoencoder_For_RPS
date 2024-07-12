#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import os
import numpy as np
from datetime import datetime


# In[9]:


path = os.getcwd()
path


# In[10]:


home_path = os.path.expanduser('~')
path=os.path.join(home_path, 'Desktop')
path 


# In[12]:


try:
    import xlrd
except ImportError:
    import sys
    get_ipython().system('{sys.executable} -m pip install xlrd')

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
def fix_hour_end1(value):
    value = str(value)
    if '00:00:00' in value:
        date_part = value.split(' ')[0]
        time_part = value.split(' ')[1]
        date = pd.to_datetime(date_part, format='%Y-%m-%d')
        if time_part == '00:00:00':
            previous_date = date - pd.Timedelta(days=1)
            return previous_date.strftime('%Y-%m-%d') + ' 00:00:00'
    return value
#01/01/2018 01:00

def fix_hour_end2(value):
    value = str(value)
    if '24:00' in value:
        date_part = value.split(' ')[0]
        time_part = value.split(' ')[1]
        date = pd.to_datetime(date_part, format='%m/%d/%Y')
        if time_part == '24:00':
            return date.strftime('%m/%d/%Y') + ' 00:00'
    return value

dfs = []
dfs2= []

for year in range(2002, 2017):
    file_name = f'{year}Demand.xls'
    file_path = os.path.join(path, file_name)
    df = pd.read_excel(file_path)
    df['Year'] = year
    print(df.shape[0])
    dfs.append(df)
# Concatenate
combined_df1 = pd.concat(dfs, ignore_index=True)
combined_df1['Hour_End']=pd.to_datetime(combined_df1['Hour_End'])
combined_df1['Hour_End']=combined_df1['Hour_End'].apply(lambda x: x.replace(microsecond=0))
combined_df1['Hour_End'] = combined_df1['Hour_End'].apply(fix_hour_end1)
combined_df1['Hour_End']=pd.to_datetime(combined_df1['Hour_End'])
# Convert 'Hour_End' to datetime to filter out February 29
#combined_df1['Hour_End'] = pd.to_datetime(combined_df1['Hour_End'], errors='coerce')
# Filter out February 29th data for leap years
for year in range(2002, 2017):
    if is_leap_year(year):
        start_date = f'{year}-02-29'
        end_date = f'{year}-03-01'
        combined_df1 = combined_df1[~((combined_df1['Hour_End'] >= start_date) & (combined_df1['Hour_End'] < end_date))]


for year in range(2017, 2024):
    file_name = f'{year}Demand.xls'
    file_path = os.path.join(path, file_name)
    df = pd.read_excel(file_path)
    df['Year'] = year 
    print(df.shape[0])
    dfs2.append(df)

# Concatenate
combined_df2 = pd.concat(dfs2, ignore_index=True)
#combined_df2['Hour_End'] = combined_df2['Hour_End'].apply(fix_hour_end)
# Convert 'Hour_End' to datetime to filter out February 29
#combined_df2['Hour_End'] = pd.to_datetime(combined_df2['Hour_End'], format='%m/%d/%Y %H:%M', errors='coerce')
#combined_df2['Hour_End']=combined_df2['Hour_End'].apply(lambda x: x.replace(microsecond=0))
combined_df2['Hour_End'] = combined_df2['Hour_End'].apply(fix_hour_end2)
combined_df2['Hour_End'] = pd.to_datetime(combined_df2['Hour_End'], format='%m/%d/%Y %H:%M', errors='coerce')
for year in range(2017, 2024):
    if is_leap_year(year):
        start_date = f'{year}-02-29'
        end_date = f'{year}-03-01'
        combined_df2 = combined_df2[~((combined_df2['Hour_End'] >= start_date) & (combined_df2['Hour_End'] < end_date))]

#combined_df=pd.concat([combined_df1, combined_df2], ignore_index=True)
# The data has duplicate hours
combined_df2=combined_df2.dropna()


# ### Separate the date and hour to individual columns

# In[13]:


combined_df1=combined_df1[['Hour_End', 'ERCOT', 'Year']]
combined_df1['Date']=combined_df1['Hour_End'].dt.date
combined_df1['Hour']=(combined_df1['Hour_End'].dt.hour+(combined_df1['Hour_End'].dt.minute / 60)+ 
            (combined_df1['Hour_End'].dt.second / 3600)).round().astype(int)
#combined_df['Hour']=combined_df['Hour'].replace({0: 24})
#combined_df1=combined_df1.drop(columns=['Hour_End'])

combined_df2=combined_df2[['Hour_End', 'ERCOT', 'Year']]
combined_df2['Date']=combined_df2['Hour_End'].dt.date
combined_df2['Hour']=(combined_df2['Hour_End'].dt.hour+(combined_df2['Hour_End'].dt.minute / 60)).round().astype(int)
#combined_df['Hour']=combined_df['Hour'].replace({0: 24})
#combined_df2=combined_df2.drop(columns=['Hour_End'])

combined_df=pd.concat([combined_df1, combined_df2], ignore_index=True)
# Drop the original 'Hour_End' column
#filtered_df = filtered_df.drop(columns=['Hour_End'])


# In[14]:


null_rows=combined_df[combined_df.isnull().any(axis=1)]
null_rows


# In[15]:


mean_value=combined_df['ERCOT'].mean()
combined_df['ERCOT'][130079]=mean_value
combined_df=combined_df.dropna()


# In[16]:


combined_df.isnull().sum()


# In[17]:


combined_df


# In[ ]:





# In[ ]:





# ### Missing value: Imputation with Mean

# #### The missing values due to Winter-time and Summer-time Conversion

# In[18]:


all_dates = pd.date_range(start=combined_df['Date'].min(), end=combined_df['Date'].max(), freq='D')
all_dates = all_dates[(all_dates.month != 2) | (all_dates.day != 29)]
all_hours = pd.DataFrame({
    'Date': all_dates.repeat(24),
    'Hour': list(range(24)) * len(all_dates)
})


# In[19]:


all_hours


# In[20]:


all_hours.loc[all_hours['Hour'] == 0, 'Date'] = all_hours.loc[all_hours['Hour']==0, 'Date'] - pd.Timedelta(days=1)


# In[21]:


all_hours = all_hours[~((all_hours['Date'].dt.month == 2) & (all_hours['Date'].dt.day == 29))]


# In[22]:


all_hours


# In[23]:


combined_df['Date']=combined_df['Date'].astype(str)
combined_df['Hour']=combined_df['Hour'].astype(str)


# In[24]:


all_hours['Date']=all_hours['Date'].astype(str)
all_hours['Hour']=all_hours['Hour'].astype(str)


# In[ ]:





# In[25]:


all_same_dtype = combined_df['Date'].apply(type).nunique() == 1
print("All values in 'Date' column have the same data type:", all_same_dtype)
all_same_dtype = combined_df['Hour'].apply(type).nunique() == 1
print("All values in 'Hour' column have the same data type:", all_same_dtype)


# In[26]:


all_hours.isnull().sum()


# In[27]:


merged_df=pd.merge(all_hours, combined_df, how='left', on=['Date', 'Hour'])


# In[28]:


missing_hours = merged_df[merged_df['Hour_End'].isnull()]
print(missing_hours[['Date', 'Hour']])


# In[29]:


mean_ercot = combined_df['ERCOT'].mean()

# Create a DataFrame for the missing hours
missing_hours_df = missing_hours[['Date', 'Hour']].copy()


# In[ ]:





# In[30]:


missing_hours_df['Date']=pd.to_datetime(missing_hours_df['Date'], format='%Y-%m-%d', errors='coerce')


# In[31]:


missing_hours_df['Hour']=missing_hours_df['Hour'].astype(int)


# In[32]:


missing_hours_df['Hour_End']=missing_hours_df.apply(lambda row: pd.Timestamp(row['Date'])+pd.Timedelta(hours=row['Hour']), axis=1)


# In[33]:


missing_hours_df['ERCOT'] = mean_ercot


# In[34]:


missing_hours_df['Year'] = missing_hours_df['Date'].dt.year


# In[35]:


missing_hours_df.shape


# In[36]:


missing_hours_df = missing_hours_df.drop(missing_hours_df.index[0])


# In[37]:


combined_df['Date'] = pd.to_datetime(combined_df['Date'],errors='coerce')


# In[38]:


combined_df['Hour'] = combined_df['Hour'].astype(int)


# In[39]:


imputation_count = 0

for _, row in missing_hours_df.iterrows():
    date_hour = row['Hour_End']
    date = row['Date']
    hour = row['Hour']
    
    if not ((combined_df['Date'] == date) & (combined_df['Hour'] == hour)).any():
        insertion_point = combined_df[(combined_df['Date'] == date) & (combined_df['Hour'] < hour)].index.max() + 1
        combined_df = pd.concat([combined_df.iloc[:insertion_point], pd.DataFrame([row]), combined_df.iloc[insertion_point:]]).reset_index(drop=True)
        imputation_count += 1

# Sort by Date and Hour to maintain order
combined_df = combined_df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)

# Display the final DataFrame and the number of imputations
print(combined_df)
print(f"Number of imputations: {imputation_count}")


# In[40]:


combined_df


# In[41]:


no_consecutive_same_hours = not any(combined_df['Hour'].iloc[i] == combined_df['Hour'].iloc[i + 1] for i in range(len(combined_df) - 1))
no_consecutive_same_hours


# In[42]:


same_consecutive_hours = combined_df[(combined_df['Hour'].shift() == combined_df['Hour']) & (combined_df['Date'].shift() == combined_df['Date'])]


# In[43]:


combined_df = combined_df.drop_duplicates(subset=['Date', 'Hour'], keep='first')


# In[44]:


combined_df


# In[45]:


combined_df = combined_df[~((combined_df['Date'].dt.month == 2) & (combined_df['Date'].dt.day == 29))]


# In[46]:


checker=combined_df.groupby('Year').size()
checker


# In[47]:


is_hour_end_unique = combined_df['Hour_End'].is_unique


# In[48]:


is_hour_end_unique


# In[49]:


is_date_hour_unique = combined_df.duplicated(subset=['Date', 'Hour']).sum() == 0


# In[50]:


is_date_hour_unique


# In[51]:


def check_hour_pattern(df):
    grouped = df.groupby('Date')
    for name, group in grouped:
        if not (group['Hour'].values == list(range(24))).all():
            return False
    return True

# Verify the pattern
pattern_is_correct = check_hour_pattern(combined_df)


# In[52]:


pattern_is_correct


# ### Scaling and Normalization

# In[53]:


annual_demand_2021 = combined_df[combined_df['Year']==2021]['ERCOT'].sum()
annual_demand_per_year = combined_df.groupby('Year')['ERCOT'].sum()


# In[54]:


# Scaling factor for each year to match 2021
scaling_factors = annual_demand_2021/annual_demand_per_year


# In[55]:


combined_df['Scaled_ERCOT'] = combined_df.apply(lambda row: row['ERCOT'] * scaling_factors[row['Year']], axis=1)


# In[ ]:





# In[56]:


# Step 2: Normalize the demand across all years based on peak demand in that year
peak_demand_per_year = combined_df.groupby('Year')['Scaled_ERCOT'].max()
# Apply normalization
combined_df['Normalized_ERCOT'] = combined_df.apply(lambda row: row['Scaled_ERCOT'] / peak_demand_per_year[row['Year']], axis=1)


# In[57]:


combined_df


# In[58]:


combined_df.isnull().sum()


# In[59]:


demand_df=combined_df[['Hour_End','Date','Hour','Normalized_ERCOT']]


# In[60]:


demand_df = demand_df[(demand_df['Date'].dt.year >= 2002) & (demand_df['Date'].dt.year <= 2021)]


# In[61]:


demand_df


# In[62]:


reshaped_df = demand_df.pivot(index='Date', columns='Hour', values='Normalized_ERCOT')
reshaped_df.columns = [f'Hour_{i}' for i in reshaped_df.columns]


# In[63]:


reshaped_df


# In[ ]:





# In[64]:


# Convert to string to avoid dtype mismatch
#combined_df['Date'] = combined_df['Date'].astype(str)
#missing_hours_df['Date'] = missing_hours_df['Date'].astype(str)


# ### Solar PV & Wind Data

# In[242]:


import pandas as pd
import os
import numpy as np
from datetime import datetime


# In[243]:


path = os.getcwd()
home_path = os.path.expanduser('~')
path=os.path.join(home_path, 'Desktop')
#/Solar/ERCOT_Data


# In[244]:


Solar00_09 = pd.read_excel(path + "/ERCOT-OperationalPlanned-SolarPVProfiles-2000-2009-CST-CDT.xlsx")
Solar10_19 = pd.read_excel(path + "/ERCOT-OperationalPlanned-SolarPVProfiles-2010-2019-CST-CDT.xlsx")
Solar20_21 = pd.read_excel(path + "/ERCOT-OperationalPlanned-SolarPVProfiles-2020-2021-CST-CDT.xlsx")


# In[245]:


# fucntion that gets the aggregated hourly capacity factors of operational and planned resources

def process_solar_data(df, info_rows):
    df_info = df[:info_rows]
    df_output = df[info_rows+1:]
    df_output.columns = df.iloc[info_rows]
    df_output["Date"]=[datetime.strptime(str(date), '%Y%m%d') for date in df_output.DATE]
    
    # Find the column indices of operational and planned projects
    operational_projects = [i for i, value in enumerate(df_info.iloc[5]) if "Operational" in str(value)]
    planned_projects = [i for i, value in enumerate(df_info.iloc[5]) if "Planned" in str(value) or "Part 1" in str(value) or "Part 2" in str(value)]
    
    # Calculate the total operational and planned projects output
    df_output['Total_operational_project_output'] = df_output.iloc[:, operational_projects].sum(axis=1)
    df_output['Total_planned_project_output'] = df_output.iloc[:, planned_projects].sum(axis=1)
    
    # Calculate the aggregated operational and planned capacity factors
    df_output["Aggregated Operational CF (%)"] = df_output['Total_operational_project_output'] / df_info.iloc[0][operational_projects].sum()
    df_output["Aggregated Planned CF (%)"] = df_output['Total_planned_project_output'] / df_info.iloc[0][planned_projects].sum()
    
    return df_output

Solar00_09_output = process_solar_data(Solar00_09, 6)
Solar10_19_output = process_solar_data(Solar10_19, 6)
Solar20_21_output = process_solar_data(Solar20_21, 6)


# In[246]:


Solar00_09_output['Date'] = pd.to_datetime(Solar00_09_output['DATE'], format='%Y%m%d')
Solar00_09_output['Hour'] = Solar00_09_output['TIME'].astype(int) // 100
# Create 'Hour_End' by combining 'Date' and 'Hour'
Solar00_09_output['Hour_End'] = Solar00_09_output.apply(lambda row: pd.Timestamp(row['Date']) + pd.Timedelta(hours=row['Hour']), axis=1)

Solar00_09_output = Solar00_09_output[['Date', 'Hour', 'Hour_End', 'Aggregated Operational CF (%)', 'Aggregated Planned CF (%)']]
Solar00_09_output


# In[247]:


def process_solar_output(df):
    # Convert 'DATE' to datetime
    df['Date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    # Extract the hour from 'TIME'
    df['Hour'] = df['TIME'].astype(int) // 100
    df['Hour_End'] = df.apply(lambda row: pd.Timestamp(row['Date']) + pd.Timedelta(hours=row['Hour']), axis=1)
    result_df = df[['Date', 'Hour', 'Hour_End', 'Aggregated Operational CF (%)', 'Aggregated Planned CF (%)']]
    
    return result_df


# In[248]:


Solar10_19_output=process_solar_output(Solar10_19_output)
Solar20_21_output=process_solar_output(Solar20_21_output)


# In[249]:


solar_df = pd.concat([Solar00_09_output, Solar10_19_output, Solar20_21_output], ignore_index=True)

solar_df = solar_df[(solar_df['Date'].dt.year >= 2002) & (solar_df['Date'].dt.year <= 2023)]

# Delete all the Feb 29th data
solar_df = solar_df[~((solar_df['Date'].dt.month == 2) & (solar_df['Date'].dt.day == 29))]


# In[250]:


solar_df.reset_index(drop=True)


# #### Wind Data

# In[251]:


Wind00_09 = pd.read_csv(path + "/ERCOT_WindProfiles_Operational-Planned_2000-2009_CST.csv")
Wind10_19 = pd.read_excel(path + "/ERCOT-OperationalPlanned-WindProfiles-2010-2019-CST-CDT.xlsx")
Wind20_21 = pd.read_excel(path + "/ERCOT-OperationalPlanned-WindProfiles-2020-2021-CST-CDT.xlsx")


# In[252]:


Wind00_09


# In[253]:


import pandas as pd
import re

def process_wind_csv(df):
    # Convert 'DATE' to datetime
    df['Date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['Hour'] = df['TIME'].astype(int) // 100
    df['Hour_End'] = df.apply(lambda row: pd.Timestamp(row['Date']) + pd.Timedelta(hours=row['Hour']), axis=1)
    capacity_columns = [col for col in df.columns if re.match(r'SITE_\d+:capacity=\d+(\.\d+)?', col)]
    capacities = [float(re.search(r'capacity=(\d+(\.\d+)?)', col).group(1)) for col in capacity_columns]
    df['Total_operational_project_output'] = df[capacity_columns].sum(axis=1)
    total_capacity = sum(capacities)
    df["Aggregated Operational CF (%)"] = df['Total_operational_project_output'] / total_capacity
    result_df = df[['Date', 'Hour', 'Hour_End', 'Total_operational_project_output', 'Aggregated Operational CF (%)']]
    return result_df

Wind00_09_output=process_wind_csv(Wind00_09)


# In[254]:


def process_wind_data(df, info_rows):
    df_info = df[:info_rows]
    df_output = df[info_rows + 1:]
    df_output.columns = df.iloc[info_rows]
    df_output["Date"] = [datetime.strptime(str(date), '%Y%m%d') for date in df_output.DATE]
    
    # Find the column indices of operational and planned projects
    operational_projects = [i for i, value in enumerate(df_info.iloc[5]) if "Operational" in str(value)]
    planned_projects = [i for i, value in enumerate(df_info.iloc[5]) if "Planned" in str(value) or "Part 1" in str(value) or "Part 2" in str(value)]
    
    # Calculate the total operational and planned projects output
    df_output['Total_operational_project_output'] = df_output.iloc[:, operational_projects].sum(axis=1)
    df_output['Total_planned_project_output'] = df_output.iloc[:, planned_projects].sum(axis=1)
    
    # Calculate the aggregated operational and planned capacity factors
    df_output["Aggregated Operational CF (%)"] = df_output['Total_operational_project_output'] / df_info.iloc[0][operational_projects].sum()
    df_output["Aggregated Planned CF (%)"] = df_output['Total_planned_project_output'] / df_info.iloc[0][planned_projects].sum()
    
    return df_output

Wind10_19_output = process_wind_data(Wind10_19, 6)
Wind20_21_output = process_wind_data(Wind20_21, 6)


# In[255]:


def process_wind_output(df):
    # Convert 'DATE' to datetime
    df['Date'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    # Extract the hour from 'TIME'
    df['Hour'] = df['TIME'].astype(int) // 100
    df['Hour_End'] = df.apply(lambda row: pd.Timestamp(row['Date']) + pd.Timedelta(hours=row['Hour']), axis=1)
    result_df = df[['Date', 'Hour', 'Hour_End', 'Aggregated Operational CF (%)', 'Aggregated Planned CF (%)']]
    
    return result_df
Wind10_19_output=process_wind_output(Wind10_19_output)
Wind20_21_output=process_wind_output(Wind20_21_output)


# In[256]:


Wind10_19_output


# In[257]:


Wind00_09_output


# In[258]:


Wind10_19_output=Wind10_19_output.drop(columns=['Aggregated Planned CF (%)'])
Wind20_21_output=Wind20_21_output.drop(columns=['Aggregated Planned CF (%)'])
wind_df = pd.concat([Wind00_09_output, Wind10_19_output, Wind20_21_output], ignore_index=True)
wind_df = wind_df[(wind_df['Date'].dt.year >= 2002) & (wind_df['Date'].dt.year <= 2023)]
wind_df = wind_df[~((wind_df['Date'].dt.month == 2) & (wind_df['Date'].dt.day == 29))]


# In[259]:


wind_df=wind_df.drop(columns=['Total_operational_project_output'])
wind_df=wind_df.drop(wind_df.index[-1])


# In[260]:


wind_df


# In[261]:


wind_df


# In[262]:


solar_df


# ### Merging the 2002-2021 Solar PV and Wind Data with demand_df

# #### Before the merge, performing imputation for the missing values due to Winter time and Summer time conversion, and deleteing the duplicates

# In[263]:


date_range = pd.date_range(start='2002-01-01', end='2021-12-31', freq='H')
date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]


# In[264]:


# Create a reference DataFrame with all possible Date and Hour combinations
reference_df = pd.DataFrame({'Hour_End': date_range})

# Ensure Hour_End columns are in datetime format
wind_df['Hour_End'] = pd.to_datetime(wind_df['Hour_End'])
solar_df['Hour_End'] = pd.to_datetime(solar_df['Hour_End'])

# Filter out data outside the 2002-2021 range and exclude Feb 29
wind_df = wind_df[(wind_df['Hour_End'].dt.year >= 2002) & (wind_df['Hour_End'].dt.year <= 2021)]
wind_df = wind_df[~((wind_df['Hour_End'].dt.month == 2) & (wind_df['Hour_End'].dt.day == 29))]

solar_df = solar_df[(solar_df['Hour_End'].dt.year >= 2002) & (solar_df['Hour_End'].dt.year <= 2021)]
solar_df = solar_df[~((solar_df['Hour_End'].dt.month == 2) & (solar_df['Hour_End'].dt.day == 29))]

# Identify missing hours for wind_df
missing_hours_wind = reference_df[~reference_df['Hour_End'].isin(wind_df['Hour_End'])]

# Identify missing hours for solar_df
missing_hours_solar = reference_df[~reference_df['Hour_End'].isin(solar_df['Hour_End'])]

# Function to impute missing hours with 0 values
def impute_missing_hours(df, missing_hours, column_prefix):
    missing_hours = missing_hours.copy()
    missing_hours['Date'] = missing_hours['Hour_End'].dt.date
    missing_hours['Hour'] = missing_hours['Hour_End'].dt.hour
    missing_hours['Aggregated Operational CF (%)']=0
    if column_prefix == 'Solar':
        missing_hours['Aggregated Planned CF (%)']=0
    return pd.concat([df, missing_hours], ignore_index=True).sort_values(by='Hour_End').reset_index(drop=True)

# Impute missing hours in wind_df and solar_df
wind_df = impute_missing_hours(wind_df, missing_hours_wind, 'Wind')
solar_df = impute_missing_hours(solar_df, missing_hours_solar, 'Solar')


# In[265]:


# Merge the DataFrames on 'Hour_End'
final_df = pd.merge(demand_df, wind_df, on='Hour_End', how='left')
final_df = pd.merge(final_df, solar_df, on='Hour_End', how='left')

# Ensure there are no February 29 data in final_df
final_df = final_df[~((final_df['Hour_End'].dt.month == 2) & (final_df['Hour_End'].dt.day == 29))]

# Check for null values in the final DataFrame
null_rows = final_df[final_df.isnull().any(axis=1)]

if not null_rows.empty:
    print("Rows with null values and their Hour_End values:")
    print(null_rows[['Hour_End']])
else:
    print("There are no null values in the final DataFrame.")


# In[266]:


wind_df = wind_df.drop_duplicates(subset=['Hour_End'], keep='first').reset_index(drop=True)
solar_df = solar_df.drop_duplicates(subset=['Hour_End'], keep='first').reset_index(drop=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[270]:


import pandas as pd

wind_df = wind_df.rename(columns={"Aggregated Operational CF (%)": "Wind Aggregated Operational CF (%)"})
solar_df = solar_df.rename(columns={"Aggregated Operational CF (%)": "Solar Aggregated Operational CF (%)",
                                    "Aggregated Planned CF (%)": "Solar Planned CF (%)"})

solar_df['Date'] = pd.to_datetime(solar_df['Date'])
wind_df['Date'] = pd.to_datetime(wind_df['Date'])

# Merge the DataFrames on 'Date' and 'Hour'
solar_wind_df = pd.merge(solar_df, wind_df, on=['Date', 'Hour'])

# Select and reorder columns as needed
solar_wind_df = solar_wind_df[['Date', 'Hour', 'Hour_End_x', 'Hour_End_y', 
                       'Wind Aggregated Operational CF (%)', 
                       'Solar Aggregated Operational CF (%)', 
                       'Solar Planned CF (%)']]

# Optionally, you can drop one of the 'Hour_End' columns if they are identical
solar_wind_df = solar_wind_df.drop(columns=['Hour_End_y']).rename(columns={'Hour_End_x': 'Hour_End'})


# In[271]:


solar_wind_df.isnull().sum()


# In[272]:


feb_29_data = solar_wind_df[(solar_wind_df['Date'].dt.month == 2) & (solar_wind_df['Date'].dt.day == 29)]

# Check if there are any rows with February 29th data
if not feb_29_data.empty:
    print("The merged_df contains February 29th data.")
    print(feb_29_data)
else:
    print("The merged_df does not contain any February 29th data.")


# In[273]:


demand_df['Hour_End'] = pd.to_datetime(demand_df['Hour_End'])
demand_df['Hour_End'] = demand_df['Hour_End'].dt.round('H')
solar_wind_df['Hour_End'] = pd.to_datetime(solar_wind_df['Hour_End'])
solar_wind_df['Hour_End'] = solar_wind_df['Hour_End'].dt.round('H')


# In[274]:


final_df = pd.merge(demand_df, solar_wind_df, on='Hour_End', how='left')


# In[275]:


final_df.isnull().sum()


# In[276]:


final_df = final_df[['Hour_End','Date_x', 'Hour_x', 
                        'Normalized_ERCOT',
                       'Solar Aggregated Operational CF (%)', 
                       'Solar Planned CF (%)',
                       'Wind Aggregated Operational CF (%)']]

final_df=final_df.rename(columns={"Date_x": "Date",
                                  "Hour_x": "Hour"})


# In[277]:


final_df


# In[278]:


final_df['Date'] = final_df['Hour_End'].dt.date

# Pivot the DataFrame to get the desired format
reshaped_final_df=final_df.pivot_table(
    index='Date',
    columns='Hour',
    values=['Normalized_ERCOT', 'Solar Aggregated Operational CF (%)', 'Solar Planned CF (%)', 'Wind Aggregated Operational CF (%)'],
    aggfunc='first'  # Use 'first' to handle duplicate hours, if any
)

# Flatten the multi-level columns
reshaped_final_df.columns = [f'{col[1]}_{col[0]}' for col in reshaped_final_df.columns]

# Ensure the DataFrame is sorted by Date
reshaped_final_df=reshaped_final_df.sort_index()


# In[279]:


reshaped_final_df


# In[280]:


# Check the shape of the reshaped DataFrame
print(f'The reshaped DataFrame has {reshaped_final_df.shape[0]} rows and {reshaped_final_df.shape[1]} columns.')


# In[281]:


missing_final_df = reference_df[~reference_df['Hour_End'].isin(final_df['Hour_End'])]


# In[282]:


missing_final_df 


# In[284]:


reshaped_final_df.isnull().sum().sum()


# In[288]:


save_to = '/Users/louis/Desktop/reshaped_final_df.csv'
reshaped_final_df.to_csv(save_to, index=False)


# ## K-Means Clustering Analysis

# In[285]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import numpy as np

#scaler = StandardScaler()
#reshaped_final_df= scaler.fit_transform(reshaped_final_df)

# Determine the optimal number of clusters using the Silhouette method
silhouette_scores = []
range_n_clusters = range(2, 11)  # Testing cluster numbers from 2 to 10

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reshaped_final_df)
    silhouette_avg = silhouette_score(reshaped_final_df, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f'The optimal number of clusters is: {optimal_n_clusters}')

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(reshaped_final_df)
centroids = kmeans.cluster_centers_

# Find the data points closest to the centroids
closest, _ = pairwise_distances_argmin_min(centroids, reshaped_final_df)

# Create a DataFrame to store the results
results_df = reshaped_final_df.copy()
results_df['Cluster'] = cluster_labels

# Find the vectors closest to the centroids
closest_points = results_df.iloc[closest]

# Display the results
print("The vectors closest to the centroids are:")
print(closest_points)

# Plotting silhouette scores
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.show()


# In[286]:


closest_points


# ### Type 1 RPS Autoencoder

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# Assuming reshaped_final_df is your dataframe
data = reshaped_final_df.values

# Define the autoencoder model
input_dim = data.shape[1]
encoding_dim = 25  # Dimension of the latent space

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder in smaller batches
batch_size = 64  # Smaller batch size
epochs = 10  # Fewer epochs
num_batches = data.shape[0] // batch_size

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_data = data[start_idx:end_idx]
        if batch_data.shape[0] > 0:
            autoencoder.train_on_batch(batch_data, batch_data)

# Encoder model to get the encoded data
encoder = Model(input_layer, encoded)
encoded_data = encoder.predict(data)

# Convert encoded data back to a DataFrame
encoded_df = pd.DataFrame(encoded_data)

print(encoded_df.head())


# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

data = reshaped_final_df.values
# Define the autoencoder model
input_dim = data.shape[1]
encoding_dim = 25  # k for the latent space
# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)
# Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# Train the autoencoder
autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
# Encoder model to get the encoded data
encoder = Model(input_layer, encoded)
encoded_data = encoder.predict(data)
# Apply K-means clustering on the encoded data
n_clusters = 10  # Changable accordingly
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(encoded_data)
centroids = kmeans.cluster_centers_
# Find the vectors closest to the centroids
closest, _ = pairwise_distances_argmin_min(centroids, encoded_data)
# Create a DataFrame to store the results
results_df = reshaped_final_df.copy()
results_df['Cluster'] = cluster_labels
# Find the vectors closest to the centroids
closest_points = results_df.iloc[closest]

# Convert encoded data back to a DataFrame for further use
encoded_df = pd.DataFrame(encoded_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




