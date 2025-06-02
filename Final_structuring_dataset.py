#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[26]:


geslin_data = pd.read_excel('geslinactualitzat.xlsx')
geslin_data['Fecha'] = pd.to_datetime(geslin_data['Fecha'], format='%d-%m-%Y', errors='coerce')
geslin_data = geslin_data.dropna(subset=['Fecha'])
geslin_data = geslin_data[(geslin_data['Fecha'] >= '2024-01-22') & (geslin_data['Fecha'] <= '2025-02-21')]
geslin_data = geslin_data.sort_values('Fecha')


# In[27]:


# Load and Merge Pizarra Files
files = ['pizarra.xlsx', 'pizarra1.xlsx', 'pizarra17062102.xlsx']
pizarra_data = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
pizarra_data['Data'] = pd.to_datetime(pizarra_data['Data'], dayfirst=True, errors='coerce')
pizarra_data = pizarra_data.dropna(subset=['Data'])
pizarra_data = pizarra_data[(pizarra_data['Data'] >= '2024-01-22') & (pizarra_data['Data'] <= '2025-02-21')]


# In[28]:


# Clean Geslin NHC Columns
nhc_columns = ['NHC_AV', 'NHC_AFO', 'NHC_AFP', 'NHC_IA', 'NHC_IS', 'NHC_IF', 'NHC_PF', 'NHC_F']
geslin_data[nhc_columns] = geslin_data[nhc_columns].replace(r'^\s*$', np.nan, regex=True)
geslin_filtered = geslin_data[['Fecha', 'Turno'] + nhc_columns].copy()
geslin_filtered = geslin_filtered[
    geslin_filtered[nhc_columns].apply(lambda row: row.apply(lambda x: isinstance(x, (int, float))).any(), axis=1)
]
geslin_filtered = geslin_filtered.drop_duplicates(subset=['Fecha', 'Turno'], keep='first')


# In[29]:


# Standardize and Restructure
structured_rows = []
seen = set()
for _, row in geslin_filtered.iterrows():
    fecha, turno = row['Fecha'], row['Turno']
    for col in nhc_columns:
        nhc = row[col]
        if pd.notna(nhc):
            key = (nhc, fecha, turno) 
            if key not in seen:
                seen.add(key)
                structured_rows.append({
                    'NHC': str(int(nhc)), 'Fecha': fecha, 'Turno': turno,
                    **{c: int(c == col) for c in nhc_columns}
                })

nhc_data = pd.DataFrame(structured_rows)
nhc_data = nhc_data.sort_values(by=['Fecha', 'NHC']).reset_index(drop=True)


# In[30]:


# Clean and Assign Turno in Pizarra
pizarra_data['Pacient'] = pd.to_numeric(pizarra_data['Pacient'], errors='coerce')
pizarra_data = pizarra_data.dropna(subset=['Pacient'])
pizarra_data['Pacient'] = pizarra_data['Pacient'].astype(int).astype(str)
pizarra_data['Fecha'] = pizarra_data['Data'].dt.date
pizarra_data['Hora'] = pizarra_data['Data'].dt.hour

def classify_turno(hour):
    if pd.isna(hour): return 'Unknown'
    if 8 <= hour < 15: return 'M'
    elif 15 <= hour < 22: return 'T'
    return 'N'

pizarra_data['Turno'] = pizarra_data['Hora'].apply(classify_turno)
pizarra_data['Fecha'] = pd.to_datetime(pizarra_data['Fecha'])
pizarra_clean = pizarra_data.drop(columns=['Hora', 'Data'])
pizarra_clean = pizarra_clean.rename(columns={'Pacient': 'NHC'})


# In[31]:


# Merge Turno and Items from Pizarra
nhc_data['NHC'] = nhc_data['NHC'].astype(str)
pizarra_clean['NHC'] = pizarra_clean['NHC'].astype(str)
nhc_data = pd.merge(
    nhc_data,
    pizarra_clean[['NHC', 'Fecha', 'Turno', 'Ítem', 'Càrrega assistencial']],
    on=['NHC', 'Fecha', 'Turno'],
    how='left'
)


# In[32]:


# Add patients from pizarra that are not in geslin
nhc_in_geslin = set(nhc_data['NHC'].unique())
nhc_in_pizarra = set(pizarra_clean['NHC'].unique())
missing_nhcs = nhc_in_pizarra - nhc_in_geslin
missing_rows = pizarra_clean[pizarra_clean['NHC'].isin(missing_nhcs)].copy()

for col in nhc_data.columns:
    if col not in missing_rows.columns:
        if col in nhc_columns:
            missing_rows[col] = 0
        elif col in ['Ítem', 'Càrrega assistencial']:
            missing_rows[col] = ""
        else:
            missing_rows[col] = np.nan

missing_rows = missing_rows[nhc_data.columns]
nhc_data = pd.concat([nhc_data, missing_rows], ignore_index=True)


# In[33]:


# Keyword mapping (supporting strings and lists of synonyms/variants)
keywords = {
    'VIA': 'autolítiques',
    'VIPM': 'passives de mort',
    'AMG': [
        r'autolesions moderades/greus\*',
        r'autolesions.*moderades',
        r'autolesions.*greus',
        r'autolesions moderades',
        r'autolesions greus',
        r'autolesions(?!.*lleus)',
        r'autolesivas(?!.*lleus)'
    ],
    'ALT': 'autolesions lleus',
    'CAU': 'conductes autolítiques',
    'AOPAH': 'objecte',
    'AMED': 'medicació',
    'SRAMF': 'família',
    'OPCC': 'punts cecs',
    'PAP': 'preagitació',
    'MVE': 'manifestació verbal',
    'HVA': 'hipervigilància',
    'DCAT': 'alta durant el torn',
    'PIDT': 'dni',
    'NPMT': 'medicació',
    'AFPI': 'antecedents de fugida',
    'DCMSI': 'deteriorament cognitiu',
    'PII': 'ingrés involuntari',
    'CCA': 'canvi cognitiu',
    'SPS': 'suspicàcia',
    'CHP': 'heteroagresives cap a persones',
    'CHO': 'heteroagresives cap a objectes',
    'HVT': 'heteroagresivitat verbal',
    'CA': 'amenaçadora',
    'DR': 'recent*',
    'PH': 'hiperdemandant',
    'PD-ABVD': 'abvd',
    'PAH': 'higiene',
    'PCA': 'ambiental',
    'AC': 'contagi',
    'PTCA': 'tca',
    'PCM': 'mecànica',
    'RAC': 'acompanyament constant',
    'RAI': 'acompanyament intermitent',
    'RCE-PSA': 'somàtica',
    'RCO': 'oposicionistes'
}

# Classify acronyms into source columns
item_acronyms = list(keywords.keys())[:25]
careload_acronyms = list(keywords.keys())[25:]

# Normalize text fields to lowercase
nhc_data['Ítem'] = nhc_data['Ítem'].astype(str).str.lower()
nhc_data['Càrrega assistencial'] = nhc_data['Càrrega assistencial'].astype(str).str.lower()

# Pattern matching loop (handles lists, wildcards, etc.)
for acronym, keyword_patterns in keywords.items():
    if isinstance(keyword_patterns, str):
        keyword_patterns = [keyword_patterns]

    regex_parts = []
    for pattern in keyword_patterns:
        if pattern.endswith('*') and not pattern.endswith(r'\*'):
            # Treat as wildcard
            regex_parts.append(r'\b' + re.escape(pattern[:-1]) + r'\w*')
        else:
            # Use as-is (already escaped or raw regex)
            regex_parts.append(pattern)

    full_pattern = '|'.join(regex_parts)

    source_col = 'Ítem' if acronym in item_acronyms else 'Càrrega assistencial'
    nhc_data[acronym] = nhc_data[source_col].str.contains(full_pattern, regex=True, case=False, na=False).astype(int)


# In[34]:


# Risk Category Aggregation
nhc_data['Aggressive'] = nhc_data[['NHC_AV', 'NHC_AFO', 'NHC_AFP']].sum(axis=1)
nhc_data['Self-Harm'] = nhc_data[['NHC_IA', 'NHC_IS']].sum(axis=1)
nhc_data['Absconding'] = nhc_data[['NHC_IF', 'NHC_PF', 'NHC_F']].sum(axis=1)
nhc_data['No_risk'] = (nhc_data[['Aggressive', 'Self-Harm', 'Absconding']].sum(axis=1) == 0).astype(int)


# In[35]:


print("RISK CATEGORIES (Total Counts)")
print(nhc_data[['Aggressive', 'Self-Harm', 'Absconding', 'No_risk']].sum())


# In[36]:


print(nhc_data.columns.tolist())


# In[37]:


risk_categories = ['Aggressive', 'Self-Harm', 'Absconding', 'No_risk']
print("\nRISK CATEGORIES (Total Counts):")
print(nhc_data[risk_categories].sum())

subcategories = {
    'Aggressive': ['NHC_AV', 'NHC_AFO', 'NHC_AFP'],
    'Self-Harm': ['NHC_IA', 'NHC_IS'],
    'Absconding': ['NHC_IF', 'NHC_PF', 'NHC_F']
}

print("\nSUBCATEGORY DISTRIBUTION:")
for category, cols in subcategories.items():
    print(f"\n{category}:")
    print(nhc_data[cols].sum())


# In[38]:


# Extract Sociodemographic Data from Geslin
col_map = {
    'P': 15, 'Y': 24, 'AH': 33,
    'P_cols': [17, 18, 19, 20, 21, 22],
    'Y_cols': [26, 27, 28, 29, 30, 31],
    'AH_cols': [35, 36, 37, 38, 39, 40]
}
sociodemographic_columns = [
    'Género', '¿Diagnóstico de esquizofrenia?', '¿Mayor de 35 años?',
    'Ingreso involuntario', 'Ingreso por la presencia de riesgo de suicidio',
    'Ingreso por riesgo de hacer daño a otros'
]
nhc_data[sociodemographic_columns] = np.nan

for idx, row in nhc_data.iterrows():
    nhc = row['NHC']
    for col_key, cols in zip(['P', 'Y', 'AH'], ['P_cols', 'Y_cols', 'AH_cols']):
        found = geslin_data[geslin_data.iloc[:, col_map[col_key]].astype(str) == nhc]
        if not found.empty:
            nhc_data.loc[idx, sociodemographic_columns] = found.iloc[0, col_map[cols]].values
            break


# In[39]:


# Reorder Columns and Export
base_cols = ['NHC', 'Fecha', 'Turno']
risk_cols = ['Aggressive', 'Self-Harm', 'Absconding', 'No_risk']
final_cols = base_cols + sociodemographic_columns + item_acronyms + careload_acronyms + risk_cols
nhc_data = nhc_data[final_cols].drop_duplicates(subset=['NHC', 'Fecha', 'Turno'], keep='first')


# In[40]:


# Export binary dataset
nhc_data.to_excel('final_updated_nhc_binary_complete.xlsx', index=False)


# In[41]:


# Define weights
item_weights = {
    'VIA': 2, 'VIPM': 1, 'AMG': 2, 'ALT': 1, 'CAU': 2, 'AOPAH': 2,
    'AMED': 2, 'SRAMF': 1, 'OPCC': 1, 'PAP': 2,
    'MVE': 2, 'HVA': 2, 'DCAT': 1, 'PIDT': 1, 'NPMT': 1, 'AFPI': 2,
    'DCMSI': 1, 'PII': 1, 'CCA': 1, 'SPS': 1,
    'CHP': 2, 'CHO': 2, 'HVT': 1, 'CA': 1, 'DR': 2,
    'PH': 1, 'PD-ABVD': 2, 'PAH': 1, 'PCA': 2, 'AC': 2,
    'PTCA': 1, 'PCM': 2, 'RAC': 2, 'RAI': 1, 'RCE-PSA': 1, 'RCO': 1
}


# In[42]:


# Create weighted version
df_weighted = nhc_data.copy()
for col, weight in item_weights.items():
    if col in df_weighted.columns:
        df_weighted[col] = df_weighted[col].apply(lambda x: weight if x == 1 else 0)


# In[43]:


# Export weighted dataset
df_weighted.to_excel("final_weighted_dataset.xlsx", index=False)


# In[44]:


# Complete shifts Geslin

# Extract existing shifts from nhc_data
existing_turnos = nhc_data[['Fecha', 'Turno']].drop_duplicates()

# Prepare the new DataFrame
new_dataset = pd.DataFrame(columns=['Fecha', 'Turno', 'Aggressive', 'Self-Harm', 'Absconding'])

# Fill the DataFrame with the existing shifts and NHCs
for date, turno in existing_turnos.values:
    # Filter the NHCs for this date and shift
    filtered = nhc_data[(nhc_data['Fecha'] == date) & (nhc_data['Turno'] == turno)]
    
    # Get the NHCs for each risk category, separated by commas
    aggressive_nhcs = ', '.join(filtered[filtered['Aggressive'] == 1]['NHC'].astype(str)) if not filtered[filtered['Aggressive'] == 1].empty else 'NA'
    selfharm_nhcs = ', '.join(filtered[filtered['Self-Harm'] == 1]['NHC'].astype(str)) if not filtered[filtered['Self-Harm'] == 1].empty else 'NA'
    absconding_nhcs = ', '.join(filtered[filtered['Absconding'] == 1]['NHC'].astype(str)) if not filtered[filtered['Absconding'] == 1].empty else 'NA'
    
    # Add the row to the new DataFrame
    new_dataset = pd.concat([new_dataset, pd.DataFrame([{
        'Fecha': date,
        'Turno': turno,
        'Aggressive': aggressive_nhcs,
        'Self-Harm': selfharm_nhcs,
        'Absconding': absconding_nhcs
    }])], ignore_index=True)

# Check geslin_data for missing shifts and add them if they exist
for date in existing_turnos['Fecha'].unique():
    # Get the shifts for that date in nhc_data
    existing_shifts = existing_turnos[existing_turnos['Fecha'] == date]['Turno'].tolist()
    
    # Get the unique shifts for that date in geslin_data, avoiding NaNs
    geslin_shifts = geslin_data[(geslin_data['Fecha'] == date) & (geslin_data['Turno'].notna())]['Turno'].unique()
    
    # Find which shifts are missing
    missing_shifts = list(set(geslin_shifts) - set(existing_shifts))
    
    # Create rows for the missing shifts
    for shift in missing_shifts:
        new_dataset = pd.concat([new_dataset, pd.DataFrame([{
            'Fecha': date,
            'Turno': shift,
            'Aggressive': 'NA',
            'Self-Harm': 'NA',
            'Absconding': 'NA'
        }])], ignore_index=True)

# Sort by date and shift for clarity
new_dataset.sort_values(by=['Fecha', 'Turno'], inplace=True)

# Save to Excel with a different name
new_dataset.to_excel('turnos_completos_unicos.xlsx', index=False)


# In[45]:


# Generate a pivot table to see which shifts are present for each date
pivot_table = new_dataset.pivot_table(index='Fecha', columns='Turno', aggfunc='size', fill_value=0)

# Get the total number of days in the data
total_days = len(pivot_table)

# Calculate the percentage of completion for each shift
percentages = {}
for turno in ['M', 'T', 'N']:
    completed_days = (pivot_table[turno] > 0).sum()
    percentage = (completed_days / total_days) * 100
    percentages[turno] = percentage

# Display the summary
print("Percentage of completion for each shift:")
for turno, percentage in percentages.items():
    print(f"{turno}: {percentage:.2f}%")


# In[46]:


# We add the columns Risk and No_Risk

# Replace 'NA' with NaN for easier checks
new_dataset.replace('NA', pd.NA, inplace=True)

# Define the logic for No_risk and Risk
def calculate_risk(row):
    # Check if all three are NA
    if pd.isna(row['Aggressive']) and pd.isna(row['Self-Harm']) and pd.isna(row['Absconding']):
        return 1, 0  # No_risk = 1, Risk = 0
    else:
        return 0, 1  # No_risk = 0, Risk = 1

# Apply the logic to the DataFrame
new_dataset[['No_risk', 'Risk']] = new_dataset.apply(calculate_risk, axis=1, result_type='expand')

# Replace back the NaNs with 'NA' for clarity
new_dataset.fillna('NA', inplace=True)

# Save the updated DataFrame
new_dataset.to_excel('turnos_completos_unicos_risk.xlsx', index=False)


# In[47]:


# Columns to extract
items_columns = [
    'VIA', 'VIPM', 'AMG', 'ALT', 'CAU', 'AOPAH', 'AMED', 'SRAMF', 'OPCC',
    'PAP', 'MVE', 'HVA', 'DCAT', 'PIDT', 'NPMT', 'AFPI', 'DCMSI', 'PII',
    'CCA', 'SPS', 'CHP', 'CHO', 'HVT', 'CA', 'DR', 'PH', 'PD-ABVD', 'PAH',
    'PCA', 'AC', 'PTCA', 'PCM', 'RAC', 'RAI', 'RCE-PSA', 'RCO'
]

# ===========================
# Generate the Binary Dataset
# ===========================
# Take only the turns that exist in nhc_data and the specified items
binary_dataset = nhc_data[['NHC', 'Fecha', 'Turno'] + items_columns].drop_duplicates()

# Fill missing NHC with 'NA'
binary_dataset['NHC'] = binary_dataset['NHC'].fillna('NA')

# Fill missing items with 0 (although it should not happen, just in case)
binary_dataset[items_columns] = binary_dataset[items_columns].fillna(0)

# Drop duplicates if any
binary_dataset.drop_duplicates(subset=['Fecha', 'Turno'], keep='first', inplace=True)

# Save to Excel
binary_dataset.to_excel('binary_items_dataset.xlsx', index=False)

# ===========================
# Generate the Weighted Dataset
# ===========================
# Take only the turns that exist in df_weighted and the specified items
weighted_dataset = df_weighted[['NHC', 'Fecha', 'Turno'] + items_columns].drop_duplicates()

# Fill missing NHC with 'NA'
weighted_dataset['NHC'] = weighted_dataset['NHC'].fillna('NA')

# Fill missing items with 0 (although it should not happen, just in case)
weighted_dataset[items_columns] = weighted_dataset[items_columns].fillna(0)

# Drop duplicates if any
weighted_dataset.drop_duplicates(subset=['Fecha', 'Turno'], keep='first', inplace=True)

# Save to Excel
weighted_dataset.to_excel('weighted_items_dataset.xlsx', index=False)


# In[48]:


# List of columns to extract
sociodemographic_columns = [
    'NHC', 'Género', '¿Diagnóstico de esquizofrenia?',
    '¿Mayor de 35 años?', 'Ingreso involuntario',
    'Ingreso por la presencia de riesgo de suicidio',
    'Ingreso por riesgo de hacer daño a otros'
]

# Extract only the necessary columns
sociodemographic_data = nhc_data[sociodemographic_columns]

# Drop duplicates to have only one row per NHC
sociodemographic_data = sociodemographic_data.drop_duplicates(subset=['NHC'], keep='first')

# Strip spaces and replace empty strings or spaces with NaN
for col in sociodemographic_columns[1:]:
    sociodemographic_data[col] = sociodemographic_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# Replace empty strings, spaces, and '-' with NaN
sociodemographic_data.replace('', pd.NA, inplace=True)
sociodemographic_data.replace(' ', pd.NA, inplace=True)
sociodemographic_data.replace('-', pd.NA, inplace=True)

# Remove rows where all sociodemographic columns are NaN
columns_to_check = sociodemographic_columns[1:]  # Skip NHC
sociodemographic_data.dropna(subset=columns_to_check, how='all', inplace=True)

# Save to Excel
sociodemographic_data.to_excel('sociodemographic_data.xlsx', index=False)

