from dsm.datasets import load_dataset as load_dsm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from pycox import datasets
import pandas as pd
import numpy as np

EPS = 1e-8

def load_dataset(dataset='SUPPORT', path = './', normalize = True, **kwargs):
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis = 'columns')
    elif dataset == 'SEER':
        df = pd.read_csv(path + 'data/export.csv')
        df = process_seer(df)
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    else:
        return load_dsm(dataset, normalize = normalize, **kwargs)

    covariates = df.drop(['duration', 'event'], axis = 'columns')
    return StandardScaler().fit_transform(covariates.values).astype(float) if normalize else covariates.values.astype(float),\
           df['duration'].values.astype(float),\
           df['event'].values.astype(int),\
           covariates.columns

# def process_seer(df):
#     # Remove multiple visits
#     # df = df.groupby('Patient ID').first()

#     # Encode using dictionary to remove missing data
#     df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace = True)
#     df["Sequence number"].replace(['88', '99'], np.nan, inplace = True)
#     df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
#     df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
#     df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

#     # Remove patients without survival time
#     df = df[~df.duration.isna()]

#     # Outcome 
#     df['duration'] = df['duration'].astype(float)
#     df['event'] = df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)" # Death 
#     df['event'].loc[(df["COD to site recode"] == "Diseases of Heart") & (df["SEER cause-specific death classification"] == "Alive or dead of other cause")] = 2 # CVD 

#     df = df.drop(columns = ["COD to site recode"])

#     # Imput and encode categorical
#     ## Categorical
#     categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
#         "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
#         "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
#         "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "RX Summ--Surg Prim Site (1998+)",
#         "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]
#     ordinal_col = ["Age recode with <1 year olds", "Grade", "Year of diagnosis"]

#     imputer = SimpleImputer(strategy='most_frequent')
#     enc = OrdinalEncoder()
#     # Only keep categorical columns that are present in df
#     categorical_col = [col for col in categorical_col if col in df.columns]

#     # Now encode
#     df_cat = pd.DataFrame(
#         enc.fit_transform(imputer.fit_transform(df[categorical_col])),
#         columns=categorical_col,
#         index=df.index
#     )

    
#     df_ord = pd.DataFrame(imputer.fit_transform(df[ordinal_col]), columns = ordinal_col, index = df.index)
#     # Replace age groups and grade labels with ordinal integers
#     df_ord["Age recode with <1 year olds"] = df_ord["Age recode with <1 year olds"].replace({
#         '00 years': 0,
#         '01-04 years': 1,
#         '05-09 years': 2,
#         '10-14 years': 3,
#         '15-19 years': 4,
#         '20-24 years': 5,
#         '25-29 years': 6,
#         '30-34 years': 7,
#         '35-39 years': 8,
#         '40-44 years': 9,
#         '45-49 years': 10,
#         '50-54 years': 11,
#         '55-59 years': 12,
#         '60-64 years': 13,
#         '65-69 years': 14,
#         '70-74 years': 15,
#         '75-79 years': 16,
#         '80-84 years': 17,
#         '85-89 years': 18,
#         '90-94 years': 19,
#         '95-99 years': 20,
#         '90+ years': 19, 
#         '100+ years': 21,
#         'Unknown age': np.nan})

#     df_ord["Grade"] = df_ord["Grade"].replace({
#         'Well differentiated; Grade I': 0,
#         'Moderately differentiated; Grade II': 1,
#         'Poorly differentiated; Grade III': 2,
#         'Undifferentiated; anaplastic; Grade IV': 3,
#         'Unknown/unspecified': np.nan
#     })

#     ## Numerical
#     numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
#           "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
#     imputer = SimpleImputer(strategy='mean')
#     df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns = numerical_col, index = df.index)

#     return pd.concat([df_cat, df_num, df_ord, df[['duration', 'event']]], axis = 1)



def process_seer(df):
    # Clean categorical values and rename duration
    df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace=True)
    df["Sequence number"].replace(['88', '99'], np.nan, inplace=True)
    df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace=True)
    df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace=True)
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns={"Survival months": "duration"})

    # Filter patients with valid survival
    df = df[~df.duration.isna()]
    df['duration'] = df['duration'].astype(float)

    # Encode event variable
    df['event'] = (df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)").astype(int)
    df.loc[
        (df["COD to site recode"] == "Diseases of Heart") & 
        (df["SEER cause-specific death classification"] == "Alive or dead of other cause"), 
        "event"
    ] = 2
    df = df.drop(columns=["COD to site recode"])

    # Define columns
    categorical_col = [
        "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", 
        "RX Summ--Surg Prim Site (1998+)", "CS extension (2004-2015)", 
        "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", 
        "Origin recode NHIA (Hispanic, Non-Hisp)"
    ]
    ordinal_col = ["Age recode with <1 year olds", "Year of diagnosis"]
    numerical_col = [
        "Total number of in situ/malignant tumors for patient", 
        "Total number of benign/borderline tumors for patient",
        "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", 
        "Regional nodes positive (1988+)"
    ]

    # Filter to only columns that exist
    categorical_col = [col for col in categorical_col if col in df.columns]
    ordinal_col = [col for col in ordinal_col if col in df.columns]
    numerical_col = [col for col in numerical_col if col in df.columns]

    # Encode categorical
    imputer_cat = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder()
    df_cat = pd.DataFrame(
        encoder.fit_transform(imputer_cat.fit_transform(df[categorical_col])),
        columns=categorical_col,
        index=df.index
    )

    # Impute ordinal
    imputer_ord = SimpleImputer(strategy='most_frequent')
    df_ord = pd.DataFrame(imputer_ord.fit_transform(df[ordinal_col]), columns=ordinal_col, index=df.index)

    # Encode age group
    df_ord["Age recode with <1 year olds"] = df_ord["Age recode with <1 year olds"].replace({
        '00 years': 0, '01-04 years': 1, '05-09 years': 2, '10-14 years': 3,
        '15-19 years': 4, '20-24 years': 5, '25-29 years': 6, '30-34 years': 7,
        '35-39 years': 8, '40-44 years': 9, '45-49 years': 10, '50-54 years': 11,
        '55-59 years': 12, '60-64 years': 13, '65-69 years': 14, '70-74 years': 15,
        '75-79 years': 16, '80-84 years': 17, '85-89 years': 18, '90-94 years': 19,
        '90+ years': 19, '95-99 years': 20, '100+ years': 21, 'Unknown age': np.nan
    })

    df_ord = df_ord.astype(float)

    # Encode numerical
    imputer_num = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(
        imputer_num.fit_transform(df[numerical_col].astype(float)),
        columns=numerical_col,
        index=df.index
    )

    result = pd.concat([df_cat, df_num, df_ord, df[['duration', 'event']]], axis=1)

    # Check for non-numeric columns before modeling
    non_numeric = result.select_dtypes(include='object')
    print("⚠️ Non-numeric columns before modeling:", non_numeric.columns.tolist())
    print(non_numeric.head())
    assert all(result.dtypes != 'object'), "Non-numeric columns found before modeling."

    return result
