import json
import pandas as pd
import config


col_names = ['abbreviation', 'name', 'long name', 'group']
filepath = config.path_to_data_dir / 'ExaminationTypes.csv'
df_modality_types = pd.read_csv(filepath, names=col_names)
df_modality_types = df_modality_types.drop(columns=['name', 'long name'])
modality_dict = dict(zip(df_modality_types['abbreviation'].tolist(), df_modality_types['group'].tolist()))
a_file = open("modality_data.json", "w")
json.dump(modality_dict, a_file)
a_file.close()
