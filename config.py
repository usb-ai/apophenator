import distutils.util as du
import os
import sys
from dotenv import dotenv_values
from pathlib import Path


# import env file
current_dir = os.path.dirname(__file__)
path_to_env_file = os.path.join(current_dir, '.env')
settings = dotenv_values(path_to_env_file)

# Flow control
batch_size=256
debug=False
force_recalculate=False
persist_output=True

# the directory, where the training data and the results are stored as Path
if 'PATH_TO_DATA_DIR' not in settings.keys():
    print('No data path defined!')
    sys.exit(-1)

root_dir = Path(settings['PATH_TO_DATA_DIR'])

# the directory, where the predictor is stored as string
path_to_predictor_dir = settings.get('PATH_TO_DATA_DIR', str(root_dir))

# filename of the training data as string
training_data_filename = settings.get(
    settings['FILENAME_TRAINING_DATA'],
    'laurent_training_data.csv'
)

# path to the training data as string
# TODO this could be all data, maybe the name is wrong - raw_data_path
raw_data_training_path = str(root_dir / training_data_filename)

# filename without the extension
training_data_filename_wo_extension = settings['FILENAME_TRAINING_DATA_WO_EXTENSION']


# the filename of the preprocessed training data as string
processed_data_path = settings.get(
    settings['FILENAME_PREPROC_TRAINING_DATA'],
    'laurent_data_training_set_preprocessed.csv'
)
# path to the preprocessed training data as string
path_to_preprocessed_training_data_file = str(root_dir / processed_data_path)

# filename of the predicted labels as string
predicted_labels_filename = settings.get(
    'FILENAME_PREDICTED_LABELS',
    'predicted_labels.txt'
)

# path to the predicted labels as string
path_to_predicted_labels = str(root_dir / predicted_labels_filename)

# username and password for the USB database
db_user = settings['DB_USER']
db_pw = settings['DB_PW']

# username and password for the graph database neo4j
neo4j_uri = settings['NEO4J_URI']
neo4j_user = settings['NEO4J_USER']
neo4j_pw = settings['NEO4J_PW']

# boolean if the train_and_save_predictor function should be executed
train_settings_raw = settings.get('TRAIN', 'True')
train = train_settings_raw == 'True'

train_data_segmented_path = root_dir / 'train_data_segmented_path.pkl'
label_out_of_bounds_path = root_dir / 'label_out_of_bounds.csv'
gmb_dataframe_path = root_dir / 'generated_conll.pkl'

test_data_sentencized_path = root_dir / 'test_sentencized.pkl'
test_data_out_of_thin_air = root_dir / 'test_data_modalities_extraction.csv'
test_data_segmented_path = root_dir / 'test_data_segmented.pkl'
test_data_conll_path = root_dir / 'test_data_conll.pkl'
test_data_gmb_predictions_path = root_dir / 'test_data_predictions.pkl'

all_reports_path = root_dir / 'all_radiology_reports.csv'
all_reports_segmented_pickle = root_dir / 'all_radiology_reports_segmented.pkl'
all_reports_pickle = root_dir / 'all_reports.pkl'
all_reports_empty_conll_pickle = root_dir / 'all_reports_empty_conll.pkl'
all_tokens_per_sentence = root_dir / 'all_tokens_per_sentence.pkl'

all_predictions = root_dir / 'all_predictions_list.pkl'
all_predictions_df = root_dir / 'all_predictions_df.pkl'
all_predictions_final = root_dir / 'all_predictions_final.pkl'

all_predicted_sequences = root_dir / 'all_predicted_sequences.pkl'
all_predicted_sequences_with_modality = root_dir / 'all_predicted_sequences_with_modality.pkl'

parsed_dmy_dates = root_dir / 'parsed_dmy_dates.pkl'

reference_date_pickle = root_dir / 'reference_date.pkl'

merged_references = root_dir / 'merged_references.pkl'


document_ids_missing_label = root_dir / 'document_ids_missing_label.pkl'

references_for_the_database = root_dir / 'references_the_database.pkl'