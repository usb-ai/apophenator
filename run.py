import config
import json
import pandas as pd
import pickle
from evaluation_and_visualization.evaluate_predicted_results import evaluate_and_update_db
from evaluation_and_visualization.ris_report_conn import get_data_from_db
from preprocess_data.preprocess_ner import preprocess_ner, preproc_data_set_for_prediction
from training_parts.ner_train_and_predict import train, predict


def load_and_prepare_data_for_preprocess(dataset):
    df_temporal = dataset[dataset['layer_name'] == 'temporal link']  # only keep temporal links
    idxs = df_temporal.loc[df_temporal['class_name'] == 'missing'].iloc[:, 0].tolist()
    df_temporal.loc[idxs, 'labeled_text'] = None
    df_temporal_final = df_temporal[df_temporal['class_name'] != 'yesterday']  # remove the 27 'yesterday' class_names
    return df_temporal_final


def save_data_as_csv(dataset, path_to_data_dir, filename):
    columns = ['sent_nr', 'sent_nr_training', 'word', 'tag']
    path_to_preprocessed_data = path_to_data_dir / (filename + '_training_set_preprocessed.csv')
    dataset.to_csv(path_to_preprocessed_data, columns=columns)


def get_metadata_as_df(unique_docs: pd.DataFrame, drop_metadata_column: bool = True) -> pd.DataFrame:
    df_metadata = pd.json_normalize(
        [json.loads(json_string) for json_string in unique_docs['metadata'].tolist()]
    )
    df_metadata.index = unique_docs.index
    if drop_metadata_column:
        unique_docs.drop(columns=['metadata'], inplace=True)
    return df_metadata


# PREPROCESS

# load training data and extend with meta data
df_training_data = pd.read_csv(config.path_to_training_data_file)

# load accession numbers for prediction
df_control_set = pd.read_csv(config.path_to_data_dir / 'test_data_modalities_extraction.csv')
df_control_set_unique = df_control_set.drop_duplicates(subset=['accession_nr'])
accession_nr_list = df_control_set_unique['accession_nr'].values.tolist()


# prepare, preprocess and save the data-file
df_prepared = load_and_prepare_data_for_preprocess(df_training_data)
training_data_filename_wo = config.training_data_filename_wo_extension
training_data_preprocessed = preprocess_ner(df_prepared, config.path_to_data_dir, training_data_filename_wo)
save_data_as_csv(training_data_preprocessed, config.path_to_data_dir, training_data_filename_wo)


# TRAINING

if config.train:
    train(config.path_to_preprocessed_training_data_file,
          lr=1e-2,
          number_of_epochs=1,
          predictor_dir_path=config.path_to_predictor_dir
          )


# PREDICTION

# load the data from the database and preprocess it
data_df = get_data_from_db(accession_nr_list)
preprocessed_data_df = preproc_data_set_for_prediction(data_df)

# predict the given patient reports in the preprocessed data
predicted_labels_with_report_id = predict(preprocessed_data_df, config.path_to_predictor_dir)[0]

# save the predicted labels
with open(config.path_to_predicted_labels, "wb") as fp:
    pickle.dump(predicted_labels_with_report_id, fp)

# function to evaluate the predicted results and store it in the graph database neo4j
path_to_predicted_labels_with_report_id = config.path_to_predicted_labels
evaluate_and_update_db(data_df, path_to_predicted_labels_with_report_id)
