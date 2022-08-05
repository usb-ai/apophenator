import pickle

import pandas as pd

import config
import utils

# TODO is this obsolete? Or is it used for the paper?

df_all_predictions = utils.read_df_from_pickle(config.all_predicted_sequences)

unique_doc_ids_noprev = df_all_predictions[
 df_all_predictions.label == 'no previous'
 ].document_id.unique().tolist()

unique_doc_ids_prev = df_all_predictions[
 (df_all_predictions.label == 'date') | (df_all_predictions.label == 'today')
 ].document_id.unique().tolist()


unique_doc_ids_prev_and_noprev = set(unique_doc_ids_prev).intersection(set(unique_doc_ids_noprev))

unique_doc_ids_noprev = set(unique_doc_ids_noprev) - unique_doc_ids_prev_and_noprev

unique_doc_ids_prev = set(unique_doc_ids_prev) - unique_doc_ids_prev_and_noprev

# FIXME reports having previous and no previous will be counted as previous
df_prev =  pd.DataFrame(list(unique_doc_ids_prev | unique_doc_ids_prev_and_noprev), columns=['document_id'])
df_prev['label'] = 'previous'
df_no_prev = pd.DataFrame(list(unique_doc_ids_noprev), columns=['document_id'])
df_no_prev['label'] = 'no previous'

with open('/home/marko/projects/apophenator/document_ids_missing_label', 'rb') as f:
    unique_doc_ids_missing = pickle.load(f)

df_missing = pd.DataFrame(list(unique_doc_ids_missing), columns=['document_id'])
df_missing['label'] = 'missing'

overlap_missing_not_missing = ((unique_doc_ids_prev | unique_doc_ids_prev_and_noprev) | unique_doc_ids_noprev).intersection(set(unique_doc_ids_missing))
assert len(overlap_missing_not_missing) == 0

df = pd.concat([df_prev, df_no_prev, df_missing], axis=0, ignore_index=True)

assert df.document_id.unique().size == (df_prev.shape[0] + df_no_prev.shape[0] + df_missing.shape[0])

df_segmented = utils.read_df_from_pickle(config.all_reports_segmented_pickle)

df_segmented = df_segmented[
    ['patient_id',
     'accession_id',
     'examination_date',
     'examination_type',
     'exam_name',
     'exam_description',
     'organ',
     'organ_description',
     'dicom_modality',
     'dicom_modality_description',
     'document_id'
     ]
]
# df_segmented = df_segmented[(df_segmented.examination_date > 20120000) & (df_segmented.examination_date < 20220000)]
df_segmented.examination_date = df_segmented.examination_date.apply(str)

df_merged = df_segmented.merge(df, on='document_id', how='inner')

assert df_merged.document_id.unique().size == df_merged.document_id.size
# unique_labels_per_document = df_merged.groupby('document_id')['label'].unique()
# df_labels_per_document = df_merged.groupby('document_id')['label'].unique().reset_index()

df_merged.to_csv('/home/marko/projects/apophenator/laurent_analysis_labels.csv')  # TODO this is also not relative


print('hallo')