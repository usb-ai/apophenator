from xtag_preprocessor import XTagPreprocessor
import pandas as pd
import config

df_init_dataset = pd.read_csv(config.raw_data_training_path)

training_unique_document_ids = (
    pd.read_pickle(config.gmb_dataframe_path)
        .document_id.unique().tolist()
)
df_training_unique_documents = XTagPreprocessor.get_unique_documents(df_init_dataset).loc[training_unique_document_ids]
df_training_metadata_pacscrawler = XTagPreprocessor.get_metadata_as_df(df_training_unique_documents)

training_report_ids = (
        # df_training_metadata_pacscrawler.PatientID.str.replace('USB000', '')
        # + '_' +
        df_training_metadata_pacscrawler.AccessionNumber
)

assert training_report_ids.unique().shape[0] == training_report_ids.shape[0]
assert len(training_unique_document_ids) == len(training_report_ids)


test_unique_document_ids = (
    pd.read_pickle(config.test_data_gmb_predictions_path)
        .document_id.unique().tolist()
)
df_test_unique_documents = XTagPreprocessor.get_unique_documents(df_init_dataset).loc[test_unique_document_ids]
df_test_metadata_pacscrawler = XTagPreprocessor.get_metadata_as_df(df_test_unique_documents)

test_report_ids = (
        # df_test_metadata_pacscrawler.PatientID.str.replace('USB000', '')
        # + '_' +
        df_test_metadata_pacscrawler.AccessionNumber
).tolist()

assert len(test_unique_document_ids) == len(test_report_ids)


df_all_reports = pd.read_pickle('/media/marko/storage/all_reports.pkl')

df_all_reports['report_id'] = (
        # df_all_reports['patient_id'].apply(str)
        # + '_' +
        df_all_reports['accession_id'].apply(str)
)

assert len(df_all_reports.report_id.unique().tolist()) == len(df_all_reports.report_id.tolist())

df_training_metadata = df_all_reports[
    df_all_reports.report_id.isin(training_report_ids.tolist())
]

# debug
if not (len(df_training_metadata.report_id.tolist()) == len(training_report_ids.tolist())):
    missing_report_ids = list(set(training_report_ids.tolist()) - set(df_training_metadata.report_id.tolist()))
    # missing report_ids are removed already. this code is just to identify them ['26602557', '26602605']

df_test_metadata = df_all_reports[
    df_all_reports.report_id.isin(test_report_ids)
]

# debug
if not (len(df_test_metadata.report_id.tolist()) == len(test_report_ids)):
    missing = list(set(test_report_ids) - set(df_test_metadata.report_id.tolist()))


df_training_metadata.to_csv('/home/marko/projects/apophenator-noeminellen-data/training_metadata.csv')
df_test_metadata.to_csv('/home/marko/projects/apophenator-noeminellen-data/test_metadata.csv')

