import config
import dateparser as dp
import os
import pandas as pd
import pickle
from evaluation_and_visualization.compare_predicted_vs_true_data import evaluate_and_print_classification_results
from evaluation_and_visualization.graph_builder import update_graph_db
from evaluation_and_visualization.ris_report_conn import extract_db_info, exam_type_and_region_from_db, ReportNode


def get_today_date(report_id: int, df: pd.DataFrame):
    date = df[df['report_id'] == report_id]['study_date']
    return date


# returns a date in the metadata format
def parse_date(date: str):
    if dp.parse(date, date_formats=['%Y%m%d'], languages=['de']):
        parsed_date = dp.parse(date, date_formats=['%Y%m%d'], languages=['de']).strftime('%Y%m%d')
    else:
        parsed_date = 'not a date'
    return parsed_date


# returns a list of the predicted dates belonging to the given document_id
def get_predicted_dates(report_id: int, df: pd.DataFrame, df_test_info: pd.DataFrame):
    df = df[df['report_id'] == report_id]
    examination_types = len(df) * ['unknown']

    df_dates = df[df['tag'] == 'date']
    date_dates = []
    if not df_dates.empty:
        date_list = df_dates['label_text'].tolist()
        date_dates = [parse_date(date) for date in date_list]
        examination_types = df_dates['type'].tolist()

    df_today = df[df['tag'] == 'today']
    today_dates = []
    if not df_today.empty:
        today_dates = get_today_date(report_id, df_test_info).tolist()
        examination_types = df_today['type'].tolist()

    df_no_previous = df[df['tag'] == 'no previous']
    no_previous = []
    if not df_no_previous.empty:
        no_previous_list_len = len(df_no_previous['label_text'].tolist())
        no_previous = ['no previous' for i in range(no_previous_list_len)]
        examination_types = df_no_previous['type'].tolist()

    df_missing = df[df['tag'] == 'missing']
    missing = []
    if not df_missing.empty:
        missing_list_len = len(df_missing['label_text'].tolist())
        missing = ['missing' for i in range(missing_list_len)]
        examination_types = df_missing['type'].tolist()

    dates = date_dates + today_dates + missing + no_previous
    return dates, examination_types


def search_for_previous_examinations(report_ids, predictions_df, df_test_info):
    prev_doc_list = []

    for report_id in report_ids:

        # get patient_id, text, date, accession_nr and examination type of current document
        patient_nr = df_test_info[df_test_info['report_id'] == report_id]['patient_id'].iloc[0]
        doc_date = df_test_info[df_test_info['report_id'] == report_id]['study_date'].iloc[0]
        doc_text = df_test_info[df_test_info['report_id'] == report_id]['text'].iloc[0]
        accession_number = df_test_info[df_test_info['report_id'] == report_id]['accession_nr'].iloc[0]
        doc_exam_type, doc_region = exam_type_and_region_from_db(accession_number, patient_nr)

        # check if exam type exists otherwise document isn't important and we can skip it
        if not doc_exam_type:
            continue

        # get info about previous examinations with dates
        previous_examination_dates, doc_types = get_predicted_dates(report_id, predictions_df, df_test_info)

        # search in data for documents with the extracted patient_id and the previous dates and return the documents
        previous_documents = extract_db_info(
            accession_number,
            patient_nr,
            previous_examination_dates,
            doc_types
        )
        current_doc = ReportNode(patient_nr, accession_number, doc_date, doc_text, doc_exam_type, doc_region)
        prev_doc_list.append([current_doc] + previous_documents)

    return prev_doc_list


# todo: evaluate predicted modalities
def evaluate_modality_extraction(df_preds):
    df_true = pd.read_csv(config.path_to_data_dir / 'prepared_control_set_modalities.csv')
    y_true = df_true['layer_name'].values.tolist()
    y_preds = df_preds['type'].values.tolist()
    class_names = ['Szintigrafie', 'MRI', 'CT', 'RX', 'Fluoroskopie', 'PET-CT', 'Sonographie', 'unknown']
    if len(y_true) == len(y_preds):
        print('Evaluation: Modality extraction')
        evaluate_and_print_classification_results(y_true, y_preds, class_names)
    return


def evaluate_and_update_db(df_test_info, path_to_predicted_labels):
    # load file with predicted labels if the variable predicted_label is a file
    if type(path_to_predicted_labels) == str:
        if os.path.isfile(path_to_predicted_labels):
            with open(path_to_predicted_labels, "rb") as fp:
                predicted_labels = pickle.load(fp)
        else:
            print('file does not exist')

    df_predictions = pd.DataFrame(predicted_labels, columns=['report_id', 'label_text', 'tag', 'type'])

    # evaluate_modality_extraction(df_predictions)

    # iterate through documents and search for previous examinations
    report_ids = df_predictions['report_id'].unique().tolist()
    results = search_for_previous_examinations(report_ids, df_predictions, df_test_info)

    # prepare results in order to make a graph
    df_results = pd.DataFrame(results)
    df_prev_docs = df_results[df_results.iloc[:, 1].notna()]
    update_graph_db(df_prev_docs)
