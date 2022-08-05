import json
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import spacy
from sqlalchemy import create_engine
from tqdm import tqdm
from collections import Counter

import config
from mappers import modality_data, region_data

import re

# TODO THIS NEEDS TO GO INTO ENV FILE
def establish_connection(user, password):
    connection_string = f'hana://{user}:{password}@ictthdbdwlp1.uhbs.ch:33241/?encrypt=true&sslValidateCertificate=false'
    engine = create_engine(connection_string)
    return engine.connect()


def is_overlap(set1: set, set2: set, minimal_overlap_amount: int = 2) -> bool:
    overlap = set1.intersection(set2)
    if len(overlap) == minimal_overlap_amount:
        return True
    return False


def is_df_empty(df: pd.DataFrame):
    return len(df.index) == 0


def get_na(df: pd.DataFrame, column: str):
    return df.iloc[np.where(df[column].isna())]


def is_valid_json_string(potetial_json_string: str):
    try:
        json_object = json.loads(potetial_json_string)
    except ValueError as e:
        return False
    return True


def get_nlp():
    try:
        spacy.require_gpu()
    except Exception as e:
        print('gpu fail:\n', e)

    nlp = spacy.load('de_core_news_sm',
                     exclude=[
                         "tok2vec",
                         "tagger",
                         "parser",
                         "ner",
                         "attribute_ruler",
                         "lemmatizer",
                         "entity_linker",
                         "entity_ruler",
                         "textcat",
                         "textcat_multilabel",
                         "morphologizer",
                         "senter",
                         "transformer"
                     ])
    nlp.add_pipe('sentencizer')
    return nlp


def create_gmb_dataframe(df, out_of_bounds_path=None):
    nlp = get_nlp()

    df_unique_docs = (
        df[['target_text', 'document_id']]
        .drop_duplicates(subset=['document_id'], keep='first')
    )

    unique_docs_generator = nlp.pipe(df_unique_docs.target_text) #, n_process=6, batch_size=20)

    # use spacy on the texts to get documents, sentences and tokens
    unique_document_ids = df_unique_docs['document_id'].values

    # initialise columns for dataFrame
    gmb_labels = []
    tokens = []
    sentence_id = []
    document_ids = []
    sentence_id_doc = []
    sent_count = 0
    offsets_out_of_bounds = []

    # in each document look for the token, that gets labeled differently

    for document_id, doc in tqdm(zip(unique_document_ids, unique_docs_generator)):
        document_id_filter = df['document_id'] == document_id  # todo put index on this in step 1
        labeled_texts = df['labeled_text'][document_id_filter].values.tolist()
        xtag_labels = df['class_name'][document_id_filter].values.tolist()
        start_offsets, end_offsets, offset_out_of_bounds_doc_ids = get_offsets(doc.text, labeled_texts, document_id)  # TODO: check inside

        df_labels = pd.DataFrame(
            {
                'start_offsets': start_offsets,
                'end_offsets': end_offsets,
                'labels': xtag_labels}
        )

        offsets_out_of_bounds.extend(offset_out_of_bounds_doc_ids)

        sent_count_per_doc = 0

        for sent in doc.sents:
            sent_count += 1
            sent_count_per_doc += 1

            # label token according to 'start' and 'end' from unique_doc
            for token in sent:

                token_start_idx = token.idx
                token_end_idx = token_start_idx + len(token)


                conll_label = get_label(df_labels, token_start_idx, token_end_idx)

                if conll_label is None:
                    print(f'flawed {document_id}')
                    conll_label = 'O'

                gmb_labels.append(conll_label)

                tokens.append(token.text)
                sentence_id.append(sent_count)
                sentence_id_doc.append(sent_count_per_doc)
                document_ids.append(document_id)

    if out_of_bounds_path is not None:
        print("Label out of bounds: ", len(offsets_out_of_bounds))
        label_out_of_bounds_df = df[df['document_id'].isin(offsets_out_of_bounds)]
        label_out_of_bounds_df.to_csv(out_of_bounds_path)

    # initialise dataFrame in order to create a file in teh right format for ner
    df_ner = pd.DataFrame(
        {
            'document_id': document_ids,
            'sentence_id_doc': sentence_id_doc,
            'sentence_id_training': sentence_id,
            'token': tokens,
            'conll_label': gmb_labels
        }
    )

    return df_ner


def create_conll_no_label_dataframe(df):
    nlp = get_nlp()

    df_unique_docs = (
        df[['target_text', 'document_id']]
        .drop_duplicates(subset=['document_id'], keep='first')
    )

    unique_docs_generator = nlp.pipe(df_unique_docs.target_text)

    # use spacy on the texts to get documents, sentences and tokens
    unique_document_ids = df_unique_docs['document_id'].values

    # initialise columns for dataFrame
    tokens = []
    sentence_id = []
    document_ids = []
    sentence_id_doc = []
    sent_count = 0
    # in each document look for the token, that gets labeled differently

    for document_id, doc in tqdm(zip(unique_document_ids, unique_docs_generator)):
        sent_count_per_doc = 0

        for sent in doc.sents:
            sent_count += 1
            sent_count_per_doc += 1

            for token in sent:

                tokens.append(token.text)
                sentence_id.append(sent_count)
                sentence_id_doc.append(sent_count_per_doc)
                document_ids.append(document_id)

    # initialise dataFrame in order to create a file in teh right format for ner
    df_ner = pd.DataFrame(
        {
            'document_id': document_ids,
            'sentence_id_doc': sentence_id_doc,
            'sentence_id_training': sentence_id,
            'token': tokens,
        }
    )
    return df_ner


def create_gmb_file(data_pickle_path, output_path=None, failed_output_path=None):
    df = pd.read_pickle(data_pickle_path)

    # for training
    gmb_df = create_gmb_dataframe(  # todo there is a fixed path inside this function
        df,
        failed_output_path,
    )

    if output_path is not None:
        gmb_df.to_pickle(output_path)

    return gmb_df


def get_finding(text, cnt):
    finding = ''
    lax_sufix_finding = ''
    convo_finding = ''

    regular_pattern = '\n+[\\s:]*befund[\\s:]*\n+(.*)\n+\\s*beurteilung.{0,5}\n+(.*)'
    regular_search_result = re.search(
        regular_pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if regular_search_result:
        first_group = regular_search_result.group(1)
        if first_group:
            cnt['regular_finding'] += 1
            finding = first_group
            return finding, lax_sufix_finding, convo_finding

    just_finding_search = '\n+[\\s:]*befund[\\s:]*\n+(.*)'
    just_finding_result = re.search(
        just_finding_search,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if just_finding_result:
        first_group = just_finding_result.group(1)
        if first_group:
            cnt['just_finding'] += 1
            finding = first_group
            return finding, lax_sufix_finding, convo_finding

    # TODO can add another without beurteilung as last resort
    convo_search_result = re.search(
        '\n+[^\n]{0,14}befund[^\n]{0,14}\n+(.*)[\n ]*beurteilung',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if convo_search_result:
        first_group = convo_search_result.group(1)
        if first_group:
            cnt['convo_finding'] += 1
            convo_finding = first_group
            return finding, lax_sufix_finding, convo_finding

    super_lax_search_result = re.search(
        '\n+\\s*[Bb]efund[ ]([A-Z]{1}.*)[Bb]eurteilung',
        text,
        flags=re.DOTALL
    )

    if super_lax_search_result:
        first_group = super_lax_search_result.group(1)
        if first_group:
            cnt['super_lax_finding'] += 1
            lax_sufix_finding = first_group
            return finding, lax_sufix_finding, convo_finding

    return finding, lax_sufix_finding, convo_finding


def get_impression(text, cnt):
    impression = ''
    convo_impression = ''

    regular_impression_pattern = '\n+\\s*beurteilung\\s*:?\\s*\n+(.*)'

    regular_search_result = re.search(
        regular_impression_pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if regular_search_result:
        first_group = regular_search_result.group(1)
        if first_group:
            cnt['regular_impression'] += 1
            impression = first_group
            return impression, convo_impression

    convo_search_result = re.search(
        '\n+[^\n]{0,5}beurteilung[^\n]{0,2}\n+(.*)',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if convo_search_result:
        first_group = convo_search_result.group(1)
        if first_group:
            cnt['convo_impression'] += 1
            convo_impression = first_group
            return impression, convo_impression

    return impression, convo_impression


def get_bnb(text, cnt):
    bnb = ''

    bnb_pattern = '\n+\\s*\\b(?:befund|beurteilung)\\b[^\n]{0,7}\\b(?:befund|beurteilung)\\b\\s*:*\\s*\n+(.*)'

    regular_search_result = re.search(
        bnb_pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if regular_search_result:
        first_group = regular_search_result.group(1)
        if first_group:
            cnt['bnb'] += 1
            bnb = first_group
            return bnb

    return bnb


def add_target_text(df, do_drop_na=True):
    df['target_text'] = None

    bnb_cond = df.finding_and_impression != ''
    df.loc[bnb_cond, 'target_text'] = df.finding_and_impression[bnb_cond]

    lax_cond = df.lax_sufix_finding != ''
    df.loc[lax_cond, 'target_text'] = df.lax_sufix_finding[lax_cond]

    convo_cond = df.convo_finding != ''
    df.loc[convo_cond, 'target_text'] = df.convo_finding[convo_cond]

    finding_cond = df.finding != ''
    df.loc[finding_cond, 'target_text'] = df.finding[finding_cond]

    # todo has 19 of them, fix or explain or whatever
    if do_drop_na:
        df = df.dropna(subset=['target_text'])

    return df


def do_segmentation(df, do_drop_na=True):
    findings = []
    convo_findings = []
    lax_sufix_findings = []
    impressions = []
    convo_impressions = []
    bnbs = []
    just_findings = []
    just_impressions = []
    bad_flags = []

    cnt = Counter()

    for entry in tqdm(df.text):
        text = str(entry)

        finding, lax_sufix_finding, convo_finding = ['', '', '']
        impression, convo_impression = '', ''

        bnb_text = get_bnb(text, cnt)
        if not bnb_text:
            finding, lax_sufix_finding, convo_finding = get_finding(text, cnt)
            impression, convo_impression = get_impression(text, cnt)

        bnbs.append(bnb_text)
        findings.append(finding)
        convo_findings.append(convo_finding)
        lax_sufix_findings.append(lax_sufix_finding)
        impressions.append(impression)
        convo_impressions.append(convo_impression)
        just_findings.append(
            any([finding, lax_sufix_finding, convo_finding]) and not any([impression, convo_impression]))
        just_impressions.append(
            not any([finding, lax_sufix_finding, convo_finding]) and any([impression, convo_impression]))
        bad_flags.append(
            not any(
                [
                    finding,
                    lax_sufix_finding,
                    convo_finding,
                    impression,
                    bnb_text,
                ]
            )
        )

    print(cnt)

    df['finding'] = findings
    df['impression'] = impressions
    df['convo_impression'] = convo_impressions
    df['finding_and_impression'] = bnbs
    df['just_finding'] = just_findings
    df['just_impression'] = just_impressions
    df['convo_finding'] = convo_findings
    df['lax_sufix_finding'] = lax_sufix_findings
    df['bad_flag'] = bad_flags

    df = add_target_text(df, do_drop_na)

    return df


def mark_predictions_without_b(curr_label, previous):
    if curr_label.startswith('B') or curr_label.startswith('O'):
        previous['label'] = curr_label
        previous['is_faulty'] = False
        return False

    if curr_label.startswith('I') and previous['label'].startswith('I'):
        label_is_faulty = True
        if curr_label.split('-')[-1] == previous['label'].split('-')[-1]:
            label_is_faulty = previous['is_faulty']
        return label_is_faulty

    if curr_label.startswith('I') and previous['label'].startswith('B'):
        label_is_faulty = not (curr_label.split('-')[-1] == previous['label'].split('-')[-1])
        previous['label'] = curr_label
        previous['is_faulty'] = label_is_faulty
        return label_is_faulty

    if curr_label.startswith('I') and previous['label'].startswith('O'):
        previous['label'] = curr_label
        previous['is_faulty'] = True
        return True


# insert all labeled words from one document as DataFrame and extract the labeled text
def extract_labels(df_gmb: pd.DataFrame) -> pd.DataFrame:
    df_gmb_labeled_B = df_gmb[df_gmb.pred.str.startswith('B')]

    all_labels = []
    # add a new index at the end in order to get the last one
    indexes = df_gmb_labeled_B.index
    indexes = indexes.append(pd.Index([df_gmb.index[-1] + 1]))

    for start_idx, end_idx in zip(indexes[:-1], indexes[1:]):
        df_labeled = df_gmb.loc[start_idx:end_idx-1]
        labeled_text = ' '.join(df_labeled['token'].values.tolist())

        predicted_label = df_labeled['pred'].iloc[0][2:]
        document_id = df_labeled['document_id'].iloc[0]
        all_labels.append([document_id, labeled_text, predicted_label])

    return pd.DataFrame(all_labels, columns=['document_id', 'labeled_text', 'label'])

#
# def search_for_previous_examinations(df_predictions, df_metadata):
#
#     previous_documents_list = []
#
#     def f(df_row):
#         metadata_per_doc= df_metadata.loc[df_row.document_id]
#
#         df_previous_per_document = get_previous_reports_from_cdwh(  # todo check inside, tehre is mapping that is not needed
#             patient_id=metadata_per_doc.PatientID,
#             accession_number=metadata_per_doc.AccessionNumber,
#             study_date=df_row.parsed_prediction,
#         )
#
#         df_previous_per_document['document_id'] = df_row.document_id
#         previous_documents_list.append(df_previous_per_document)
#
#     df_predictions.apply(func=f, axis=1)
#
#     df_previous_documents = pd.concat(previous_documents_list, axis=0)
#
#     df_previous_documents.reset_index(inplace=True, drop=True)
#
#     return df_previous_documents
#
#
# def get_previous_reports_from_cdwh(patient_id, accession_number, study_date):
#     df_previous = pd.DataFrame()
#
#     if study_date not in ['no previous', None]:
#         conn = establish_connection(config.db_user, config.db_pw)
#         query = f'''
#                     SELECT
#                     e.PAT_BK AS patient_id,
#                     e.IME_BK AS accession_number,
#                     e.IME_BEGIN_DAY_BK AS study_date,
#                     r.IER_REPORT AS report_text,
#                     e.IET_BK AS examination_type
#                     FROM CDWH.V_IL_FCT_IMAGING_EXAM e
#                     INNER JOIN CDWH.V_IL_DIM_IMAGING_EXAM_TYPE_CUR t
#                     ON t.IET_BK = e.IET_BK
#                     INNER JOIN CDWH.V_IL_FCT_IMAGING_EXAM_REPORT r
#                     ON r.IER_BK = e.IER_BK
#                     WHERE e.IME_STATUS_CODE = 'f'
#                     AND e.PAT_BK = \'{patient_id.replace('USB', '')}\'
#                     AND e.IME_BK != \'{accession_number}\'
#                     AND e.IME_BEGIN_DAY_BK = \'{study_date.strftime('%Y%m%d')}\'
#         '''
#
#         df_previous = pd.read_sql(sql=query, con=conn)
#
#         if is_df_empty(df_previous):
#             df_previous.loc[0] = [None for i in range(0, df_previous.shape[1])]
#
#         def resolve_modality_and_region(df_row):
#             modality = None
#             region = None
#             if df_row.examination_type is not None:
#                 modality = modality_data.mod_dict.get(df_row.examination_type)
#                 if modality is None:
#                     modality = 'unknown'
#                 region = region_data.region_dict.get(df_row.examination_type)
#                 if region is None:
#                     region = 'unknown'
#             df_row.examination_type = modality
#             df_row['region'] = region
#             return df_row
#
#         df_previous = df_previous.apply(resolve_modality_and_region, axis=1)
#
#     return df_previous


def save_to_pickle(variable, pickle_path, variable_type):
    print(f'Saving to {pickle_path}...', end='')

    start_time = datetime.now()

    if variable_type == 'df':
        variable.to_pickle(pickle_path)

    elif variable_type == 'list':
        with open(pickle_path, "wb") as open_file:
            pickle.dump(variable, open_file)

    else:
        raise ValueError(
            f'Unknown variable type. '
            f'Expected variable_type of df or list '
            f'but {variable_type} received'
        )

    print('done. Duration: ', datetime.now() - start_time)


def read_df_from_pickle(pickle_path):
    if pickle_path.is_file():
        print(f'File {pickle_path} exists, loading...', end='')
        start_time = datetime.now()
        df = pd.read_pickle(pickle_path)
        print('done. Duration: ', datetime.now() - start_time)
        return df

    print(f'File "{pickle_path}" does not exist')
    return None


def get_offsets(doc_text, labeled_texts, doc_id):
    if not labeled_texts:
        start_indices = [-1]
        end_indices = [-1]
        return start_indices, end_indices, []

    start_indices = []
    end_indices = []
    offset_out_of_bounds_doc_ids = []  # todo rename - these are errors and no label texts

    for idx in range(0, len(labeled_texts)):
        start_index = doc_text.find(labeled_texts[idx])
        start_indices.append(start_index)

        if start_indices[idx] != -1:
            end_indices.append(start_indices[idx] + len(labeled_texts[idx]))
            continue

        end_indices.append(-1)
        offset_out_of_bounds_doc_ids.append(doc_id)

    return start_indices, end_indices, offset_out_of_bounds_doc_ids


def get_label(df_labels, token_start_idx, token_end_idx):
    df_beginnings = df_labels[
        (token_start_idx <= df_labels.start_offsets) &
        (df_labels.start_offsets < token_end_idx)
    ]

    # assert 0 <= len(df_beginnings) <= 1
    if not (0 <= len(df_beginnings) <= 1):
        print(f'flawed!')
        return None

    if len(df_beginnings) != 0:
        return f'B-{df_beginnings.labels.iloc[0]}'

    df_intermediates = df_labels[
        (df_labels.start_offsets < token_start_idx) &
        (token_start_idx < df_labels.end_offsets)
    ]

    assert 0 <= len(df_intermediates) <= 1

    if len(df_intermediates) != 0:
        return f'I-{df_intermediates.labels.iloc[0]}'

    return 'O'


# todo not used afaik
def string2date(s: int):
    return datetime(
        year=int(s[0:4]),
        month=int(s[4:6]),
        day=int(s[6:8])
    )


def int2date(argdate: int) -> datetime.date:
    """
    If you have date as an integer, use this method to obtain a datetime.date object.

    Parameters
    ----------
    argdate : int
      Date as a regular integer value (example: 20160618)

    Returns
    -------
    dateandtime.date
      A date object which corresponds to the given value `argdate`.
    """
    year = int(argdate / 10000)
    month = int((argdate % 10000) / 100)
    day = int(argdate % 100)

    return datetime(year, month, day)
