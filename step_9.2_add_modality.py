import config
import utils
from tqdm import tqdm
import re
from mappers.modality_data import subtext_to_dicom_modality_dict


def extract_modality(labeled_text, text, previous_label, modality_pattern):
    # prepare labeled_text: strip and remove wrong spaces left from joining tokens
    labeled_text = re.sub(r'\s+([?.!"])', r'\1', labeled_text.strip())

    label_index = text.find(labeled_text) # todo unfortunately we've lost the information on the location of tha labeled text. We can improve this.

    text_part = text[previous_label['idx']:label_index]
    newline_indexes = [match.end() for match in re.finditer(r"[\n]", text_part)]
    if newline_indexes:
        text_part = text_part[newline_indexes[-1]:]

    modality = None

    match = re.search(modality_pattern, text_part)  # config.modality_regex_pattern, text_part)
    if match is not None:
        exam_type = text_part[match.start():match.end()]
        modality = subtext_to_dicom_modality_dict[exam_type]

    previous_label['idx'] = label_index
    return modality

df_all_pred_sequences = utils.read_df_from_pickle(config.all_predicted_sequences)

df_segmented = utils.read_df_from_pickle(config.all_reports_segmented_pickle)

assert df_segmented.shape[0] == df_segmented.document_id.unique().size

# doing it only for the ones where we have predictions
prediction_document_ids = df_all_pred_sequences.document_id.unique()
reports_with_predictions_filter = df_segmented.document_id.isin(prediction_document_ids)
df_target_text = df_segmented[['document_id', 'target_text']]
df_target_text = df_target_text[reports_with_predictions_filter]

# TODO not documented by I think we want this to be en empty set
assert len(set(df_target_text.document_id.unique()) - set(df_all_pred_sequences.document_id.unique())) == 0

# todo maybe break this up
# first pandas.Series.str.replace to do that replace form above, that should be very slow

tqdm.pandas()


def f(df_group, df_documents, modality_pattern):
    # this works by using references in python.
    # The values are being used in a loop
    # todo rewrite in a more pythonic way
    document_id = df_group.document_id.iloc[0]
    previous_label = {'idx': 0}
    df_group['modality'] = df_group['labeled_text'].apply( # but if tehre is no previous or not labeled, we don't need to do this
        extract_modality,
        text=df_documents.target_text.loc[document_id], # df_documents[df_documents.document_id == document_id].target_text.iloc[0],
        previous_label=previous_label,
        modality_pattern=modality_pattern
    )
    return df_group


df_target_text = df_target_text.set_index('document_id')
modality_regex_pattern = f'|'.join(
    map(
        lambda x: f'({x})',
        subtext_to_dicom_modality_dict.keys()
    )
)

result = df_all_pred_sequences.groupby('document_id').progress_apply(
    f,
    df_documents=df_target_text,
    modality_pattern=modality_regex_pattern
)

utils.save_to_pickle(
    result,
    config.all_predicted_sequences_with_modality,
    'df'
)
