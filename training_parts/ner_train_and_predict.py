import config
import numpy as np
import os
import pandas as pd
import re
import spacy
import tensorflow as tf
import ktrain
from ktrain import text
from datetime import datetime
from mappers.extracted_modality_data import extracted_mod_dict

# todo: figure out where to run with crf layer or without and with character embedding or without
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['DISABLE_V2_BEHAVIOR'] = '1'


class CustomTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sentence, *args, **kwargs):
        tokens = self.tokenizer(sentence)
        return [token.text for token in tokens]


def get_examination_methode(doc_text):
    # NOTE regex only extracts first modality, if multiple then only first is taken
    method_type = 'unknown'
    regex_pattern = '(PET-CT)|' \
                    '(-CT)|' \
                    '(PET/CT)|' \
                    '(CT)|' \
                    '(CTA)|' \
                    '(Mammografie)|' \
                    '(MRI-)|' \
                    '(MRI)|' \
                    '(MRT)|' \
                    '(R\u00f6ntgen)|' \
                    '(Szinti)|' \
                    '(Sonografie)'
    match_result = re.search(regex_pattern, doc_text)
    if match_result:
        exam_type = doc_text[match_result.start():match_result.end()]
        method_type = extracted_mod_dict[exam_type]
    return method_type


def get_text_part(text_list, label, previous_label_index):
    report_text = ' '.join(text_list)
    label_index = report_text.find(label)
    part = report_text[previous_label_index:label_index]
    newline_indexes = [match.end() for match in re.finditer(r"[\n]", part)]
    if newline_indexes:
        part = part[newline_indexes[-1]:]
    return part, label_index


# insert all tagged words from one document as DataFrame and extract the labeled text
def extract_labels(df_rows, report_id):
    labeled_rows = df_rows[df_rows['tag'] != 'O']
    tags_with_b = labeled_rows[labeled_rows['tag'].str.contains("B")]
    tag = 'B-missing'
    labeled_text = np.nan
    all_labels = []

    if len(tags_with_b.index) >= 1:
        # add a new index at the end in order to get the last one
        indexes = tags_with_b.index
        expanded_indexes = indexes.append(pd.Index([indexes[-1] + 1]))
        prev_label_idx = 0

        for start_idx, end_idx in zip(expanded_indexes[:-1], expanded_indexes[1:]):
            df_labeled_text = labeled_rows.loc[start_idx:end_idx-1]
            labeled_text_list = df_labeled_text['labeled_text'].values.tolist()
            labeled_text = ' '.join(labeled_text_list)
            tag = df_labeled_text['tag'].iloc[0]

            # extract the part of the text, where you find the examination method
            text_part, prev_label_idx = get_text_part(
                df_rows['labeled_text'].values.tolist(),
                labeled_text,
                prev_label_idx)
            method = get_examination_methode(text_part)
            all_labels.append([report_id, labeled_text, tag[2:], method])

    else:
        all_labels.append([report_id, labeled_text, tag[2:], None])

    return all_labels


# predicts on new data and returns the found labels with their report id
# the output is a list of dataFrames with one word per row and its label
# and a list with a report id per row and the report's labels
def get_labels_from_prediction(data_set, pred, tokenizer):
    unique_ids = data_set['report_id'].drop_duplicates().tolist()
    results = []
    df_results = []
    for report_id in unique_ids:
        test_sent_list = data_set[data_set['report_id'] == report_id]['sent_list'].values.tolist()
        predictions = pred.predict(test_sent_list, custom_tokenizer=tokenizer)
        cols = ['labeled_text', 'tag']
        df_result = pd.DataFrame([token for prediction in predictions for token in prediction], columns=cols)
        df_results.append(df_result)
        labels = extract_labels(df_result, report_id)
        for label in labels:
            results.append(label)
    return results, df_results


def get_wv_model(preproc, force_download=False):
    if force_download:
        wv_path_or_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz'
    else:
        wv_path_or_url = config.path_to_data_dir / 'cc.de.300.vec'
    return text.sequence_tagger('bilstm-crf', preproc, wv_path_or_url=str(wv_path_or_url))


def get_callback(dir_name):
    log_dir = config.path_to_data_dir / dir_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_images=True,
        update_freq='batch'
    )


def train(preprocessed_data_path, lr, number_of_epochs, predictor_dir_path=None):
    # lr: 1e-2 the value we are using most often

    (trn, val, preproc_model) = text.entities_from_txt(
        preprocessed_data_path,
        sentence_column='sent_nr_training',
        word_column='word',
        tag_column='tag',
        data_format='gmb',
        use_char=False
    )

    wv_model = get_wv_model(preproc_model)
    learner = ktrain.get_learner(
        wv_model,
        train_data=trn,
        val_data=val,
        batch_size=128
    )

    tb_call_back = get_callback('tag_train')

    learner.fit(
        lr=lr,
        n_cycles=number_of_epochs,
        cycle_len=1,
        early_stopping=3,
        callbacks=[tb_call_back]
    )

    predictor = ktrain.get_predictor(learner.model, preproc_model)

    if predictor_dir_path is not None:
        predictor.save(predictor_dir_path)

    return predictor, learner


def predict(df_data, predictor):
    if type(predictor) is str:
        predictor = ktrain.load_predictor(predictor)

    # make predictions
    spacy.require_gpu()
    nlp = spacy.load('de_core_news_sm')
    custom_ner_tokenizer = CustomTokenizer(nlp.tokenizer)

    return get_labels_from_prediction(
        df_data,
        predictor,
        custom_ner_tokenizer
    )
