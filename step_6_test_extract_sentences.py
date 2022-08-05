import config
import pandas as pd
import ktrain
from seqeval import metrics
import numpy as np
from pass_through_tokenizer import PassThroughTokenizer


if __name__ == '__main__':
    reloaded_predictor = ktrain.load_predictor(config.path_to_predictor_dir)
    pass_thorough_tokenizer = PassThroughTokenizer()

    test_gmb_df = pd.read_pickle(config.test_data_conll_path)
    test_gmb_df = test_gmb_df.sort_values(
        by=[
            'document_id',
            'sentence_id_doc'
        ]
    )

    tokens_per_sentence = test_gmb_df.groupby(
        by=[
            'document_id',
            'sentence_id_doc'
        ]
    )['token'].apply(list).to_list()

    target_labels = test_gmb_df.groupby(
        by=[
            'document_id',
            'sentence_id_doc'
        ]
    )['conll_label'].apply(list).to_list()

    predictions = reloaded_predictor.predict(
        tokens_per_sentence,
        custom_tokenizer=pass_thorough_tokenizer
    )

    predictions_flat = sum(predictions, [])

    prediction_tokens, prediction_labels = zip(*predictions_flat)

    test_gmb_df['token_pred'] = prediction_tokens  # just for assertion
    test_gmb_df['conll_label_pred'] = prediction_labels  # todo rename conll to gmb

    # save prediction results
    test_gmb_df.to_pickle(config.test_data_gmb_predictions_path)

    test_pred = test_gmb_df.groupby(
        by=[
            'document_id',
            'sentence_id_doc'
        ]
    )['conll_label_pred'].apply(list).to_list()

    failed_filter = test_gmb_df.conll_label != test_gmb_df.conll_label_pred
    failed_predictions = test_gmb_df[failed_filter]

    test_segmented_df = pd.read_pickle(config.test_data_segmented_path)

    # print(test_segmented_df[test_segmented_df.document_id == 11967].target_text.values[0])

    metrics.f1_score(target_labels, test_pred)
    metrics.accuracy_score(target_labels, test_pred)
    metrics.precision_score(target_labels, test_pred)
    metrics.recall_score(target_labels, test_pred)
    metrics.performance_measure(target_labels, test_pred)
    print(
        metrics.classification_report(target_labels, test_pred)
    )

    y_true = sum(target_labels, [])
    y_pred = sum(test_pred, [])
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    def plot_confusion_matrix(y_true, y_pred, labels, do_normalize=False):
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )

        if do_normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cmd = ConfusionMatrixDisplay(
            cm,
            display_labels=labels
        )

        cmd.plot(xticks_rotation=20)


    labels = list(set(y_true))
    plot_confusion_matrix(y_true, y_pred, labels)
    plot_confusion_matrix(y_true, y_pred, labels, True)

    y_true_no_prefix = [token.replace('I-', '').replace('B-', '') for token in y_true]
    y_pred_no_prefix = [token.replace('I-', '').replace('B-', '') for token in y_pred]

    labels_no_prefix = list(set(y_true_no_prefix))
    plot_confusion_matrix(y_true_no_prefix, y_pred_no_prefix, labels_no_prefix)
    plot_confusion_matrix(y_true_no_prefix, y_pred_no_prefix, labels_no_prefix, True)
