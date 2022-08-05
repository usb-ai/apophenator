import pandas as pd
import utils
from datetime import datetime
import config

'''
┌────────────┬───────────────┬─────────────────────┬────────────────────────────────┬─────────────────────────────┬─────────────────────────────┬─────────────────────────────┬──────────────────────┬──────────────────────────────────────────┐
│            │ DataFrame     │                     │ Index                          │                             │                             │                             │                      │                                          │
├────────────┼───────────────┼─────────────────────┼────────────────────────────────┼─────────────────────────────┼─────────────────────────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
│ B-         │ df_preds_all  │ df_preds_without_o  │ unique_document_ids_without_o  │ unique_document_ids_with_b  │                             │                             │ unique_document_ids  │                                          │
├────────────┼───────────────┼─────────────────────┼────────────────────────────────┼─────────────────────────────┼─────────────────────────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
│ B- and I-  │ df_preds_all  │ df_preds_without_o  │ unique_document_ids_without_o  │ unique_document_ids_with_b  │                             │                             │ unique_document_ids  │ unique_document_ids_with_b_and_faulty_i  │
├────────────┼───────────────┼─────────────────────┼────────────────────────────────┼─────────────────────────────┼─────────────────────────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
│ I-         │ df_preds_all  │ df_preds_without_o  │ unique_document_ids_without_o  │                             │                             │ unique_document_ids_only_i  │ unique_document_ids  │                                          │
├────────────┼───────────────┼─────────────────────┼────────────────────────────────┼─────────────────────────────┼─────────────────────────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────────────────┤
│ O-         │ df_preds_all  │                     │                                │                             │ unique_document_ids_only_o  │                             │ unique_document_ids  │                                          │
└────────────┴───────────────┴─────────────────────┴────────────────────────────────┴─────────────────────────────┴─────────────────────────────┴─────────────────────────────┴──────────────────────┴──────────────────────────────────────────┘
'''


def get_prediction_data():
    print('Reading all_predictions_final pickle...', end='')
    start_time = datetime.now()
    df_preds_all = pd.read_pickle(config.all_predictions_final)
    print('done, Duration: ', datetime.now()-start_time)

    if config.debug:
        print('DEBUG: filter to reduce the amount of data')
        document_ids = df_preds_all.document_id.unique()[:1000]
        df_preds_all = df_preds_all[df_preds_all.document_id.isin(document_ids)]

    return df_preds_all


def remove_o_prediction_rows(df):
    # This effectively also removes documents that have no predictions.
    print('Getting rows without "O-"...', end='')
    start_time = datetime.now()
    df_preds_without_o = df[df.pred != 'O']
    print('done, Duration: ', datetime.now()-start_time)
    return df_preds_without_o


if __name__ == '__main__':
    df_preds_all = get_prediction_data()

    print('Getting all doc ids...', end='')
    unique_document_ids = df_preds_all.document_id.unique()
    print('done')

    df_preds_without_o = remove_o_prediction_rows(df_preds_all)

    print('Getting doc ids where doc has at least one "B-" or "I-"...', end='')
    unique_document_ids_without_o = df_preds_without_o['document_id'].unique()
    print('done')

    print('Getting rows with only "B-"...', end='')
    df_preds_with_b = df_preds_without_o[df_preds_without_o.pred.str.startswith('B')]
    print('done')

    print('Getting doc ids where doc has at least one "B-" but contains possibly "I-"...', end='')
    unique_document_ids_with_b = df_preds_with_b.document_id.unique()
    print('done')

    print('Getting doc ids only "O-"...', end='')
    unique_document_ids_only_o = df_preds_all[~df_preds_all.document_id.isin(unique_document_ids_without_o)].document_id.unique()
    utils.save_to_pickle(unique_document_ids_only_o, config.document_ids_missing_label, 'list')
    print('done')

    print('Getting doc ids only "I-" (set)...', end='')
    start_time = datetime.now()
    unique_document_ids_only_i_set =  set(unique_document_ids_without_o)-set(unique_document_ids_with_b)
    print('done. Duration: ', datetime.now()-start_time)

    print('Find faulty predictions', end='')
    start_time = datetime.now()
    doc_has_b_filter = df_preds_all.document_id.isin(unique_document_ids_with_b)

    df_preds_with_b_i_o = df_preds_all[doc_has_b_filter].copy()

    df_preds_with_b_i_o['is_faulty'] = df_preds_with_b_i_o.pred.apply(
        utils.mark_predictions_without_b,
        previous={'label': 'O', 'is_faulty': False}
    )
    print('done. Duration: ', datetime.now()-start_time)

    # todo add faulty predictions from the O- if we are doing it an df_preds_all - at least for the future us-s
    # todo tbd if needed
    # unique_document_ids_with_b_and_faulty_i = None

    # filter None because <pad> exists as label
    df_preds_with_b_i_o = df_preds_with_b_i_o[~df_preds_with_b_i_o.is_faulty.isna()]

    print('Getting labels...')
    start_time = datetime.now()
    df_preds_with_b_i_o.sort_index(inplace=True)
    has_b_and_i_no_faulty = (~df_preds_with_b_i_o.is_faulty) & (df_preds_with_b_i_o.pred != 'O')
    df_all_predictions = utils.extract_labels( # this function is used in 7, better not touch the insides
        df_gmb=df_preds_with_b_i_o[has_b_and_i_no_faulty]
    )
    print('done. Duration: ', datetime.now()-start_time)

    utils.save_to_pickle(df_all_predictions, config.all_predicted_sequences, 'df')
