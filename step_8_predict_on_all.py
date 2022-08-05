import pandas as pd

import config
import utils
import ktrain
from datetime import datetime
import pickle
import itertools

from pass_through_tokenizer import PassThroughTokenizer


def get_reports():
    print('Step 1: get reports from csv file.')
    df = utils.read_df_from_pickle(config.all_reports_pickle)
    if df is not None:
        return df

    print('(Step 1) Loading data from the csv file...', end='')
    start_time = datetime.now()
    df = pd.read_csv(config.all_reports_path)
    df.columns = df.columns.str.lower()
    print("done. Duration:", datetime.now() - start_time)

    print("Filtering data between 2010 and 2021...", end="")
    start_time = datetime.now()
    df = df[(df.examination_date > 20100000) & (df.examination_date < 20220000)]
    print("done. Duration:", datetime.now() - start_time)

    if config.persist_output:
        utils.save_to_pickle(df, config.all_reports_pickle, 'df')

    return df


def get_segmented():
    print('Step 2: Segment reports - focusing primarily on impression if found.')

    df = utils.read_df_from_pickle(config.all_reports_segmented_pickle)

    if df is not None:
        return df

    df_all_reports = get_reports()

    print('(Step 2) Calculating...', end='')
    start_time = datetime.now()
    df = utils.do_segmentation(df_all_reports, do_drop_na=False)
    # target text can be none even if it is not bad flag, it could be just impression
    df.target_text.fillna('', inplace=True)
    df['document_id'] = (
            df['patient_id'].apply(str)
            + '_'
            + df['accession_id'].apply(str)
    )
    print("done. Duration:", datetime.now() - start_time)

    if config.persist_output:
        utils.save_to_pickle(df, config.all_reports_segmented_pickle, 'df')

    return df


def get_conll():
    print('Step 3: Convert to CoNLL without labels and sort.')

    df = utils.read_df_from_pickle(config.all_reports_empty_conll_pickle)
    if df is not None:
        return df

    df_segmented = get_segmented()

    print('(Step 3) Calculating...', end='')
    start_time = datetime.now()
    df = utils.create_conll_no_label_dataframe(df_segmented)
    print("done. Duration:", datetime.now() - start_time)

    # todo save this? or rather not?
    print('Sorting by doc id and sentence id...', end='')
    df = df.sort_values(
        by=[
            'document_id',
            'sentence_id_doc'
        ]
    )
    print('done.')

    if config.persist_output:
        utils.save_to_pickle(df, config.all_reports_empty_conll_pickle, 'df')

    return df


def get_tokenized_sentences():
    print('Step 4: Getting list of lists for tokens grouped by sentence.')

    if config.all_tokens_per_sentence.is_file():
        print(f'File {config.all_tokens_per_sentence} exists, loading...', end='')
        start_time = datetime.now()
        with open(config.all_tokens_per_sentence, "rb") as open_file:
            token_sentence_nested_list = pickle.load(open_file)
        print('done. Duration: ', datetime.now() - start_time)
        return token_sentence_nested_list

    df_all_reports_empty_conll = get_conll()
    start_time = datetime.now()
    print('(Step 4) Grouping data...', end='')
    tokens_per_sentence = df_all_reports_empty_conll.groupby(
        by=[
            'document_id',
            'sentence_id_doc'
        ]
    )['token'].apply(list).to_list()
    print('done. Duration: ', datetime.now() - start_time)

    if config.persist_output:
        utils.save_to_pickle(tokens_per_sentence, config.all_tokens_per_sentence, 'list')
        # print(f'Saving to {config.all_tokens_per_sentence}...', sep='')
        # with open(config.all_tokens_per_sentence, "wb") as open_file:
        #     pickle.dump(tokens_per_sentence, open_file)
        # print('done.')
    return tokens_per_sentence


def run_prediction():
    print('Step 5: Running prediction')

    if (not config.force_recalculate) and config.all_predictions.is_file():
        print('Predictions file exists, loading...', end='')
        with open(config.all_predictions, "rb") as open_file:
            # Loading the list, it is not df
            start_time = datetime.now()
            predictions = pickle.load(open_file)
            print('done - time taken:', datetime.now() - start_time)
        return predictions

    tokenized_sentences = get_tokenized_sentences()

    print('Total sentences to predict: ', len(tokenized_sentences))
    print('There should be roughly 10x more tokens.')

    reloaded_predictor = ktrain.load_predictor(
        config.path_to_predictor_dir,
        batch_size=config.batch_size
    )

    print('Predicting with batch size ', reloaded_predictor.batch_size)

    pass_thorough_tokenizer = PassThroughTokenizer()

    start_time = datetime.now()
    print("^--Start time:", start_time.strftime("%H:%M:%S"))
    predictions = reloaded_predictor.predict(
        tokenized_sentences,
        custom_tokenizer=pass_thorough_tokenizer
    )
    finish_time = datetime.now()
    print("^--Finish at:", finish_time.strftime("%H:%M:%S"))

    duration = finish_time - start_time
    print('   ^--Duration: ', duration)

    if config.persist_output:
        utils.save_to_pickle(predictions, config.all_predictions, 'list')

    return predictions


# def debug_predict(tokens_per_sentence):
#     print("Debug mode is ON - you know what you're doing, right?")
#     sample_size = int(1e6)
#     print('reducing print size to ', sample_size)
#     tokens_per_sentence = tokens_per_sentence[:sample_size]
#
#     _ = predict(tokens_per_sentence, 32)
#     _ = predict(tokens_per_sentence, 128)
#     _ = predict(tokens_per_sentence, 256)
#     _ = predict(tokens_per_sentence, 512)
#     _ = predict(tokens_per_sentence, 1024)
#     _ = predict(tokens_per_sentence, 2048)
#     _ = predict(tokens_per_sentence, 2048 + 512)


def unravel_lists_of_predictions():
    print(
        'Step 6: Unraveling nested lists. '
        'Output is a list of tokens and a list of predicted labels'
    )

    predictions = run_prediction()

    #todo should be fast, that's why it is not stored
    print('(Step 6) Flattening prediction results...', end='')
    start_time = datetime.now()
    predictions_flat = list(itertools.chain(*predictions))
    print('done. Duration (itertools):', datetime.now() - start_time)
    return predictions_flat


def get_predictions():
    print('Step 7: Formating predictions')

    if (not config.force_recalculate) and config.all_predictions_df.is_file():
        return utils.read_df_from_pickle(config.all_predictions_df)

    predictions_flat = unravel_lists_of_predictions()

    # At least 5x faster than zip
    print('Pivoting predictions (df)...', end='')
    start_time = datetime.now()
    df_preds = pd.DataFrame(predictions_flat, columns=['token', 'pred'])
    print('done. Duration: ', datetime.now() - start_time)

    if config.persist_output:
        utils.save_to_pickle(df_preds, config.all_predictions_df, 'df')

    return df_preds


def merge_input_with_predicitons():
    print('Step 8: Merging token info and the results.')
    predictions_df = get_predictions()

    conll_df = utils.read_df_from_pickle(config.all_reports_empty_conll_pickle)
    if conll_df is None:
        raise ValueError('CoNLL file has to exist, predictions are based on it.')

    if not config.debug:
        conll_df.drop(columns=['sentence_id_training', 'token'], inplace=True)

    predictions_df.set_index(conll_df.index, inplace=True)

    if config.debug:
        df_hc = conll_df.head(100)
        df_hp = predictions_df.head(100)
        df_hf = pd.concat([df_hc, df_hp], axis=1, keys=['CoNLL', 'PREDS'])
        print(df_hf)

    df = pd.concat([conll_df, predictions_df], sort=False, axis=1)
    if config.persist_output:
        utils.save_to_pickle(df, config.all_predictions_final, 'df')

    return df


def run():
    if config.force_recalculate:
        print('IMPORTANT: Force recalculate is ON')

    if not config.persist_output:
        print('IMPORTANT: Persistance is OFF. Everything not saved will be lost!')

    full_results_df = merge_input_with_predicitons()

    print(full_results_df.head(10))


if __name__ == '__main__':
    run()
    print('Amazing!')
