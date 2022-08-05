import pandas as pd
import config
import ktrain
import time
import spacy
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

class CustomTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sentence, *args, **kwargs):
        tokens = self.tokenizer(sentence)
        return [token.text for token in tokens]


def get_wv_model(preproc, force_download=True):
    file_path = config.root_dir / 'cc.de.300.vec'
    file_exists = file_path.is_file()

    print('Force download: {force_download}')
    print(f'Local file available: {file_exists}')
    print('Getting wv model ', end='')

    if not file_exists or force_download:
        print('from url.')
        wv_path_or_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz'
    else:
        print('from local file.')
        wv_path_or_url = file_path.as_posix()

    print('Creating tagger...', end='')
    model = ktrain.text.sequence_tagger(
        'bilstm-crf',
        preproc,
        wv_path_or_url=wv_path_or_url
    )
    print('done.')

    return model


def learner_fit(train_data, validation_data, wv_model, batch_size, lr, number_of_epochs, early_stopping):
    learner = ktrain.get_learner(
        wv_model,
        train_data=train_data,
        val_data=validation_data,
        batch_size=batch_size
    )
    t = time.time()
    hist = learner.fit(
        lr=lr,
        n_cycles=number_of_epochs,
        # cycle_len=None,
        # lr_decay=1e-3,
        early_stopping=early_stopping,
        # callbacks=[tb_call_back]
    )
    elapsed = time.time() - t
    print('Elapsed train time: ', elapsed)

    return learner, hist


def run_train(
        conll_dataframe_path,
        lr,
        number_of_epochs,
        batch_size=1024,
        early_stopping=3,
        use_char=False,
        validation_percentage=0.05
):
    print('Loading data...', end='')
    df = pd.read_pickle(conll_dataframe_path)
    print('done.')

    print('Converting to train/test data from df...', end='')
    (trn, val, preproc_model) = ktrain.text.entities_from_df(
        df,
        word_column='token',
        tag_column='conll_label',
        sentence_column='sentence_id_training',
        use_char=use_char,
        val_pct=validation_percentage,
        # verbose=1
    )
    print('done.')

    wv_model = get_wv_model(preproc_model, force_download=False)

    learner, hist = learner_fit(
        trn,
        val,
        wv_model,
        batch_size,
        lr,
        number_of_epochs,
        early_stopping
    )

    predictor = ktrain.get_predictor(learner.model, preproc_model)
    predictor.save(config.path_to_predictor_dir)

    score = learner.validate()
    # print('score:\n', score)
    # print('acc:\n', acc)

    learner.view_top_losses(n=10)

    # Testing reload
    reloaded_predictor = ktrain.load_predictor(config.path_to_predictor_dir)
    nlp = spacy.load('de_core_news_sm')
    custom_ner_tokenizer = CustomTokenizer(nlp.tokenizer)
    print(
        reloaded_predictor.predict('Gluten Tag.', custom_tokenizer=custom_ner_tokenizer)
    )


def grid_search(trn, val, preproc_model, number_of_epochs, early_stopping, max_retries=5):
    # parameters = {'batch_size': (1024, 2048), 'lr': [1e-1, 1e-2]}
    results = {
        'batch_size': [],
        'lr': [],
        'f1_micro_avg': [],
        # 'f1_macro_avg': [],
        # 'f1_weighted_avg': [],
        # 'precisson_micro_avg': [],
        # 'precisson_macro_avg': [],
        # 'precisson_weighted_avg': [],
        # 'recall_micro_avg': [],
        # 'recall_macro_avg': [],
        # 'recall_weighted_avg': [],
        # 'support': [],
        'acc': [],
    }

    for batch_size in (1, 2, 4):  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
        for lr in (1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-4, 1e-5):  # (1e-1, 1e-2) (1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-4, 1e-5)
            print(f'||batch size: {batch_size} || lr: {lr} ||')
            retry_count = 0
            while retry_count < max_retries:
                try:
                    wv_model = get_wv_model(preproc_model, True)

                    learner = learner_fit(
                            trn,
                            val,
                            wv_model,
                            batch_size,
                            lr,
                            number_of_epochs,
                            early_stopping
                        )

                    score, acc = learner.validate()

                    print('score:\n', score)
                    print('acc:\n', acc)

                    results['batch_size'].append(batch_size)
                    results['lr'].append(lr)
                    results['f1_micro_avg'].append(score)
                    results['acc'].append(acc)

                # todo not sure which exception it is
                # 19/84 [=====>........................] - ETA: 3s - loss: 0.22182021-10-11 20:45:58.179404: E tensorflow/stream_executor/cuda/cuda_event.cc:29] Error polling for event status: failed to query event: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
                except:
                    retry_count += 1
                    print(f'Trying again... ({retry_count})')
                    continue

                # Yes, it is overwriting the file all the time.
                # Cheaper than losing the data and easier than updating.
                pd.DataFrame(results).to_csv('grid_search.csv')

                break

    result_df = pd.DataFrame(results)
    result_df.to_csv('grid_search.csv')
    return result_df


if __name__ == '__main__':
    validation_percentage = 0.1
    early_stopping = 5
    lr = 1e-2
    batch_size = 1024
    max_epochs = 30
    use_char = False  # True will not work with TF2

    # df = pd.read_pickle(config.gmb_dataframe_path)

    # (trn, val, preproc_model) = ktrain.text.entities_from_df(
    #     df,
    #     word_column='token',
    #     tag_column='conll_label',
    #     sentence_column='sentence_id_training',
    #     use_char=True,
    #     val_pct=validation_percentage,
    #     # verbose=1
    # )

    print('get the goodies to stop here')

    # result_df = grid_search(trn, val, preproc_model, max_epochs, early_stopping)
    #
    # first = result_df.sort_values(by='f1_micro_avg', ascending=False).loc[0]
    #
    # lr = first.lr
    # batch_size = first.batch_size


    run_train(
        config.gmb_dataframe_path,
        lr=lr,
        number_of_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        use_char=use_char,
        validation_percentage=validation_percentage,
    )

    print('All done.')
