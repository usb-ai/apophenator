import pandas as pd
import config
import ktrain
from sklearn.model_selection import KFold


def get_wv_model(preproc, force_download=True):
    if force_download:
        wv_path_or_url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz'
    else:
        wv_path_or_url = (config.root_dir / 'cc.de.300.vec').as_posix()
    return ktrain.text.sequence_tagger(
        'bilstm-crf',
        preproc,
        wv_path_or_url=wv_path_or_url
    )


def run_crossvalidation(conll_dataframe_path, lr, number_of_epochs):
    kf = KFold(
        n_splits=5,
        random_state=42,
        shuffle=True
    )
    df = pd.read_pickle(conll_dataframe_path)
    unique_document_ids = df.document_id.unique()
    current_split = 0
    for train_indices, val_indices in kf.split(unique_document_ids):
        current_split += 1
        print('Current split: ', current_split)

        train_doc_ids = unique_document_ids[train_indices]
        val_doc_ids = unique_document_ids[val_indices]
        train_df = df[df.document_id.isin(train_doc_ids)]
        val_df = df[df.document_id.isin(val_doc_ids)]

        (trn, val, preproc_model) = ktrain.text.entities_from_df(
            train_df,
            val_df=val_df,
            word_column='token',
            tag_column='conll_label',
            sentence_column='sentence_id_training',
            use_char=False,
            val_pct=0.2,
            # verbose=1
        )

        wv_model = get_wv_model(preproc_model, True)

        learner = ktrain.get_learner(
            wv_model,
            train_data=trn,
            val_data=val,
            batch_size=1024
        )

        learner.fit(
            lr=lr,
            n_cycles=number_of_epochs,
            cycle_len=None,
            # lr_decay=1e-3,
            early_stopping=5,
        )

        score, acc = learner.validate()
        print('score:\n', score)
        print('acc:\n', acc)

        learner.view_top_losses(n=1)


if __name__ == '__main__':
    run_crossvalidation(config.gmb_dataframe_path, 1e-2, 30)
