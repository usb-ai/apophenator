import pandas as pd
import config
import ktrain


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


def get_lr(conll_dataframe_path):
    df = pd.read_pickle(conll_dataframe_path)
    (trn, val, preproc_model) = ktrain.text.entities_from_df(
        df,
        val_df=None,
        word_column='token',
        tag_column='conll_label',
        sentence_column='sentence_id_training',
        use_char=False,
        val_pct=0.1,
    )

    wv_model = get_wv_model(preproc_model, True)

    learner = ktrain.get_learner(
        wv_model,
        train_data=trn,
        val_data=val,
        batch_size=2048
    )

    learner.lr_find(stop_factor=13)
    learner.lr_plot()


if __name__ == '__main__':
    get_lr(config.gmb_dataframe_path)
