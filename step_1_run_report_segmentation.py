import config
import pandas as pd
import re
from tqdm import tqdm
from collections import Counter
import utils


def filter_data(df):
    # Yesterday has been kicked out because there is not
    # enough support (number of samples is too low).

    df = df[
        (df.layer_name == 'temporal link') &
        (df.class_name != 'yesterday') &
        (df.class_name != 'missing')
        ]

    return df


if __name__ == '__main__':
    print('Segmenting all the data')
    df = pd.read_csv(config.raw_data_training_path)
    df_filtered = filter_data(df)

    df_segmented = utils.do_segmentation(df_filtered)

    print(
        'Getting data arbitrarily selected test dataset.'
        'This file is used only for filtering.'
    )
    noemi_test_df = pd.read_csv(config.test_data_out_of_thin_air)
    noemi_test_filter = df_segmented.text.isin(noemi_test_df.text)

    df_train_and_test = df_segmented[~noemi_test_filter]  # noemi's parts have no 'no previous' we have to remove it

    # unique_doc_ids = df_train_and_test.document_id.unique()
    # import numpy as np
    # np.random.choice(unique_doc_ids, 100)

    print('Creating train segmented dataset')
    df_train = df_train_and_test.sample(frac=0.95)
    df_train.to_pickle(config.train_data_segmented_path)

    print('Creating test segmented dataset')
    # df_test = df_segmented[noemi_test_filter]
    df_test = df_train_and_test.drop(df_train.index)
    df_test.to_pickle(config.test_data_segmented_path)

    print('All done!')
