import config

import utils


if __name__ == '__main__':
    # create gmb for train data
    train_conll_df = utils.create_gmb_file(
        data_pickle_path=config.train_data_segmented_path,
        output_path=config.gmb_dataframe_path,
    )

    # create one for the test
    test_conll_df = utils.create_gmb_file(
        data_pickle_path=config.test_data_segmented_path,
        output_path=config.test_data_conll_path,
    )
