import pandas as pd
import numpy as np
import json
import ast
import typing
import utils


class XTagPreprocessor:

    @staticmethod
    def get_unique_documents(
            dataset: pd.DataFrame,
            include_metadata: bool = True,
            document_id_as_index: bool = True) -> pd.DataFrame:

        column_subset = ['document_id', 'text']
        if include_metadata:
            column_subset.append('metadata')
        # remove duplicates, we only need the documents
        df_unique_docs = dataset.drop_duplicates(
            subset=['document_id'],
        )[column_subset]
        if document_id_as_index:
            df_unique_docs.set_index(keys=['document_id'], inplace=True)
        return df_unique_docs

    @staticmethod
    def get_metadata_as_df(unique_docs: pd.DataFrame, drop_metadata_column: bool = True) -> pd.DataFrame:
        df_metadata = pd.json_normalize(
            [json.loads(json_string) for json_string in unique_docs['metadata'].tolist()]
        )
        df_metadata.index = unique_docs.index
        if drop_metadata_column:
            unique_docs.drop(columns=['metadata'], inplace=True)
        return df_metadata

    @staticmethod
    def get_series_as_df(metadata: pd.DataFrame, drop_inplace: bool = True) -> pd.DataFrame:
        metadata['_childDocuments_'] = metadata['_childDocuments_'].apply(
            lambda x: x if utils.is_valid_json_string(x) else ast.literal_eval(x)
            # ast.listeral_eval creates a valid list of valid json objects
            # TODO rethink is_valid_json_string as check!
            # TODO why did I use that (json.dumps(ast.literal_eval(x)) before
        )
        df_series = pd.DataFrame()
        df_series['_childDocuments_'] = metadata['_childDocuments_']

        if drop_inplace:
            metadata.drop(columns=['_childDocuments_'], inplace=True)

        df_series = df_series.explode('_childDocuments_')
        series_df_index = df_series.index
        df_series = pd.json_normalize(df_series['_childDocuments_'].values)  #TODO why is it needed to pass nd.array values?
        df_series.index = series_df_index
        return df_series

    @staticmethod
    def get_binary_wide_format(dataset: pd.DataFrame, columns, unstack_cycles: int = 1):
        subset = ['document_id']

        if isinstance(columns, list) and all(isinstance(i, str) for i in columns):
            subset.extend(columns)
        elif isinstance(columns, str):
            subset.append(columns)

        # remove duplicates since we want to have binary values
        x = dataset.drop_duplicates(subset=subset)[subset]
        grouped = x.groupby(subset).size()

        max_level = len(subset) - 1
        if unstack_cycles > max_level:
            unstack_cycles = max_level

        for i in range(0, unstack_cycles):
            grouped = grouped.unstack()

        return grouped.fillna(0).astype('int')

    @staticmethod
    def get_overlapping_sequence_labels():
        # TODO
        return None

