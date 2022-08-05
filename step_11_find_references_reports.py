import config
import utils
import pandas as pd


def remove_and_rename_columns(df, remove_n, new_names):
    # todo easier to just drop the merged ones than to look them up, feel free to fix.
    #  Hacky, I know. I only do this cause it is easier to work with normal names.
    #  I can later make a function out of it.

    print('removing extra columns and renaming')
    df.drop(df.columns[-remove_n:], axis=1, inplace=True)
    df.columns = new_names
    assert len(set(df.columns) - set(new_names)) == 0
    return df


def get_correct_and_incorrect_dataframes(df, remove_n, new_names):
    print('Cleaning up df and splitting in correct and mistargeted.')
    correct_filter = df.document_id_y.notna()  # don't remove the target document id duh
    print('Managed to reference ', correct_filter.sum(), 'references.')
    df_correct = df[correct_filter].copy()
    df_correct = remove_and_rename_columns(df_correct, remove_n-1, list(new_names) + [df_correct.columns[-remove_n]])

    mistargeted_filter = df.document_id_y.isna()
    print('Failed to reference ', mistargeted_filter.sum(), 'references.')
    df_mistargeted = df[mistargeted_filter].copy()
    df_mistargeted = remove_and_rename_columns(df_mistargeted, remove_n, new_names)

    return df_correct, df_mistargeted


if __name__ == '__main__':
    # todo if no previous refer to self?

    df = utils.read_df_from_pickle(config.reference_date_pickle)

    # We don't have any use from the predictions that do not resolve into  dates here.
    df = df[df.target_date.notna()]

    # I will get examination dates from the merge
    df.drop(['examination_date'], axis=1, inplace=True)

    df.modality.fillna('unknown', inplace=True)

    df.target_date = df.target_date.apply(
        lambda x: x.strftime('%Y%m%d'))  # todo why convert to string? I guess for neo4j

    df_segmented = utils.read_df_from_pickle(config.all_reports_segmented_pickle)

    # adjust df segmented formats
    df_segmented.examination_date = df_segmented.examination_date.apply(str)

    print(df_segmented.groupby('dicom_modality').document_id.count())

    df_segmented_complete = df_segmented

    df_segmented = df_segmented[['document_id', 'organ', 'dicom_modality', 'examination_date', 'patient_id']]

    df_merged_date_and_mod = df_segmented.merge(
        right=df,
        on=['document_id'],
        how='outer'
    )

    # just checking if the merge is correct
    assert df_merged_date_and_mod.label.notna().sum() == len(df)

    # just want to make sure all reports have organ
    assert not df_merged_date_and_mod.organ.isna().any()

    not_unknown_mod_filter = df_merged_date_and_mod.modality != 'unknown'  # todo this is the target mod, can be unknown again
    print('Not unknown_filter amount of unknown modalities ', not_unknown_mod_filter.sum())
    print('unknown_filter amount of unknown modalities ', (df_merged_date_and_mod.modality == 'unknown').sum())

    no_previous_filter = df_merged_date_and_mod.label != 'no previous'
    print('No previous count: ', no_previous_filter.sum())

    # No unknown mod targets, no 'no previous' and no the ones without label
    df_possible_references = df_merged_date_and_mod[
        df_merged_date_and_mod.label.notna()
        & no_previous_filter
        & not_unknown_mod_filter
        ]

    # there should be no targets that are na, since we dropped all the failed predictions
    assert df_possible_references.target_date.isna().sum() == 0

    print('The amount of possible targets: ', len(df_possible_references))

    print('Merging on patient, date and modality and organ -> a perfect match')
    df_perfect_merge = df_possible_references.merge(
        df_segmented,
        left_on=['target_date', 'patient_id', 'modality', 'organ'], # modality is target modality
        right_on=['examination_date', 'patient_id', 'dicom_modality', 'organ'], # dicom_modality comes from database
        how='left'
    )

    default_column_names = df_possible_references.columns
    df_correct, df_mistargeted = get_correct_and_incorrect_dataframes(df_perfect_merge, 3, default_column_names)

    print('Merging on patient, date and modality')
    df_merged_without_organ = df_mistargeted.merge(
        df_segmented,
        left_on=['target_date', 'patient_id', 'modality'],  # modality is target modality
        right_on=['examination_date', 'patient_id', 'dicom_modality'],  # dicom_modality comes from database
        how='left'
    )

    print('no organ stats')
    df_correct_without_organ, df_mistargeted_without_organ = get_correct_and_incorrect_dataframes(
        df_merged_without_organ,
        4,
        default_column_names
    )

    print('Merging on patient, date and organ')
    df_merged_without_mod = df_mistargeted_without_organ.merge(
        df_segmented,
        left_on=['target_date', 'patient_id', 'organ'],  # modality is target modality
        right_on=['examination_date', 'patient_id', 'organ'],  # dicom_modality comes from database
        how='left'
    )

    print('no mod stats')

    df_correct_without_mod, df_mistargeted_without_mod = get_correct_and_incorrect_dataframes(
        df_merged_without_mod,
        3,
        default_column_names
    )

    print('merging the rest, only on patient and date')
    # We have the same amount of extra cols as before without mod, the mod is NOT duplicated <- nost sure what this refers to

    # todo check
    df_merged_without_mod_nor_organ = df_mistargeted_without_mod.merge(
        df_segmented,
        left_on=['target_date', 'patient_id'],  # modality is target modality
        right_on=['examination_date', 'patient_id'],  # dicom_modality comes from database
        how='left'
    )

    print('no mod stats')

    df_correct_without_mod_org, df_mistargeted_without_mod_org = get_correct_and_incorrect_dataframes(
        df_merged_without_mod_nor_organ,
        4,
        default_column_names
    )

    df_correct['merge_success_rate'] = 'all'
    df_correct_without_organ['merge_success_rate'] = 'modality'
    df_correct_without_mod['merge_success_rate'] = 'organ'
    df_correct_without_mod_org['merge_success_rate'] = 'date'
    # todo please rename dont be stupid
    the_ultimate = pd.concat([df_correct, df_correct_without_organ, df_correct_without_mod, df_correct_without_mod_org])
    # simple check the concat went the proper way
    assert len(the_ultimate) == sum(map(len, [df_correct, df_correct_without_organ, df_correct_without_mod, df_correct_without_mod_org]))
    # another check just in case
    assert len(the_ultimate[the_ultimate['merge_success_rate'] == 'all']) == len(df_correct)

    # todo check here
    xx = the_ultimate.drop_duplicates()
    g = xx.groupby(['document_id', 'dicom_modality', 'patient_id', 'modality', 'label', 'labeled_text', 'target_date', 'document_id_y', 'merge_success_rate'])

    gc = g.count()
    gc_df = gc[gc.organ > 1].reset_index()
    assert len(gc_df) == 0

    utils.save_to_pickle(the_ultimate, config.references_for_the_database, 'df')

    print('Ready to roll!')
