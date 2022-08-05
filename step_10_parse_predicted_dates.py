import config
import utils
import pandas as pd
import datetime
import datefinder  # TODO we are importing a git version of the datefinder
import dateparser as dp
from tqdm import tqdm
from dateparser.date import DateDataParser

# todo fix dates like 14.12.16
# Progress_apply comes from tqdm, it's the same as apply just with progress bar.
tqdm.pandas()


def get_data():
    print('***get_data***')

    print(
        f'[{datetime.datetime.now()}] '
        f'Loading the predicted dataframe and assigning report dates.'
    )

    # print('this is not ok, change code!!!') # todo remove
    df_preds_all = utils.read_df_from_pickle(config.all_predicted_sequences_with_modality)
    # print('we cannot work like this') # todo remove
    # df_preds_all = df_preds_all.iloc[:20000]  # todo delete this!

    df_preds_all.labeled_text = df_preds_all.labeled_text.str.strip()

    print('Dropping rows where predictions are effectively empty.')
    # empty_prediction_binary_index = df_preds_all.labeled_text.str.fullmatch('[\r\n]+|\s') # .index

    #df_preds_all.labeled_text.replace('[\r\n]+|\s','', regex=True, inplace=True)

    # non_empty_predictions_filter = df_all_predicted_sequences.labeled_text != ''
    empty_label_filter = ~df_preds_all.labeled_text.apply(bool)
    empty_label_indices = df_preds_all[empty_label_filter].index
    df_preds_all.drop(empty_label_indices, inplace=True)

    # df_preds_all = df_all_predicted_sequences.loc[non_empty_predictions_filter]  # TODO loc will make a copy

    current_time = datetime.datetime.now()
    print(f'[{current_time}] Creating report date series')
    print('    Reading segmented data pickle...', end='')
    df_all_reports_segmented = pd.read_pickle(config.all_reports_segmented_pickle)
    print("done.")
    # df_all_reports_segmented['examination_date'] = df_all_reports_segmented['examination_date'].apply(str)

    start_time = datetime.datetime.now()
    print('    Removing excess data and converting dates from type int to date...', end='')
    report_date_series = (
        df_all_reports_segmented[['document_id', 'examination_date']]
        .drop_duplicates('document_id')
        .set_index('document_id')
        ['examination_date']
        .apply(utils.int2date)
    )
    print('done.\n    Total duration: ', datetime.datetime.now() - start_time)


    print('Assigning report dates from the original data.')
    return df_preds_all.merge(report_date_series, left_on='document_id', right_index=True)


def assign_dates(preds_and_dates_df):
    print('***assign_dates***')
    today_predicitons_filter = preds_and_dates_df.label == 'today'
    yesterday_predicitons_filter = preds_and_dates_df.label == 'yesterday'
    no_previous_filter = preds_and_dates_df.label == 'no previous'
    date_filter = preds_and_dates_df.label == 'date'

    assert (len(preds_and_dates_df) - sum(today_predicitons_filter) - sum(yesterday_predicitons_filter) - sum(
        no_previous_filter) - sum(date_filter)) == 0

    preds_and_dates_df['target_date'] = None
    # preds_and_dates_df.target_date[today_predicitons_filter] = preds_and_dates_df.examination_date.loc[today_predicitons_filter]
    preds_and_dates_df.loc[today_predicitons_filter, 'target_date'] = preds_and_dates_df.examination_date[today_predicitons_filter]

    # preds_and_dates_df.target_date[yesterday_predicitons_filter] = preds_and_dates_df.examination_date[yesterday_predicitons_filter] - datetime.timedelta(1)
    preds_and_dates_df.loc[yesterday_predicitons_filter, 'target_date'] = preds_and_dates_df.examination_date[yesterday_predicitons_filter] - datetime.timedelta(1)

    # preds_and_dates_df.target_date[no_previous_filter] = 'no previous'
    preds_and_dates_df.loc[no_previous_filter, 'target_date'] = preds_and_dates_df.examination_date[no_previous_filter]  # 'no previous' # todo see if setting todays date for this makes sense

    # checking that nothing slipped through - only dates can have NA at this point.
    print(
        'Asserting that only class "date" has NA. '
        'Dates that could not be converted because they do not have '
        'a complete day, month and year structure. ', end=''
    )
    assert preds_and_dates_df.target_date[date_filter].isna().sum() == preds_and_dates_df.target_date.isna().sum()
    print('Passed.')

    return preds_and_dates_df


def parse_dmy_dates(df):
    print('***parse_dmy_dates***')
    print(
        f'[{datetime.datetime.now()}]'
        f'Parsing labeled text to dates (dateparser).'
    )

    na_filter = df.target_date.isna()  # TODO check that it is 806505 !!!

    # By using DateDataParser like this, it is not being created on every apply. Speeds up code by 20-30%.
    parser = DateDataParser(
        languages=['de'],
        settings={'REQUIRE_PARTS': ['day', 'month', 'year']}
    )
    # data = parser.get_date_data(date_string, date_formats=['%Y%m%d'])

    # preds_and_dates_df = utils.read_df_from_pickle(config.parsed_dmy_dates)
    # if preds_and_dates_df is not None:
    df.loc[na_filter, 'target_date'] = df.labeled_text[na_filter].progress_apply(
        parser.get_date_data,
        date_formats=['%Y%m%d']
    ).progress_apply(lambda x: x['date_obj'])

    print(
        'Total labeled dates still unresolved:',
        df.target_date[na_filter].isna().sum(),
        'out of',
        (df.label == 'date').sum()
    )

    print('Saving ymd parsed dates...', end='')
    utils.save_to_pickle(df, config.parsed_dmy_dates, 'df')
    print('done.')

    return df


def parse_md_dates(preds_and_dates_df):
    print('***parse_md_dates***')

    unresolved_filter = preds_and_dates_df.target_date.isna()

    start_time = datetime.datetime.now()
    print(f'[{start_time}] Parsing month_day dates...', end='')
    preds_and_dates_df['md'] = None

    parser = DateDataParser(
        languages=['de'],
        settings={
            'REQUIRE_PARTS': ['day', 'month'],
            'RELATIVE_BASE': datetime.datetime(1900, 1, 1),
        }
    )

    preds_and_dates_df.loc[unresolved_filter, 'md'] = preds_and_dates_df.labeled_text[unresolved_filter].progress_apply(
        parser.get_date_data,
    ).progress_apply(lambda x: x['date_obj'])

    print("Years are being set to the current year.")
    resolved_md_filter = unresolved_filter & preds_and_dates_df.md.notna()


    #todo check 1896-02-29 dates, these are 29.2.1900, but since impossible they are set as 1986
    def try_replace(arg_date, replacement_year):
        try:
            return arg_date.replace(year=replacement_year)
        except ValueError:
            pass  # ignoring the impossible conversion to e.g. 29.2.2014
        return None

    preds_and_dates_df.loc[resolved_md_filter, 'md'] = preds_and_dates_df[resolved_md_filter].progress_apply(
        lambda x:
            try_replace(x.md, x.examination_date.year),
        axis=1
    )

    #     preds_and_dates_df[resolved_md_filter].progress_apply(
    #     lambda x: x.md.replace(year=x.examination_date.year),
    #     axis=1,
    # )
    #
    preds_and_dates_df.target_date.fillna(preds_and_dates_df['md'], inplace=True)

    print(
        'Total labeled dates still unresolved:',
        preds_and_dates_df.target_date.isna().sum(),
        'out of',
        (preds_and_dates_df.label == 'date').sum()
    )

    # resolves most ^^, but stuff like 15.07. are missed
    # asd1 = preds_and_dates_df[unresolved_filter] # todo remove this, just for debug

    return preds_and_dates_df


def custom_md_parser(preds_and_dates_df):
    print('***custom_md_parser***')
    print('Custom parsing month-day dates.')

    custom_unresolved_filter = preds_and_dates_df.target_date.isna() & preds_and_dates_df.md.isna()

    parser = DateDataParser(
        languages=['de'],
        settings={
            'REQUIRE_PARTS': ['day', 'month'],
            'RELATIVE_BASE': datetime.datetime(1900, 1, 1),
            'PARSERS': ['custom-formats'],
            # 'NORMALIZE': True,
        }
    )
    date_formats = [
        '%m.%d.',
        '%m.%d',
        '%m/%d',
        '%m/%d/',
        '%B%d',
        '%d%B',
        '%d.%B',
        '%d/%B',
        '%B/%d',
        '%d-%B',
        '%B-%d',

    ]
    preds_and_dates_df.loc[custom_unresolved_filter, 'md_custom'] = (
        preds_and_dates_df
        .labeled_text[custom_unresolved_filter]
        .progress_apply(
            parser.get_date_data,
            date_formats=date_formats
        ).progress_apply(lambda x: x['date_obj']))

    print("Years are being set to the current year.")
    resolved_custom_md_filter = custom_unresolved_filter & preds_and_dates_df.md_custom.notna()
    preds_and_dates_df.loc[custom_unresolved_filter, 'md_custom'] = preds_and_dates_df[resolved_custom_md_filter].progress_apply(
        lambda x: x.md_custom.replace(year=x.examination_date.year),
        axis=1
    )

    preds_and_dates_df.target_date.fillna(preds_and_dates_df['md_custom'], inplace=True)

    print(
        'Total labeled dates still unresolved:',
        preds_and_dates_df.target_date.isna().sum(),
        'out of',
        (preds_and_dates_df.label == 'date').sum()
    )
    return preds_and_dates_df


def custom_parse_ym_dates(preds_and_dates_df):
    print('***custom_parse_ym_dates***')
    print('Custom parsing year-month dates, these are gonna be ambiguous.')

    unresolved_filter = preds_and_dates_df.target_date.isna()

    parser = DateDataParser(
        languages=['de'],
        settings={
            'REQUIRE_PARTS': ['year', 'month'],
            'RELATIVE_BASE': datetime.datetime(1900, 1, 1),
            'PARSERS': ['custom-formats'],
            # 'NORMALIZE': True,
        }
    )
    date_formats = [
        '%m.%Y',
        '%m %y',
        '%Y.%m.',
        '%Y %m.',
        '%B %Y',
        '%Y %B',
        '%Y.%B',
        '%Y. %B',
        '%Y,%B',
        '%Y, %B',
    ]
    preds_and_dates_df.loc[unresolved_filter, 'ym'] = (
        preds_and_dates_df
        .labeled_text[unresolved_filter]
        .progress_apply(
            parser.get_date_data,
            date_formats=date_formats
        ).progress_apply(lambda x: x['date_obj']))

    preds_and_dates_df.target_date.fillna(preds_and_dates_df['ym'], inplace=True)

    print(
        'Total labeled dates still unresolved:',
        preds_and_dates_df.target_date.isna().sum(),
        'out of',
        (preds_and_dates_df.label == 'date').sum()
    )

    return preds_and_dates_df


def custom_ymd_parser(preds_and_dates_df):
    print('***custom_ymd_parser***')
    unresolved_filter = preds_and_dates_df.target_date.isna()

    parser = DateDataParser(
        languages=['de'],
        settings={
            'REQUIRE_PARTS': ['year', 'month', 'day'],
            'RELATIVE_BASE': datetime.datetime(1900, 1, 1),
            'PARSERS': ['custom-formats'],
        }
    )

    date_formats = [
        '%d.%m,%Y',
        '%d%m.%Y',
    ]

    preds_and_dates_df.loc[unresolved_filter, 'ymd'] = (
        preds_and_dates_df
        .labeled_text[unresolved_filter]
        .progress_apply(
            parser.get_date_data,
            date_formats=date_formats
        ).progress_apply(lambda x: x['date_obj']))

    preds_and_dates_df.target_date.fillna(preds_and_dates_df['ymd'], inplace=True)

    print(
        'Total labeled dates still unresolved:',
        preds_and_dates_df.target_date.isna().sum(),
        'out of',
        (preds_and_dates_df.label == 'date').sum()
    )

    return preds_and_dates_df


def parse_with_datefinder(preds_and_dates_df):
    print('***parse_with_datefinder***')
    print('Applying datefinder on rest...', end='')
    unresolved_filter = preds_and_dates_df.target_date.isna()
    d = datefinder.DateFinder(
        base_date=datetime.datetime(1900, 1, 1),
        first='day'
    )

    def try_find_dates(arg_date):
        l = []
        gen = d.find_dates(arg_date)
        while True:
            try:
                v = next(gen)
                l.append(v)
            except TypeError : #calendar.IllegalMonthError:
                return None
            except StopIteration:
                return l

    # try_find_dates('17.05 .. 2018')


    # this will fail '17.05 .. 2018'
    preds_and_dates_df.loc[unresolved_filter, 'other'] = preds_and_dates_df.labeled_text[unresolved_filter].progress_apply(
        try_find_dates # d.find_dates
    ) # .progress_apply(list)

    start_time = datetime.datetime.now()
    print(f'[{start_time}] Exploding datefinder results...', end='')
    preds_and_dates_df = preds_and_dates_df.explode('other')
    print('done. Duration: ', datetime.datetime.now() - start_time)

    # Logic behind this:
    # Dates before 1.1.1900 are impossible in any way.
    # At this stage most 1.1.1900 are hours converted to dates.
    before_legal_time_indices = preds_and_dates_df[preds_and_dates_df.other < datetime.datetime(1900, 1, 2)].index
    preds_and_dates_df.drop(before_legal_time_indices, inplace=True)
    print(f'Dropped {len(before_legal_time_indices)} of entries with year before 1900')

    after_legal_time_indices = preds_and_dates_df[preds_and_dates_df.other > datetime.datetime(2022, 1, 1)].index
    preds_and_dates_df.drop(after_legal_time_indices, inplace=True)
    print(f'Dropped {len(after_legal_time_indices)} of entries with year before 2022')

    #todo are there years after 2021?

    # We convert 1900 to the report's year.
    print("1900 years are being set to the current year.")
    year_1900_filter = (preds_and_dates_df.other[preds_and_dates_df.other.notna()].apply(lambda x: x.year == 1900)) # todo this also looks bad, no time to fix atm
    year_1900_indices = preds_and_dates_df[preds_and_dates_df.other.notna()][year_1900_filter].index  # TODO why did I use index here? No need I'd say. Hm I use it for loc.

    # preds_and_dates_df.other.loc[preds_and_dates_df.other.notna(), other] = preds_and_dates_df.other[preds_and_dates_df.other.notna() & (preds_and_dates_df.other.apply(lambda x: x.year) == 1900)].apply(lambda x: x.year == 1900)
    # resolved_custom_md_filter = custom_unresolved_filter & preds_and_dates_df.md_custom.notna()
    preds_and_dates_df.loc[year_1900_indices, 'other'] = preds_and_dates_df.loc[year_1900_indices].progress_apply(
        lambda x: x.other.replace(year=x.examination_date.year),
        axis=1
    )

    preds_and_dates_df.target_date.fillna(preds_and_dates_df['other'], inplace=True)

    print(
        'Total labeled dates still unresolved:',
        preds_and_dates_df.target_date.isna().sum(),
        'out of',
        (preds_and_dates_df.label == 'date').sum()
    )

    return preds_and_dates_df


def comma_separated_last_piece_of_magic(preds_and_dates_df):
    print('Making comma-filter...', end='')
    start_time = datetime.datetime.now()
    comma_separated_date_filter = (
            preds_and_dates_df.target_date.isna()
            & (preds_and_dates_df.labeled_text.apply(len) > 4)
            & preds_and_dates_df.labeled_text.str.contains(',')
    )
    print('done. Duration: ', datetime.datetime.now() - start_time)

    print('Splitting comma-separated...', end='')
    start_time = datetime.datetime.now()
    preds_and_dates_df['comma_separated'] = (
        preds_and_dates_df[comma_separated_date_filter].labeled_text
            .str.replace(' ', '')
            .str.split(',')
    )
    print('done. Duration: ', datetime.datetime.now() - start_time)

    print('Exploding comma-separated...', end='')
    start_time = datetime.datetime.now()
    df_preds = preds_and_dates_df.explode('comma_separated').reset_index()
    print('done. Duration: ', datetime.datetime.now() - start_time)

    print('Parsing comma-separated (dateparser)...', end='')
    start_time = datetime.datetime.now()
    parser = DateDataParser(
        languages=['de'],
        settings={
            'REQUIRE_PARTS': ['month', 'year'],
            'RELATIVE_BASE': datetime.datetime(1900, 1, 1),
            'DATE_ORDER': 'DMY'
        }
    )

    comma_filter = df_preds.comma_separated.notna()
    df_preds.loc[comma_filter, 'comma_separated_parsed'] = (
        df_preds.comma_separated[comma_filter]
            .progress_apply(parser.get_date_data)
            .progress_apply(lambda x: x['date_obj'])
    )
    print('done. Duration: ', datetime.datetime.now() - start_time)

    df_preds['target_date'].fillna(df_preds['comma_separated_parsed'], inplace=True)

    print(
        'Total labeled dates still unresolved:',
        preds_and_dates_df.target_date.isna().sum(),
        'out of',
        (preds_and_dates_df.label == 'date').sum()
    )
    return df_preds


def remove_impossible_entries(df):
    # TODO check for impossible -> refferencig future dates!
    # todo if year 2022 -> date useless
    # todo if date before 1900 it is wrong
    # todo dates that are 1.1.1900 are most probably wrong
    pass


if __name__ == '__main__':
    preds_and_dates_df = utils.read_df_from_pickle(config.parsed_dmy_dates)
    # TODO if file exists skip this part
    if preds_and_dates_df is None:
        preds_and_dates_df = get_data()
        preds_and_dates_df = assign_dates(preds_and_dates_df)
        preds_and_dates_df = parse_dmy_dates(preds_and_dates_df)  # this is a long one

    preds_and_dates_df = parse_md_dates(preds_and_dates_df)
    preds_and_dates_df = custom_md_parser(preds_and_dates_df)
    preds_and_dates_df = custom_parse_ym_dates(preds_and_dates_df)
    preds_and_dates_df = custom_ymd_parser(preds_and_dates_df)
    preds_and_dates_df = parse_with_datefinder(preds_and_dates_df)
    preds_and_dates_df = comma_separated_last_piece_of_magic(preds_and_dates_df)


    utils.save_to_pickle(
        preds_and_dates_df[
            [
                'document_id',
                'labeled_text',
                'label',
                'target_date',
                'examination_date',  # todo proabbly won't need this, but what the hell
                'modality'
            ]
        ],
        config.reference_date_pickle, 'df'
    )
