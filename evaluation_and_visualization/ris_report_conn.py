import config
import pandas as pd
from datetime import datetime
from mappers.modality_data import mod_dict
from mappers.region_data import region_dict
from sqlalchemy import create_engine
from striprtf.striprtf import rtf_to_text





def establish_connection(user, password):
    user_and_pw = user + ':' + password
    engine_str = 'hana://' + user_and_pw + '@ictthdbdwlp1.uhbs.ch:33241/?encrypt=true&sslValidateCertificate=false'
    engine = create_engine(engine_str)
    return engine.connect()




def create_doc_node():
    return




def convert_ris_to_patient_id(df):
    return df


def convert_ris_to_accession_nr(df):
    accession_nr = str(int(df))
    return accession_nr


def convert_ris_to_study_date(df):
    study_date = df.strftime("%Y%m%d")
    return study_date


def convert_ris_rtf_to_text(df):
    report_text = rtf_to_text(df)
    return report_text


def create_report_id(df):
    report_id = df.iloc[0] + '-' + df.iloc[1]
    return report_id


# todo: make it work, that all the accession numbers are being considered
def get_data_from_db(acc_nr_list):
    acc_nr_list = acc_nr_list[2]
    conn = establish_connection(config.db_user, config.db_pw)
    command = f'''
    SELECT 
        A_BEFUND.UNTERS_BEGINN, 
        A_BEFUND.UNTERS_SCHLUESSEL, 
        A_BEFUND_TEXT_RTF.BEFUND_TEXT, 
        A_PATIENT.PAT_PID_NUMMER, 
        A_UNTBEH_UEB.UNTERS_ART 
    FROM 
        RIS.A_BEFUND 
    INNER JOIN 
        RIS.A_BEFUND_TEXT_RTF 
    ON 
        A_BEFUND.BEFUND_SCHLUESSEL = A_BEFUND_TEXT_RTF.BEFUND_SCHLUESSEL 
    INNER JOIN 
        RIS.A_PATIENT 
    ON 
        A_BEFUND.PATIENT_SCHLUESSEL = A_PATIENT.PATIENT_SCHLUESSEL 
    INNER JOIN 
        RIS.A_UNTBEH_UEB 
    ON 
        A_BEFUND.UNTERS_SCHLUESSEL = A_UNTBEH_UEB.UNTERS_SCHLUESSEL
    WHERE
        A_BEFUND.UNTERS_SCHLUESSEL LIKE \'{acc_nr_list}\'
    '''
    df_data = pd.read_sql_query(command, conn)
    df_converted = pd.DataFrame()
    df_converted['patient_id'] = df_data['pat_pid_nummer'].apply(convert_ris_to_patient_id)
    df_converted['accession_nr'] = df_data['unters_schluessel'].apply(convert_ris_to_accession_nr)
    df_converted['study_date'] = df_data['unters_beginn'].apply(convert_ris_to_study_date)
    df_converted['text'] = df_data['befund_text'].apply(convert_ris_rtf_to_text)
    df_converted['report_id'] = df_converted.apply(create_report_id, axis=1)
    return df_converted
