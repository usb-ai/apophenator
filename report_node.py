
class ReportNode:
    def __init__(self, patient_id, accession_number, study_date, report_text='unknown', examination_type='unknown', region='unknown'):
        self.patient_id = patient_id
        self.accession_number = accession_number
        self.study_date = study_date
        self.report_text = report_text
        self.examination_type = examination_type
        self.region = region

    def get_report_id(self):
        return f'{self.patient_id}-{str(self.accession_number)}'