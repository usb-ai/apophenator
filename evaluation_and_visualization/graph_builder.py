import config
from py2neo import Graph, Node, Relationship


def link_document_relations(df_data, g):
    df_data = df_data.dropna()
    home_doc_info = df_data.iloc[0]
    home_patient_id = home_doc_info.patient_id
    home_acc_nr = home_doc_info.accession_number
    home_study_date = home_doc_info.study_date
    home_type = home_doc_info.examination_type
    home_region = home_doc_info.region
    home_text = home_doc_info.report_text
    home_report_id = home_doc_info.get_report_id()

    patient = Node("Patient", id=home_patient_id)
    home_doc = Node(
        "Document",
        id=home_acc_nr,
        report_id=home_report_id,
        study_date=home_study_date,
        type=home_type,
        region=home_region,
        text=home_text
    )

    tx = g.begin()
    tx.merge(patient, primary_label="Patient", primary_key="id")
    tx.merge(home_doc, primary_label="Document", primary_key="id")
    tx.merge(Relationship(patient, "HAS", home_doc))
    prev_doc_list = df_data.iloc[1:].tolist()
    for doc_info in prev_doc_list:
        accession_nr = doc_info.accession_number
        study_date = doc_info.study_date
        report_id = doc_info.get_report_id()
        text = doc_info.report_text
        acc_type = doc_info.examination_type
        region = doc_info.region

        doc = Node(
            "Document",
            id=accession_nr,
            report_id=report_id,
            study_date=study_date,
            type=acc_type,
            region=region,
            text=text
        )
        tx.merge(doc, primary_label="Document", primary_key="id")
        tx.merge(Relationship(patient, "HAS", doc))
        tx.merge(Relationship(home_doc, "REFERS TO", doc))
    g.commit(tx)
    print('neo4j updated')


def update_graph_db(df_relation_data):
    # create graph
    graph = Graph(uri=config.neo4j_uri, user=config.neo4j_user, password=config.neo4j_pw)
    graph.delete_all()
    df_relation_data.apply(link_document_relations, args=[graph], axis=1)
