import pandas as pd

import config
from py2neo import Graph, Node, Relationship
import report_node


def link_document_relations(df_data, g):
    df_data = df_data.dropna()
    home_doc_info = df_data.iloc[0]

    patient = Node("Patient", id=home_doc_info.patient_id)
    home_doc = Node(
        "Document",
        id=home_doc_info.accession_number,
        report_id=home_doc_info.get_report_id(),
        study_date=home_doc_info.study_date,
        type=home_doc_info.examination_type,
        region=home_doc_info.region,
        text=home_doc_info.report_text
    )

    tx = g.begin()
    tx.merge(patient, primary_label="Patient", primary_key="id")
    tx.merge(home_doc, primary_label="Document", primary_key="id")
    tx.merge(Relationship(patient, "HAS", home_doc))
    prev_doc_list = df_data.iloc[1:].tolist()
    for doc_info in prev_doc_list:
        doc = Node(
            "Document",
            id=doc_info.accession_number,
            report_id=doc_info.get_report_id(),
            study_date=doc_info.study_date,
            type=doc_info.examination_type,
            region=doc_info.region,
            text=doc_info.report_text
        )
        tx.merge(doc, primary_label="Document", primary_key="id")
        tx.merge(Relationship(patient, "HAS", doc))
        tx.merge(Relationship(home_doc, "REFERS TO", doc))
    g.commit(tx)
    print('neo4j updated')


def build_graph(df_predictions, df_document_metadata, df_previous):

    previous_document_list = []

    def populate_previous_nodes_list(df_row, previous_nodes: list):
        # todo decide what to do with None values -> not found in DB
        if df_row.accession_number is not None:
            doc = report_node.ReportNode(
                patient_id=df_row.patient_id,
                accession_number=df_row.accession_number,
                study_date=df_row.study_date,
                report_text=df_row.report_text,
                examination_type=df_row.examination_type,
                region=df_row.region
            )
            previous_nodes.append(doc)

    def f(df_row):
        current_doc = report_node.ReportNode(
            patient_id=df_row.PatientID,
            accession_number=df_row.AccessionNumber,
            study_date=df_row.StudyDate,
            # doc_text, doc_exam_type, doc_region)
        )
        df_previous_of_current_doc = df_previous[df_previous.document_id == df_row.name]

        previous_nodes_list = []

        df_previous_of_current_doc.apply(
            func=populate_previous_nodes_list,
            previous_nodes=previous_nodes_list,
            axis=1
        )

        previous_document_list.append([current_doc] + previous_nodes_list)

    df_document_metadata.apply(f, axis=1)

    return previous_document_list


def update_graph(relation_data):
    df_results = pd.DataFrame(relation_data)

    df_prev_docs = df_results[df_results.iloc[:, 1].notna()]  # todo why ?

    # create graph
    graph = Graph(uri=config.neo4j_uri, user=config.neo4j_user, password=config.neo4j_pw)
    graph.delete_all()
    df_prev_docs.apply(link_document_relations, args=[graph], axis=1)
