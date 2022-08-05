from py2neo import Graph, Schema
from py2neo.errors import ClientError
from py2neo import bulk
from tqdm import tqdm
from datetime import datetime
import config
import utils
from itertools import islice

# todo refactor some functions to use batcher
# todo eliminate set unique if it is already there, it is handled now with try-catch


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def bulk_create_relationships(graph, values, ref_name, start_node, end_node, batch_size=10000):
    batches_count = len(values) // batch_size + 1
    for value_batch in tqdm(
            batcher(values, batch_size),
            desc=f'Creating {len(values)} relationships {start_node}-[{ref_name}]->{end_node} in batches of {batch_size}',
            total=batches_count
    ):
        bulk.create_relationships(
            graph,
            value_batch,
            ref_name,
            start_node_key=start_node,
            end_node_key=end_node
        )


def set_constraint(node_name, unique_field, graph):
    print(f'Setting a constraint on {node_name} node and {unique_field} field: ', end='')
    # todo maybe there is a better way than try catch, but here it is, quick and dirty for now
    # todo use log for both of these
    try:
        res = graph.schema.create_uniqueness_constraint(node_name, unique_field)
        # print('success.')
        # print(res)
    except ClientError:
        pass  # print(f'failed.\nUnique constraint for node "{node_name}" on field "{unique_field}" already exists.')


def merge_nodes(node_name, id_key, keys, values, graph, verbose=True):
    assert id_key in keys

    set_constraint(node_name, id_key, graph)

    start_time = datetime.now()
    if verbose:
        print(f'[{start_time}] Adding {node_name} nodes...', end='')

    res = bulk.merge_nodes(
        graph.auto(),
        values,
        labels={node_name},
        keys=keys,
        merge_key=(node_name, id_key)
    )
    if verbose:
        print(f'done. Duration {datetime.now() - start_time}')
        print(res)


def add_patients(df, graph):
    # Properties need to be either integer or string to be indexed.
    # The patient ID is long, meaning we have to convert it to string
    unique_patients = map(lambda x: [str(x)], df.patient_id.unique())
    merge_nodes(
        'Patient',
        id_key='patient_id',
        keys=['patient_id'],
        values=unique_patients,
        graph=graph
    )


def add_modalities(df, graph):
    group_keys = df.groupby(['dicom_modality', 'dicom_modality_description']).groups.keys()
    values = list(group_keys)
    merge_nodes(
        'Modality',
        id_key='name',
        keys=['name', 'description'],
        values=values,
        graph=graph
    )


def add_organs(df, graph):
    group_keys = df.groupby(['organ', 'organ_description']).groups.keys()
    # checking if there are no duplicate organs due to some misspelling
    assert len(df.organ.unique()) == len(group_keys)

    values = list(group_keys)
    merge_nodes(
        'Organ',
        id_key='name',
        keys=['name', 'description'],
        values=values,
        graph=graph
    )


def add_examinations(df, graph):
    group_keys = df.groupby(['examination_type', 'exam_name', 'exam_description']).groups.keys()

    # checking if there are no duplicate examination types due to some misspelling
    assert len(df.examination_type.unique()) == len(group_keys)

    values = list(group_keys)
    merge_nodes(
        'Examination',
        id_key='examination_type',
        keys=['examination_type', 'exam_name', 'exam_description'],
        values=values,
        graph=graph
    )


def add_labels(graph):
    # todo not very dinamic this one, maybe fix with all predicitons df, maybe not
    # labels = map(lambda x: [str(x)], df.label.unique())
    merge_nodes(
        'Label',
        id_key='class',
        keys=['class'],
        values=[['date'], ['today'], ['yesterday'], ['no previous']],
        graph=graph
    )


def add_reports(df, graph, batch_size=10000):
    node_name = 'Report'
    columns = [
        'document_id',
        'case_id',
        'accession_id',
        'patient_id',
        'examination_date',
        'exam_name',
        'text',
        'finding',
        'impression',
        'finding_and_impression',
        'target_text',
        'bad_flag',
        'just_finding',
        'just_impression',
    ]
    values_df = df[columns]
    values = values_df.values.tolist()

    id_key = 'document_id'
    assert id_key in columns

    batches_count = len(values) // batch_size + 1
    for value_batch in tqdm(
            batcher(values, batch_size),
            desc=f'Creating {len(values)} nodes in batches of {batch_size}',
            total=batches_count
    ):
        merge_nodes(
            node_name=node_name,
            id_key=id_key,
            keys=columns,
            values=value_batch,
            graph=graph,
            verbose=False,
        )


def get_reference_info(x):
    return {
        'class': x.label,
        'quality': x.merge_success_rate,
        'predicted_modality': x.modality,
        'prediction': x.labeled_text,
        'parsed_date': x.target_date,
    }


def add_references(df, graph):

    df['reference_info'] = df.apply(get_reference_info, axis=1)
    df_subset = df[['document_id', 'reference_info', 'document_id_y']]
    values = df_subset.values.tolist()

    bulk.create_relationships(
        graph,
        values,
        "REFERS_TO",
        start_node_key=("Report", 'document_id'),
        end_node_key=('Report', 'document_id')
    )
    print('done with this as well')


def add_patient_references(df, graph):
    df.patient_id = df.patient_id.apply(str)  # todo this can be done sooner, since it will always need to be a str
    df['since'] = df.examination_date.apply(lambda x: {'since': str(x)})
    df_subset = df[['document_id', 'since', 'patient_id']]
    values = df_subset.values.tolist()

    bulk_create_relationships(
        graph,
        values,
        ref_name="BELONGS_TO",
        start_node=('Report', 'document_id'),
        end_node=('Patient', 'patient_id')
    )


def add_organ_reference(df, graph):
    df['organ_ref_detail'] = df.organ.apply(lambda x: {'type': x})
    values_df = df[['document_id', 'organ_ref_detail', 'organ']]
    values = values_df.values.tolist()
    bulk_create_relationships(
        graph,
        values,
        ref_name="OF_BODY_PART",
        start_node=('Report', 'document_id'),
        end_node=('Organ', 'name')
    )


def add_exam_reference(df, graph):
    df['exam_ref_detail'] = df.examination_date.apply(lambda x: {'examination_date': str(x)})
    values_df = df[['document_id', 'exam_ref_detail', 'examination_type']]
    values = values_df.values.tolist()
    bulk_create_relationships(
        graph,
        values,
        ref_name="IS_OUTPUT_OF_EXAM",
        start_node=('Report', 'document_id'),
        end_node=('Examination', 'examination_type')
    )


def add_modality_reference(df, graph):
    df['since'] = df.examination_date.apply(lambda x: {'since': str(x)})
    values_df = df[['document_id', 'since', 'dicom_modality']]
    values = values_df.values.tolist()
    bulk_create_relationships(
        graph,
        values,
        ref_name="IS_MODALITY",
        start_node=('Report', 'document_id'),
        end_node=('Modality', 'name')
    )


def add_label_reference(df, graph):
    df['ref_info'] = df.labeled_text.apply(lambda x: {'labeled_text': x})
    values_df = df[['document_id', 'ref_info', 'label']]
    values = values_df.values.tolist()
    bulk_create_relationships(
        graph,
        values,
        ref_name="HAS_REFERENCE_CLASS",
        start_node=('Report', 'document_id'),
        end_node=('Label', 'class')
    )


if __name__ == '__main__':
    graph = Graph(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_pw
    )

    # graph.delete_all()  # todo this will not work on the amount of data that we have

    schema = Schema(graph)

    all_reports_df = utils.read_df_from_pickle(config.all_reports_segmented_pickle)
    all_reports_df = all_reports_df[(20220101 > all_reports_df.examination_date) & (all_reports_df.examination_date > 20101231)]

    add_patients(all_reports_df, graph)
    add_modalities(all_reports_df, graph)
    add_organs(all_reports_df, graph)
    add_examinations(all_reports_df, graph)
    add_reports(all_reports_df, graph)

    add_labels(graph)

    references_df = utils.read_df_from_pickle(config.references_for_the_database)
    add_references(references_df, graph)
    add_label_reference(references_df, graph)

    predicted_df = utils.read_df_from_pickle(config.reference_date_pickle)
    no_previous_df = predicted_df[predicted_df.label == 'no previous']  # todo: make a copy of this or loc, could be an error
    # del predicted_df
    add_label_reference(no_previous_df, graph)

    add_patient_references(all_reports_df, graph)
    add_organ_reference(all_reports_df, graph)
    add_exam_reference(all_reports_df, graph)

    add_modality_reference(all_reports_df, graph)

    print("That's all folks!")
