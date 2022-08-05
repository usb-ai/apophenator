import config
import numpy as np
import pandas as pd
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split, KFold


class_names = ['not_missing', 'missing']


def get_count(df):
    missing = df[df['class_name'] == 'missing'].shape[0]
    not_missing = df[df['class_name'] != 'missing'].shape[0]
    difference = not_missing - missing
    return missing, not_missing, difference


def prepare_missing(df, column):
    condition = df[column] != 'missing'
    df = df.copy()
    df.loc[:, 'missing'] = 1
    df.loc[:, 'not_missing'] = 0
    df['missing'].mask(condition, 0, inplace=True)
    df['not_missing'].mask(condition, 1, inplace=True)
    return df


def balance_drop(df):
    remove_n = get_count(df)[2]
    drop_indices = np.random.choice(df[df['class_name'] != 'missing'].index, remove_n, replace=False)
    df_drop_indices = df.loc[drop_indices, :]
    return df.drop(drop_indices), df_drop_indices


# get the dropped rows and duplicate random 'missing' to get a balanced data set (bootstrapping technique)
def balance_train_data(df, df_drop):
    train_df_unbalanced = pd.concat([df, df_drop.sample(int(len(df_drop.index)/2))])
    nr_of_duplication = get_count(train_df_unbalanced)[2]
    random_rows = train_df_unbalanced[train_df_unbalanced['missing'] == 1].sample(n=nr_of_duplication, replace=True)
    return pd.concat([train_df_unbalanced, random_rows]).sample(frac=1)


def prediction(predictor, df, column):
    predictions = predictor.predict(df[column].to_list())
    prediction_df = pd.DataFrame(predictions, columns=['prediction'])
    prediction_df = prepare_missing(prediction_df, 'prediction')
    prediction_array = prediction_df['missing'].to_numpy()
    control_array = df['missing'].to_numpy()
    nr_false_classified = np.absolute(control_array - prediction_array).sum()
    percentage = nr_false_classified / len(df)
    print(f"missclassified {nr_false_classified} out of {len(df)}: {percentage * 100} %")


# get dataset
df_data_seg = pd.read_csv(config.path_to_data_dir / (config.training_data_filename_wo_extension + '_segmented.csv'))
df_data_unique = df_data_seg.drop_duplicates(subset=['document_id'])
df_extracted = df_data_unique[['text', 'finding_regular', 'bnb', 'class_name']]  # .head(100)
combined = df_extracted['finding_regular'].combine_first(df_extracted['bnb']).tolist()
df_extracted['classification_text'] = combined

# prepare the data and balance it
df_extracted_drop = df_extracted.dropna(subset=['classification_text'])
df_extracted_prep = prepare_missing(df_extracted_drop, 'class_name')
balance_drop = balance_drop(df_extracted_prep)
df_bal = balance_drop[0]
df_dropped = balance_drop[1]  # the dropped data

# split into train and test data
train_df, test_df = train_test_split(df_bal, test_size=0.05)


# k-fold cross validation to evaluate the model
n = 3
seed = 42
kf = KFold(n_splits=n, random_state=seed, shuffle=True)
results = []

for train_index, val_index in kf.split(train_df):
    train_data = train_df.iloc[train_index]
    val_df = train_df.iloc[val_index]
    train_df_balanced = balance_train_data(train_data, df_dropped)

    MODEL_NAME = 'dbmdz/bert-base-german-cased'
    t = text.Transformer(MODEL_NAME, maxlen=128, class_names=["not_missing", "missing"])
    trn = t.preprocess_train(train_df_balanced['classification_text'].to_list(), train_df_balanced['missing'].to_numpy())
    val = t.preprocess_test(val_df['classification_text'].to_list(), val_df['missing'].to_numpy())
    model = t.get_classifier()

    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

    learner.lr_find(show_plot=True)

    learner.fit(1e-4, 1, cycle_len=1)

    # get accuracy of each loop and add it to the list 'results'
    confusion_matrix = learner.validate(class_names=class_names)
    acc = (confusion_matrix[0][0] + confusion_matrix[1][1])/(confusion_matrix.sum())
    results.append(acc)

    learner.view_top_losses(n=5, preproc=t)

    # evaluate the test data
    pred = ktrain.get_predictor(learner.model, t)
    prediction(pred, test_df, 'classification_text')

print(f"Mean_Precision: {sum(results) / len(results)}")


# results
# - 'dbmdz/bert-base-german-cased', all droppeds, lr = 1e-4 and batch_size=16 (ca. 0.88 val acc)
# - 'dbmdz/bert-base-german-cased', 1/5 droppeds, lr = 1e-4 and batch_size=16 (ca. 0.85 val acc)
# - 'distilbert-base-german-cased', 1/2 droppeds, lr = 1e-4 and batch_size=16 (ca. 0.87 val acc)
# - 'dbmdz/bert-base-german-cased', 1/2 droppeds, lr = 1e-4 and batch_size=16 (ca. 0.89 val acc)
# - data regular_finding, 'dbmdz/bert-base-german-cased', 1/2 droppeds, lr = 1e-4 and batch_size=16 (ca. 0.95 val acc)
# - data finding&bnb, 'dbmdz/bert-base-german-cased', 1/2 droppeds, lr = 1e-4 and batch- size=16 (ca 0.98 val acc)
