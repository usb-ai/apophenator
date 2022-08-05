# apophenator

In this project we extract dates of previous examinations of a patient report with a trained model and run them through
the database in order to get the reports of the previous examinations.

### Running Neo4j Browser on port 80
For some reason community edition of Neo4j reports port 80 as already used.
To bypass this issue you can set a rule in iptables.config
```bash
sudo apt-get install iptables-persistent
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000
sudo /sbin/iptables-save > /etc/iptables/rules.v4
```
Note that I am using port 8000 instead of the default port in this example.

The other thing to note is that the browser will try to connect to the database 
from the machine which is running the browser and not the machine that is serving it.
This means the machine and the port of the database need be reachable from the client machine.



### Neo4j config
The default memory values for Neo4j are dynamic. It is helpful to make some changes.
To get recommended values run 
```bash
neo4j-admin memrec
```

Something like this will come out as a result of the command 
```bash
dbms.memory.heap.initial_size=10200m
dbms.memory.heap.max_size=10200m
dbms.memory.pagecache.size=11200m
```
The other intersting recommendation is this 
```bash
dbms.jvm.additional=-XX:+ExitOnOutOfMemoryError
```

This will crash the app when the heap out of memory.

### Fancy additions

If GDS from Neo4j is needed and the server is remote, in order to install it 
follow the [link](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-server/). If it is local 
just do it from the Desktop app.

(See below or link for more details)
```bash
sudo service neo4j stop
wget https://s3-eu-west-1.amazonaws.com/com.neo4j.graphalgorithms.dist/graph-data-science/neo4j-graph-data-science-1.8.6-standalone.zip
sudo apt install unzip
unzip neo4j-graph-data-science-1.8.6-standalone.zip 
sudo cp neo4j-graph-data-science-1.8.6.jar /var/lib/neo4j/plugins
sudo chown --reference=/var/lib/neo4j/plugins/README.txt /var/lib/neo4j/plugins/neo4j-graph-data-science-1.8.6.jar
sudo service neo4j start
```

It can happen that Neo4j enters the crash loop. Could be checked with:
`watch -n 1 sudo service neo4j status`

In which case stop the service and simply `sudo reboot`.

If APOC is needed - same procedure [link](https://neo4j.com/labs/apoc/4.3/installation/#neo4j-server).

Example for APOC
```bash
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.1/apoc-4.4.0.1-all.jar
sudo cp apoc-4.4.0.1-all.jar /var/lib/neo4j/plugins
sudo chown --reference=/var/lib/neo4j/plugins/README.txt /var/lib/neo4j/plugins/apoc-4.4.0.1-all.jar
```
The second command assumes that the file was download to user's home dir.

I am doing chown because there were some issues when I root copied the file.
I could not bring Neo4j back up until I restarted. Apart from that,
it makes more sense that .jar belongs to the same group as the rest of the Neo4j files.
I've been experiencing these reset loop issues when I made config changes and added APOC jar,
without stopping the service first.

In config we also need to make sure apoc is allowed
```bash
dbms.security.procedures.unrestricted=gds.*,apoc.*
dbms.security.procedures.allowlist=gds.*,apoc.*
```

We are using `apoc.*` in the second command because the original 
command used in the documentation block the procedure `apoc.meta.schema procedure` that is being used by 
Graph Data Science Playground.

```bash
dbms.security.procedures.allowlist=gds.*,apoc.coll.*,apoc.load.*
```

System reboot seems to be very helpful after these changes.

I also used the newes GDS and it broke everything that GDS Playground is using.
So I went down with the version.


## THINGS BELOW ARE OUTDATED

## Description

Step 1: Preprocess <br/>
In the preprocess step we first edit the data we have, in order to be able to train the NER-model.
Therefore we segment the texts of the labeled reports in multiple parts and choose either the 'bnb'-part (Befund und
Beurteilung) or the 'finding_regular'-part (Befund), depending on which one is not empty. Moreover we extract the label
from the dataframe, locate it in the text segment and save it to a new dataframe which has the correct form for the
NER-model.

Step 2: NER-Model <br/>
With the preprocessed data we are able to train the dataset with the 'bilstm-crf'-model. After the
training we save the predictor.

Step 3: Prediction <br/>
The saved predictor can now classify what kind of labels (date, today, no previous, missing) we have
on a given patient report from any report of the Hospital's database.

Step 4: Evaluation Of The Predictions <br/>
The predicted labels are getting extracted and processed to a certain format in
order to be able to search for the previous examinations in the Hospital's database. Furthermore, we can extract the
examination modalities of the previous examinations, if the modality is mentioned in the report, and 
only select the previous examinations with the same modality. Once the reports are found,
we can visualize the connections between the reports with the graph database neo4j, including the accession number,
patient id, study date, report text and modality.

## Side Project: "missing label" - classification

This project is able to distinguish patient reports into two classes: documents with label and documents without label.
We take the label we already have (date, no previous, today, yesterday and missing) and divide it into 'missing' if the
document has the label 'missing and 'not missing' if the document has one of the other labels. The next step is to
balance the data set and train it with the 'dbmdz/bert-base-german-cased' - model and k-fold cross-validation. After the
training, you can use the prediction function to classify new documents with a mean-accuracy of approximately 0.97.

## Getting Started

### Dependencies

Conda environment, Python 3.8

### Installing

Install the following:

1. All the requirements, which are listed in the requirements.txt file with the following command:
```
pip install requirements.txt
```
2. Download spacy version, which is compatible with cuda on your computer.
   
3. Download the german language model _de_core_news_sm_ from spicy with the following command:
```
python -m spacy download de_core_news_sm
```
4. Download as well as install [DBeaver](https://dbeaver.io/download/) and log in to the USB database.
5. Download as well as install [neo4j](https://neo4j.com/try-neo4j/) and create your own graph database.
6. Set up an environment file with the following information:
```
DB_USER = *database username*
DB_PW = *database password*

NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USER = *neo4j user*
NEO4J_PW = *neo4j password*

FILENAME_TRAINING_DATA = *csv-filename of the training data*

FILENAME_TRAINING_DATA_WO_EXTENSION = 'laurent_data'

FILENAME_PREPROC_TRAINING_DATA = 'laurent_data_training_set_preprocessed.csv'

FILENAME_PREDICTED_LABELS = 'predicted_labels.txt'

PATH_TO_DATA_DIR = *path to the directory in which you want to store or are storing your (training-)data*

PATH_TO_PREDICTOR_DIR = *path to the directory in which you want to store or are storing the predictor*

TRAIN = *boolean if you want to train your training data or use an existing predictor*
```

### Executing Program
- run the _run.py_ file to train, predict or evaluate your patient reports from the Hospital's database
- run the _doc_classification_temporal_missing_notmissing.py_ file to classify a csv-file of temporal links into reports 
  with a label and reports without a label.
- run the -k-fold_crossvalidation.py_ file to evaluate the model


# To do

## Diagrams
- [x] Chart the whole system.
  - [x] Make a diagram - pipeline
  - [x] Data flow from the user perspective
  - [x] Use-case example
  
<br/>[Here](https://docs.google.com/presentation/d/1MlnNpZVINH0VjTC1-x5s4_LDbso-gytI5I8_k124XNw/edit#slide=id.p) you can
find the diagrams.

## Demo app
- [ ] Make json to feed to the frontend generator (arch graph) 
  [example](https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/data_network.json)
  (noemi)
- [ ] Make a simple search frontend with just textbox and the button (ivan / noemi)
- [ ] Backend - query the database, process and return json. e.g. localhot/patientid=123456 (noemi)
- [ ] Integrate the js code for the visual part (ivan)
- [ ] Improvement - make search more fuzzy, to return a couple of options in case no exact match found.
- [ ] Improvement - add modality, text, dates, etc.
- [ ] Make json from neo4j and directly from the code (before loading to neo4j).


## Results
### Date references
- [ ] Note down used parameters for the model
  - [x] learning rate
    - [x] Comment how you got the learning rate.
  - [ ] Char model training
    - [ ] Comment on user_char / investigate why it fails
  - [ ] Model training   
    - [x] Implement how long training takes
    - [ ] Note down how long training is on which hardware (CPU GPU) and Memory usage
    - [x] Make plots both for validation accuracy and for the convergence of 
      the error as 2 graphs (or overlapping).
      <br/>
      In order to see the plot, use the following command.
      ```
      tensorboard --logdir=path_to_your_logs
      ```
      
    - [ ] Selecting the best model based on saves - how is it done - comment
    - [x] Make the cross validation training.
    - [ ] Show validation results and make a plot with validation results.
    - [x] Show test results.
  
- [x] Prepare test data and test predictions (noemi)
  - [x] Confusion matrix for the date reference extraction
  - [x] results for date extraction - precision, recall (sensitivity), specificity, f1
  
- [ ] graph check - reference check
  - [ ] sample 100 patients and check how well it works.
  - [ ] How many out of 100 had it 100% correct links attached.
    How many had 1 wrong, how many had more than 1 wrong, how many missing.

- [ ] the average amount of references per patient.
  - [ ] True average, predicted average.Helps to do the frontend json first.
  - [ ] Graph showing this as candlesticks or bar graph, whatever looks nice.
  
### Modality
- [x] Label 100 more modalities to be the real test data, 
  the other 100 are used as the development dataset.
- [ ] Calculate precision, recall (sensitivity), specificity, f1 for modality in reference.  
- [ ] Draw confusion matrix for the modality in reference
- [ ] Plot distribution of the modalities and why we do not use szintis
- [x] Make code for region extraction and get it into the neo4j.Check email from Laurent for details.


## Refactor the code and the finishing touches
- [x] Organize in logical groups
- [ ] Make a repo more standard looking
- [ ] Make a Dockerfile for the demo
- [ ] Make a public demo

## Bugs
- Modalities: The extraction of the modalities in the patient reports are made with regex. Therefore, 
  it is quite error-prone, and it might be better not to use the extraction at all. Either don't do extraction
  or train a model for it.
- Test set evaluation: In order to evaluate the test set, we need to bring it in the format, as shown below.
  ![here](additional_files/DataFormat.png)
  <br/>
  The problem is now, that the prediction Data Frame has approximately 30 rows less than the true Data Frame, due 
  to the tokenizer in the prediction. This causes a problem in the confusion matrix generator, and thats whz I had
  to adjust the prediction Data Frame in line 134 to 140 in the python-file kfold_crossvalidation.py. 
  There I deleted every row, which did not have the same word. In total, I deleted more or less 30 words out of ____
  , which do not impact the evaluation significantly.

## Authors

## License