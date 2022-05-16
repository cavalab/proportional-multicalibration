Reconciling Differential Fairness and Multicalibration 

Elle Lett & William La Cava

(Work in progress)


# MIMIC 4 Dataset

MIMIC (Medical Information Mart for Intensive Care) is a large, freely-available database comprising deidentified health-related data from patients who were admitted to the critical care units of the Beth Israel Deaconess Medical Center. This site https://mimic.mit.edu/docs/iv/ contains information related to the description and structures tables/features, its difference between the previous dataset, and the tuotial on how to get started with it.

For our project, We used 4 files from MIMIC 4, which include admission and patients file under Core directory, and triage and edstay file under Ed directory. You can either access the data using BigQuery from google and read them from Google Healthcare datathon repository (https://github.com/GoogleCloudPlatform/healthcare), or you can direcly access and download the files from https://physionet.org/content/mimiciv/1.0/ after you complete the necessary setup including registering an accont, signing the data use agreement, and finishing the required training(This is what we did for this project).


# Data Proprocessing:

Our python script accepts 5 input, the file path of admission,edstay,triage, and patient, and the path you want to store the final resulting file respectively. You can try using the -h document to get more help for more specific information.

Once the input are provided to our clean_mimic.py script, it will process the file and save them as the final.csv, will will be used for our further machine learning and fairness models.

See the following for our data preprocessing plans:

## Step 1 : data merging

We first join triage and edstay table on stay_id; then join the result with admission table on subject_id and finally join result with patient table to get gender and age info.

## Step 2 : Dropping Unnecessary Columns

We drop duplicates on stay_id(keeping first entry) then drop unnecessary columns for our modelling (e.g. deathtime)
We then remove outliers based on some pre-determined criteria (for example, the temperature should be between 95 and 105)
Finally we remove patients who are admitted (explained in the next section) with admission_types with 'OBSERVATION' in the name

## Step 3 : Creating New Columns
We create 3 new columns, previous number of admission, previous number of visits, and our label 'y' indicating whether one is admitted or not.

For the label 'y' indicating whether or not the patient is admitted, we just simplely defined as whether or not the column 'hadm_id' is na(then 0) or not(then 1).

For Previous number of admission, it's just the number of admission for a given subject_id prior to the current visit. Simiarly, previous number of visits is just  the number of visits for a given subject_id prior to the current visit. Note that # of visits should always be greater than or equal to number of admission, as someone who makes visits does not necessaily get admitted. We manually create these two labels as hostirically they show up in the related literature as relevant features.

## Step 4 : Transforming Data

We transform the text variable 'cheifcomplaint' using bag of words. Specifically, we one-hot encoded all of the vocabulary(using top 100 only), and treated the rest as the infrequent symptoms.

We also tried one-hot encoding other categorical variables including admission_type,admission_location,language,insuance,martial status,
and ethnicity.

We convert continopus age variable into 5 year bins.

## Step 5 : Save to Path

We finally saved our file on path provided(default is the same path), and started our model training. Also note that we drop 'chiefcomplaint' and 'admission_location' when we read files in model training.

# Run the pipeline for Machine Learning on MIMIC data:

TBD

