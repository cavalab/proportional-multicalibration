Reconciling Differential Fairness and Multicalibration 
(Work in progress)

# **Contact**
@cavalab : 
- Elle Lett (Elle.Lett@childrens.harvard.edu)
- Guangya Wan (gwan@hsph.harvard.edu)
- William La Cava ( @lacava, william.lacava@childrens.harvard.edu ) 

# **License**

GPLv3

# **Data**

* MIMIC 4 Dataset

MIMIC (Medical Information Mart for Intensive Care) is a large, freely-available database comprising deidentified health-related data from patients who were admitted to the critical care units of the Beth Israel Deaconess Medical Center. This site https://mimic.mit.edu/docs/iv/ contains information related to the description and structures tables/features, its difference between the previous dataset, and the tuotial on how to get started with it.

For our project, We used 4 files from MIMIC 4, which include admission and patients file under Core directory, and triage and edstay file under Ed directory. You can either access the data using BigQuery from google and read them from Google Healthcare datathon repository (https://github.com/GoogleCloudPlatform/healthcare), or you can direcly access and download the files from https://physionet.org/content/mimiciv/1.0/ after you complete the necessary setup including registering an accont, signing the data use agreement, and finishing the required training(This is what we did for this project).

* BCH Dataset

TBD

# **Pipeline:**

## **MIMIC Data Preprocessing**

Our python script(experiment/preprocessing/clean_mimic.py) accepts 5 input, the file path of admission,edstay,triage, and patient, and the path you want to store the final resulting file respectively. You can try using the -h document to get more help for more specific information.

Once the input are provided, it will process the file, including remove unnecessary columns,outliers, adding customized columns such as previous number of visits(look at steps below for more information)  and save it on the provided path to be used for our further machine learning and fairness models.

See the following for our data preprocessing plans:

### **Step 1 : Data Merging**

We first join triage and edstay table on stay_id; then join the result with admission table on subject_id and finally join result with patient table to get gender and age info.

### **Step 2 : Dropping Unnecessary Columns**

We drop duplicates on stay_id(keeping first entry) then drop unnecessary columns for our modelling (e.g. deathtime)
We then remove outliers based on some pre-determined criteria (for example, the temperature should be between 95 and 105)
Finally we remove patients who are admitted (explained in the next section) with admission_types with 'OBSERVATION' in the name

### **Step 3 : Creating New Columns**
We create 3 new columns, previous number of admission, previous number of visits, and our label 'y' indicating whether one is admitted or not.

For the label 'y' indicating whether or not the patient is admitted, we just simplely defined as whether or not the column 'hadm_id' is na(then 0) or not(then 1).

For Previous number of admission, it's just the number of admission for a given subject_id prior to the current visit. Simiarly, previous number of visits is just  the number of visits for a given subject_id prior to the current visit. Note that # of visits should always be greater than or equal to number of admission, as someone who makes visits does not necessaily get admitted. We manually create these two labels as hostirically they show up in the related literature as relevant features.

### **Step 4 : Transforming Data**

We transform the text variable 'cheifcomplaint' using bag of words. Specifically, we one-hot encoded all of the vocabulary(using top 100 only), and treated the rest as the infrequent symptoms.

Also note that to deal with cheifcomplanit, I used the latest feature of sklean's one-hot encoding to encode infrequent features, so mostly like you will need to update your scikit-learn to run this with this command: !pip install -U scikit-learn

We also tried one-hot encoding other categorical variables including admission_type,admission_location,language,insuance,martial status,
and ethnicity.

We convert continopus age variable into 5 year bins.

### **Step 5 : Save to Path**

We finally saved our file on path provided(default is the same path), and started our model training. Also note that we drop 'chiefcomplaint' and 'admission_location' when we read files in model training.

## **Text/Categorical data processing**

Before we started fitting our model, we first dealt with text/categorical data which can not be direcly used in our machine learning models(experiment/read_file.py). Optionally we one-hot-encode(label-encoded, or use word embedding) text
    features, and label encode categorical data(see below for more details)

### **Step 1**

Read the cleaned data file above, label-encode all of the non-numerical variables(except text) using Scikit-learn's Labelencoder,and saved the encoding into a json file.

### **Step 2**

Clean the text including strip the empty space/lower case all text, remove non-alphbetical/numerical text, and filling missing value with a special string charatcter '__'.

Depending on the input argument(with default being using one-hot encoding), we convert the text to feature vectors using three methods:

<ol>

<li>One-hot encode</li>
Using Sklearn's CountVectorizer to fit and transform the text data, with default parameter min_df = 100 to prune infrequent text tokens from text.

<li>Label encode</li>

Using Sklearn's Label encoder to fit and transform the text data, and only include the top 20% word based on their ranked appear frequency. 

<li>Word embedding</li>

Using Hugging face's pre-trained transformer model(pritamdeka/S-Biomed-Roberta-snli-multinli-stsb) based on medical text to encode the text. Instead of using the full embedding representation of vector, we use the top 50 most frequent appeared text token and calculate the cosine similarity between the chiefcomplaint and the text.

**For mode detail about the word embedding usage, please refer to experiment/preprocessing/embedding.md**

</ol>

Note that all of those setting with the text encoding can be tuned if needed.

## **Model Fitting:**

After data is ready, we used three classical machine learning methods(XGboosting, Random Forest, and Logistic Regression) to predict our predefined binary variable 'y' (which indicates wheather a patient is admitted or not)and also we evaluated and improved the fainness matrics:) ; We also used cluster to help us boost the model training speed especially for we are selecting the best set of hyper-parameters for our models.

For all of the model discussed below, we used the following pipelines:

* 1: Train and split the data into 3:1 ratio. (Optionally we can scale and sample a subset from the data)
* 2: Fit one of the model below
* 3: Saved the resulting metrics(AU-ROC, MC Loss, PMC Loss, and DC loss) with the specific hyperparameter setting or feature importance if applicable.

In addition, we allow the user to have the following input options when running the fitting model (experiment/evaluate_model.py):

  -file FILE            Data file to analyze; ensure that the target/label column is labeled as "y". If you use the preprocessing file, you do not need to do anything
  \
  -h, --help            Show this help message and exit.
  \
  -ml ML                Name of estimator (with matching file in ml/)
  \
  -results_path RDIR    Name of save file
    \
  -emb_path EMB         Path of pre_trained embedding saved file
  \
  -seed RANDOM_STATE    Seed / trial
  \
  -alpha ALPHA          Calibration tolerance (for metrics)
  \
  -n_bins N_BINS        Number of bins to consider for calibration
  \
  -gamma GAMMA          Min subpop prevalence (for metrics)
  \
  -rho RHO              Min subpop prevalence (for metrics)
  \
  -ohc {ohc,label_encoding,embedding}
                        Specificy how text should be one-hot-encoded.
    \
  -groups GROUPS        groups to protect
  \
  -text TEXT            Specify text features with comma seperated


### **Model 1: Baseline Models:**

We first fit the three machine learning models without any fairness constraint and evaluated their training and testing AUROC scores as well as other fairness metrics such as MC/PMC/DC loss. We used cross-validation to help us select the best set of hyperparameters and saved them as the baseline model for our models with fairneess improvement as discussed below.

### **Model 2: MC Models:**

We improved our model's fairness by fitting our implemented MultiCalibrator using the base estimators discussed above with MC(Multi-Calibration) as the optimized metrics for the model to fit; We finally evaluated their AUROC and other fairness metics using the best hyperparmeter selected by Cross-Validation.

### **Model 3: PMC Models:**

We improved our model's fairness by fitting our implemented MultiCalibrator using the base estimators discussed above with PMC(Proportional Multi-Calibration) as the optimized metrics for the model to fit; We finally evaluated their AUROC and other fairness metics using the best hyperparmeter selected by Cross-Validation.

**quick review on the procedure**: 

with PMC, for categories  in C:\
$\bar r$ = get mean prediction on category c
\
$\bar y$ = mean label on c
\
if MC:
\
t = alpha
\
else:
\
t = ybar * alpha
\
update the model when abs(ybar - rbar) > t:


### **Experiment Set up:**

We run 100 trials and run hyperparamter selection on 

alphas=(
0.01
0.05
0.1
)
gammas=(
0.05
0.10
)
n_binses=(
5
10
)
rhos=(
0.01
0.05
0.1
)
methods=(
    "lr_mc_cv"
    "lr_pmc_cv"
    "rf_mc_cv"
    "rf_pmc_cv"
)
using cluster. on MIMIC dataset following the above procedure.

## **Results:**

All of the above results will be stored in a json format. We created a notebook to display those results in a more tabular format as well as made some visuzizations. We showed that our innonative post-processing algorithm for learning risk prediction models that satisfy proportional multicalibration indeed improved the fairness metrics significantly on our machine learning models as suggested by the three metrics loss while still preseving its performance in terms of AUROC.
(https://github.com/cavalab/proportional-multicalibration/blob/main/notebooks/postprocess_experiment.ipynb)



