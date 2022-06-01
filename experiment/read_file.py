import ipdb
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Make read_file work for both dataset by adding extra parameter to the below functions
def clean_text(data,text_label):
    df = data.copy()
    df[text_label] = df[text_label].fillna('___')
    # Fill NA with ____, which makes sense
    df[text_label] = df[text_label].apply(lambda x: x.lower())
    df[text_label] = df[text_label].apply(lambda x: ' '.join(sorted(x.replace(',','').split(' '))))
    allsentences = df[text_label]
    vectorizer =CountVectorizer(min_df=6)
    X = vectorizer.fit_transform(allsentences)
    df[vectorizer.get_feature_names_out()] = X.toarray()
    return df

def clean_text_label(data,text_label):
    df = data.copy()
    df[text_label] = df[text_label].fillna('___') # Fill NA with ____, which makes sense
    df[text_label] = df[text_label].apply(lambda x: x.lower())
    df[text_label] = df[text_label].apply(lambda x: ' '.join(sorted(x.replace(',','').split(' '))))
    words_rep = list(df[text_label].value_counts()[np.where((df[text_label].value_counts()/df.shape[0]).cumsum()>0.80)[0]].index)
    df.loc[df[text_label].isin(words_rep),text_label] = 'infrequent'
    enc = LabelEncoder()
    df['Complaint_label_encoded'] = enc.fit_transform(df[text_label])
    return df

def read_file(filename, one_hot_encode, label,text_label):
    input_data = pd.read_csv(filename)
    # Drop these data for now,

    if(one_hot_encode):
        input_data = clean_text(input_data,text_label)
        print(' One Hot Encoded Text ')
    else:
        input_data = clean_text_label(input_data,text_label)
        print(' Label Encoded Text ')
    X = input_data.drop([label,text_label],axis = 1)
    encodings={}
    for c in X.select_dtypes(['object','category']).columns:
        print(c)
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encodings[c] = {k:list(v) if isinstance(v, np.ndarray) else v 
                        for k,v in vars(le).items()
                       }

    with open('label_encodings.json','w') as of:
        json.dump(encodings, of)
    y = input_data[label].astype(int)

    return X, y 


