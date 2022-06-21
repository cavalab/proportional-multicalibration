import ipdb
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Make read_file work for both dataset by adding extra parameter to the below functions
def one_hot_encode_text(data,text_label):
    df = data.copy()
    df[text_label] = df[text_label].fillna('___')
    # Fill NA with ____, which makes sense
    df[text_label] = df[text_label].apply(lambda x: x.lower())
    df[text_label] = df[text_label].apply(lambda x: ' '.join(sorted(x.replace(',',' ').split(' '))))
    allsentences = df[text_label]
    vectorizer =CountVectorizer(min_df=6)
    X = vectorizer.fit_transform(allsentences)
    df[vectorizer.get_feature_names_out()] = X.toarray()
    return df

def label_encode_text(data,text_label):
    """Label encoding of column text_label"""
    df = data.copy()
    df[text_label] = df[text_label].fillna('___') # Fill NA with ____, which makes sense
    df[text_label] = df[text_label].apply(lambda x: x.lower())
    df[text_label] = df[text_label].apply(lambda x: ' '.join(sorted(x.replace(',','').split(' '))))
    words_rep =
    list(df[text_label].value_counts()[np.where((df[text_label].value_counts()/df.shape[0]).cumsum()<0.80)[0]].index)
    df.loc[df[text_label].isin(words_rep),text_label] = 'infrequent'
    enc = LabelEncoder()
    df[f'{text_label}_label_encoded'] = enc.fit_transform(df[text_label])
    df = df.drop(columns=text_label, axis=1)
    return df

def read_file(filename, one_hot_encode, label, text_features=None):
    """read filename into pandas dataframe. optionally onehotencode text
    features, and label encode categorical data. returns X,y.

    text_features: list. features to treat as text. 
    one_hot_encode: bool. whether to one hot encode text_features. if False,
                    we apply label_encode_text to text_features.
    """
    input_data = pd.read_csv(filename)

    for col in text_features:
        if(one_hot_encode):
            print('One Hot Encoding',col)
            input_data = one_hot_encode_text(input_data,col)
        else:
            print(' Label Encoded Text ',col)
            input_data = label_encode_text(input_data,col)

    X = input_data.drop(label,axis = 1)

    # encode anything that is categorical with a label encoder, if not in
    # text_features
    encodings={}
    for c in X.select_dtypes(['object','category']).columns :
        if c in text_features:
            continue
        print(c)
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encodings[c] = {k:list(v) if isinstance(v, np.ndarray) else v 
                        for k,v in vars(le).items()
                       }
    with open('label_encodings.json','w') as of:
        json.dump(encodings, of)

    y = input_data[label].astype(int)

    # ipdb.set_trace()
    return X, y 


