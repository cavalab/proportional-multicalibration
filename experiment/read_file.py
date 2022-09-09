import ipdb
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
from sentence_transformers import  util


def one_hot_encode_text(data,text_label,mindf = 100):
    df = data.copy()
    df[text_label] = df[text_label].fillna('___')
    # Fill NA with ____, which makes sense
    df[text_label] = df[text_label].apply(lambda x: x.lower())
    df[text_label] = df[text_label].apply(lambda x: ' '.join(sorted(x.replace(',','').split(' '))))
    allsentences = df[text_label]
    vectorizer =CountVectorizer(min_df=mindf) # sset this parameter that we can tune later
    X = vectorizer.fit_transform(allsentences)
    df = pd.concat(
    [
        df,
        pd.DataFrame(
            X.toarray(), 
            index=df.index, 
            columns=vectorizer.get_feature_names_out()
        )
    ], axis=1
) # Try to fix the warning
    df = df.loc[:,~df.columns.duplicated()].copy() # remove potential duplicates
    return df
def label_encode_text(data,text_label,top_ratio = 0.8):
    """Label encoding of column text_label"""
    df = data.copy()
    df[text_label] = df[text_label].fillna('___') # Fill NA with ____, which makes sense
    df[text_label] = df[text_label].apply(lambda x: x.lower())
    df[text_label] = df[text_label].apply(lambda x: ' '.join(sorted(x.replace(',','').split(' '))))
    words_rep = list(df[text_label].value_counts()[np.where((df[text_label].value_counts()/df.shape[0]).cumsum()>top_ratio)[0]].index)
    enc = LabelEncoder()
    df[f'{text_label}_label_encoded'] = enc.fit_transform(df[text_label])
    return df


def embedding_encode_text(data,text_label,embedding = 'pritamdeka/S-Biomed-Roberta-snli-multinli-stsb',dim =50 ):
    """Embedding encoding of column text_label"""
    df = data.copy()
    regex = re.compile('[^a-zA-Z]')
    text=  df['chiefcomplaint'].replace('[^a-zA-Z0-9 ]', np.nan, regex=True)
    text = text.fillna('')
    text =  text.str.lower()
    text = text.replace(r'\s+', ' ', regex=True)
    text = text.str.lstrip()
    # text = text[~text.apply(lambda x: x.isnumeric())]
    sentences = text.values

    model1 = SentenceTransformer(embedding)
    embeddings1 = model1.encode(sentences)
    if(dim < embeddings1.shape[1]):
        temp = np.zeros((dim,embeddings1.shape[1]))
        temp_series = text.value_counts()
        df = df.loc[text.index].drop(text_label,axis = 1)
        text = text.reset_index().drop('index',axis = 1)
        for i in range(dim):
            #Sentences are encoded by calling model.encode()
            emb1 = model1.encode(text.values[i])
            temp[i,:] = emb1
        
        cos_sim = util.cos_sim(temp, embeddings1.astype('double'))
        df = pd.concat(
    [
        df,
        pd.DataFrame(
            cos_sim.T, 
            index=df.index, 
            columns=['Cosine Similarity of Feature : ' + temp_series.index[i] for i in range(dim)]
        )
    ], axis=1)
    else:
        df = pd.concat(
    [
        df,
        pd.DataFrame(
            embeddings1, 
            index=df.index, 
            columns=['Word Embedding : ' + str(i+1) for i in range(embeddings1.shape[1])]
        )
    ], axis=1)
    return df



def read_file(filename, one_hot_encode, label, text_features=None):
    """read filename into pandas dataframe. optionally onehotencode text
    features, and label encode categorical data. returns X,y.

    text_features: list. features to treat as text. 
    one_hot_encode: bool. whether to one hot encode text_features. if False,
                    we apply label_encode_text to text_features.
    """
    input_data = pd.read_csv(filename)
    # Drop these data for now,

    for col in text_features:
        if(one_hot_encode == 1):
            print('One Hot Encoding',col)
            input_data = one_hot_encode_text(input_data,col)
        elif(one_hot_encode == -1):
            print(' Label Encoded Text ',col) # Will add more conditions here for other situation
            input_data = label_encode_text(input_data,col)
        else:
            print(' Embedding Encoded Text ',col) # Will add more conditions here for other situation
            input_data = embedding_encode_text(input_data,col)
    X = input_data.drop([label] + text_features,axis = 1)
    encodings={}
    # 
    for c in X.select_dtypes(['object','category']).columns :
        # if c in text_features:
        #     continue
        #     print(c)
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encodings[c] = {k:list(v) if isinstance(v, np.ndarray) else v 
                        for k,v in vars(le).items()
                       }
    with open('label_encodings.json','w') as of:
        json.dump(encodings, of)
    y = input_data[label].astype(int)
    return X, y 


