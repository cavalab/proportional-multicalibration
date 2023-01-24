from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import time
import re
if __name__ == "__main__":
    data = pd.read_csv('../data/bch_final.csv')

    regex = re.compile('[^a-zA-Z0-9]')
    text= data['chiefcomplaint'].replace('[^a-zA-Z0-9 ]', ' ', regex=True)
    text = text.fillna(' ')
    text = text.dropna()
    text =  text.str.lower()
    text = text.replace(r'\s+', ' ', regex=True)
    text = text.str.lstrip()
    text = text[~text.apply(lambda x: x.isnumeric())]
    sentences = text.values

    start = time.time()
    model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
    print(time.time() - start, 's Finish loading model')
    # model2 = SentenceTransformer('S-PubMedBert-MS-MARCO-SCIFACT')
    embeddings_bch = model.encode(sentences)
    print(time.time() - start, 's Finish encoding vectors')

    np.save('BCH_embedding.npy', embeddings_bch)

    print(time.time() - start, 's Finish saving files')