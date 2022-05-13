import ipdb
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def read_file(filename, label='y', one_hot_encode=False,
              drop_columns =
              ['chiefcomplaint','admission_type','admission_location']):
    
    
    input_data = pd.read_csv(filename)
    # Drop these data for now,
    X = input_data.drop([label]+drop_columns,axis = 1)
    # feature_names = [x for x in input_data.columns.values if x != label]
    # feature_names = np.array(feature_names)

    # X = pd.get_dummies(input_data)
    # ipdb.set_trace()
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

    # if one_hot_encode:
    #     X = pd.get_dummies(input_data)
    # else:
    #     X = input_data

    # X = X.values.astype(float)
    y = input_data[label].astype(int)
        # Note that feature name might not be the same as dataset, as we use
    # one-hot encoding here
    # assert(X.shape[1] == feature_names.shape[0])

    return X, y 

