from collections import defaultdict
# demographics
A = ['Race','Ethnicity','Gender','Age']

A_rename = {
    'Hispanic Yes No': 'Ethnicity',
    'age_group': 'Age'
}

A_options = {
    'Race':defaultdict(lambda: 'Other'),
    'Ethnicity': {
       'Yes':'HL',
       'No':'NHL'
    },
    'Gender': {
        'M':'M',
        'F':'F'
    },
   # 'Age': {
   #     '>5Y':'older than 5Y',
   #     '18M-3Y':'5Y or younger', 
   #     '3-5Y':'5Y or younger', 
   #     '12-18M':'5Y or younger', 
   #     '0-3M':'5Y or younger', 
   #     '6-12M':'5Y or younger', 
   #     '3-6M':'5Y or younger'
   #     
   # }
} 
A_options['Race'].update({
        'Black or African American':'Black', 
#             'Other', 
        'White':'white', 
#             'Unable to Answer',
#             'Declined to Answer', 
#             'Unknown', 
        'Asian':'Asian',
        'American Indian or Alaska Native':'AI',
        'Native Hawaiian or Other Pacific Islander':'NHPI', 
#         'Hispanic or Latino': 'HLTN'
    })

def clean_dems(df):
    """Re-codes demographics according to dictionaries above"""
    df = df.rename(columns = A_rename)
    for a in A:
        if a in A_options.keys():
            df = df.loc[df[a].isin(list(A_options[a].keys())),:]
            df[a] = df[a].apply(lambda x: A_options[a][x])
    return df
