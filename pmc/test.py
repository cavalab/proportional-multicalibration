from MultiCalibrator import MultiCalibrator
from Auditor import Auditor
from sklearn.linear_model import LogisticRegression
from pmlb import pmlb   
import ipdb
dataset = pmlb.fetch_data('adult', local_cache_dir='/home/bill/projects/pmlb'
)
# groups = ['age','workclass','race','sex','native-country']
groups = ['race','sex']
X = dataset.drop('target',axis=1)
y = dataset['target']
est = LogisticRegression().fit(X,y)

print(f'y balance: {y.sum()/len(y)}')
MC = MultiCalibrator(
                     estimator = est,
                     auditor = Auditor(groups=groups),
                     # metric = 'PMC',
                     metric = 'MC',
                     eta = 0.5,
                     gamma=0.05,
                     alpha=0.1,
                     max_iters=10**6
                    )

MC.fit(X,y)

