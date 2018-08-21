# TWIL-20180820
This Week I Learned - 2018/08/20  
This week is mostly spent on 
1. www.metanet.hk
2. using machine learning to study tumor features and mutation data.

## Machine Learning
Pytorch multigpu example:  
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
* to specify which 2 gpus to use out of 4:  
```
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"
device = torch.device('cuda')
```


A very good pytorch video lecture series:  
https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&disable_polymer=true


Disable jupyter notebook error / warning output for one cell:  
https://ipython.readthedocs.io/en/stable/interactive/magics.html?highlight=capture#cellmagic-capture
* code:  
```
%%capture --no-stdout --no-display
```


XGBoost evaluation metrics:  
https://xgboost.readthedocs.io/en/latest/parameter.html
* For unbalanced data set, use AUC


Gradient Boosting Stratified KFold cross validation:
```
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# CV model
model = xgboost.XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)
X_all = X_train.append(X_test)
y_all = y_train.append(y_test)
results = cross_val_score(model, X_all, y_all, cv=kfold, n_jobs=-1, scoring='roc_auc')
print("AUC: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

## Julia
Julia learning resources:  
https://julialang.org/learning/

Pass through this book:  
https://www.packtpub.com/application-development/julia-high-performance-programming

## Github
https://github.com/lalonderodney/SegCaps  
https://github.com/baidu-research/NCRF  
https://github.com/mingrui/python-wsi-preprocessing/blob/master/docs/wsi-preprocessing-in-python/index.md  
https://github.com/ritchieng/the-incredible-pytorch  

## Research
Medical Imaging with Deep Learning:  
https://openreview.net/group?id=MIDL.amsterdam/2018/Conference
