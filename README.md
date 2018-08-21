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

## Julia
Julia learning resources:  
https://julialang.org/learning/

Pass through this book:  
https://www.packtpub.com/application-development/julia-high-performance-programming
