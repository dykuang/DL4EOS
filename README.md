# DL4EOS
 Using deep learning tools based on neural networks for predicting EOS surfaces under different data conditions.

 This repo contains code for reproducing results in our [paper]() and for potential readers of interest to retrain their own models on related tasks.  

 ## Dependancies
 ```
numpy==1.26.3
pandas==2.2.1
torch==2.2.0+cpu
matplotlib==3.8.2
scikit-learn==1.4.0
```
The `baysian_torch` module is optional and can be installed according to [HERE](https://github.com/IntelLabs/bayesian-torch)  

## Instructions ☘️  
🗒️`helper.py` contains some helper/utility functions.  
🗒️`Networks.py` contains different network models.  
🗒️`CV_PVTE_train.py` and `PVTE_eval.py` are scripts for training and evaluating models with cross validation. The prediction of $P$ and $E$ surfaces are supervised.  
🗒️`CV_PVTE_res_train.py` and `PVTE_res_eval.py` are scripts similar as above, but the models will be used to learning the residual after a polynomial regression as the first stage learning.  
🗒️`L2H_PVTE_train.py` and `L2H_PVTE_eval.py` are scripts for training and evluating models with the task that uses data gathered from low pressure and low temperature for the prediction on the whole domain of interest including data observed under high pressure or high temperature.  
🗒️`PVT_PV.py` aims for the case where one only have static PVT data and PV data along Hugoniont and still would like to predict EOS surfaces: $P = P(V,T)$, $E = E(V,T)$. In this case, the learning of $E = E(V,T)$ is unsupervised and the framework allows learning these data from different sources jointly.  
🗒️`PVT_PV_uq.py` is a version based on above with uncertainty quantification learned at the same time.  
🗒️`CaseStudy.py` gives several examples for enforcing different physical priors as regularizations. It also provides several methods for visualization of effectiveness with enforced physical priors.  
