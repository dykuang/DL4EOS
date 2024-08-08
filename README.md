# DL4EOS 
 Using deep learning tools based on neural networks for predicting EOS surfaces under different data conditions.  
 Compared with classical deduction based modeling pipeline, data driven models allows more flexibility and more tolerance on the usage of physical priors, as well as their forms. 
 
<img src="https://github.com/user-attachments/assets/7c02cbf4-46da-4c10-b86d-cd1ed49f99ce" width="600" />  

This repo contains codes for reproducing results in our [paper]() and for potential readers of interest to retrain their own models on related tasks.  
 
<img src="https://github.com/user-attachments/assets/246c14c5-2264-40e4-84db-2101b0956fb6" width="600" />  

The proposed framework allows learning from different data sources jointly. Different forms of physical priors can also be conveniently injected during training for enforcing physical properties of interest. Furthermore, it is also very convenient to extend to probablistic model with uncertainty quantifications via many existing techniques. 

<img src="https://github.com/user-attachments/assets/bda45b2f-b078-408d-bb4f-e7aac8808965" width="600" />  

 ## Dependancies :dependabot:
 ```
numpy==1.26.3
pandas==2.2.1
torch==2.2.0+cpu
matplotlib==3.8.2
scikit-learn==1.4.0
```
The `baysian_torch` module is optional and can be installed according to [HERE](https://github.com/IntelLabs/bayesian-torch)  

## Instructions ☘️  
*.py* files can be run as regular python scripts or interactively per cell (separated by `#%%`) if using IDEs such as `VScode` with JupyterNotebook plug-ins. 
🗒️`helper.py` contains some helper/utility functions.  
🗒️`Networks.py` contains different network models.  
🗒️`CV_PVTE_train.py` and `PVTE_eval.py` are scripts for training and evaluating models with cross validation. The prediction of $P$ and $E$ surfaces are supervised.  
🗒️`CV_PVTE_res_train.py` and `PVTE_res_eval.py` are scripts similar as above, but the models will be used to learning the residual after a polynomial regression as the first stage learning.  
🗒️`L2H_PVTE_train.py` and `L2H_PVTE_eval.py` are scripts for training and evluating models with the task that uses data gathered from low pressure and low temperature for the prediction on the whole domain of interest including data observed under high pressure or high temperature.  
🗒️`PVT_PV.py` aims for the case where one only have static PVT data and PV data along Hugoniont and still would like to predict EOS surfaces: $P = P(V,T)$, $E = E(V,T)$. In this case, the learning of $E = E(V,T)$ is unsupervised and the framework allows learning these data from different sources jointly.  
🗒️`PVT_PV_uq.py` is a version based on above with uncertainty quantification learned at the same time.  
🗒️`CaseStudy.py` gives several examples for enforcing different physical priors as regularizations. It also provides several methods for visualization of effectiveness with enforced physical priors.  
