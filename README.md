**_Self-Organizing Algorithms_**

**Introduction**

This repository contains python implementations of self-organizing algorithms.
The algorithm suite contains;  

1. Growing Self-Organizing Maps (GSOM) by [CDAC](https://www.latrobe.edu.au/centre-for-data-analytics-and-cognition)   
2. Incremental Knowledge Acquisition and Self Learning (IKASL) by [CDAC](https://www.latrobe.edu.au/centre-for-data-analytics-and-cognition)
3. Self-Organizing Maps (SOM) by [PyMVPA](http://www.pymvpa.org/)  

The usage of the implementations are as follows:
  
**_GSOM_**

Concept and implementation papers:  
[1] D. Alahakoon, S. K. Halgamuge, and B. Srinivasan, “Dynamic self-organizing maps with controlled growth for knowledge discovery,” IEEE Transactions on Neural Networks, vol. 11, no. 3, pp. 601–614, May 2000.  
[2] R. Nawaratne, D. Alahakoon, D. De Silva. “HT-GSOM: Dynamic Self-organizing Map with Transience for Human Activity Recognition”. IEEE 17th International Conference on Industrial Informatics (INDIN). IEEE, 2019.


_Required Modules_
* python 3.X (Tested with 3.5 and 3.6)
* pandas
* numpy
* scipy
* scikit-learn
* tqdm
* matplotlib
* squarify

_Usage_

I have setup a sample using ZOO animal dataset.

* Go to `gsom/applications/zoo_experiment/zoo.gsom.py`
* Update the GSOM config and File config respectively (already setup for current dataset).
* Run `zoo_gsom.py`.

**_IKASL_**

Concept and implementation papers:  
[1] D. De Silva and D. Alahakoon, “Incremental knowledge acquisition and self learning from text,” in The 2010 International Joint Conference on Neural Networks (IJCNN), 2010, pp. 1–8.

_Required Modules_
* python 3.X (Tested with 3.5 and 3.6)
* pandas
* numpy
* scipy
* scikit-learn
* tqdm
* matplotlib
* squarify
* heapq
* graphviz (Need to install both system installer and python package https://graphviz.gitlab.io/download/)

_Usage_

I have setup a sample using human activity video dataset, where features are extracted as BOW using MS Cognitive Vision Toolkit.

* Go to `ikasl/applications/collective-activity/collective-activity-bow.py`
* Update the config files respectively (already setup for current dataset).
* Run `collective-activity-bow.py`.

**_SOM_**

Please refer: http://www.pymvpa.org/examples/som.html
