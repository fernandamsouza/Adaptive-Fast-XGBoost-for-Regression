from scipy import stats
import numpy as np
import scikit_posthocs as sp

## BASE DE 500.000 INSTANCIAS - ABRUPTA ##
d_reset = [2636, 3711, 3955, 4608, 3780]
reset = [3583, 4243, 4984, 7441, 5164]
d = [2770, 4079, 4537, 6154, 4892]
d_sem_reset = [3962, 4958, 5522, 7434, 5754]
htr = [12801, 11549, 11549, 13040, 11703] 

# perform Friedman Test
print(stats.friedmanchisquare(d_reset, reset, d, d_sem_reset))

# combine three groups into one array
data = np.array([d_reset, reset, d, d_sem_reset])

# perform Nemenyi post-hoc test
print(sp.posthoc_nemenyi_friedman(data.T))


import Orange 
import matplotlib.pyplot as plt


names = ["first", "third", "second", "fourth" ]
avranks =  [1.9, 3.2, 2.8, 3.3 ]
cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()