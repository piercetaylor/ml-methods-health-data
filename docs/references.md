# References

## Datasets

Each of the four datasets is released under the Creative Commons Attribution 4.0
International license, which the UCI record for each states on its page. The UCI
metadata API does not carry a license field for any of them, so the license was
read from the record page and not from the API.

**Diabetes 130-US Hospitals for Years 1999-2008.** Clore, J., K. Cios, J.
DeShazo and B. Strack, 2014. UCI Machine Learning Repository, record 296,
[archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
101,766 encounters, 47 features. Released under CC BY 4.0.

**HCV data.** Lichtinghagen, R., F. Klawonn and G. Hoffmann, 2020. UCI Machine
Learning Repository, record 571,
[archive.ics.uci.edu/dataset/571](https://archive.ics.uci.edu/dataset/571/hcv+data).
615 subjects, 12 features. Released under CC BY 4.0.

**Liver Disorders.** BUPA Medical Research Ltd., donated by R. S. Forsyth, 1990.
UCI Machine Learning Repository, record 60,
[archive.ics.uci.edu/dataset/60](https://archive.ics.uci.edu/dataset/60/liver+disorders).
345 subjects, 5 blood test features and a continuous target. Released under
CC BY 4.0. The record lists the associated task as regression, gives `drinks`
the role of target, and gives `selector` the role of other.

**Breast Cancer Wisconsin (Diagnostic).** Wolberg, W. H., W. N. Street and O. L.
Mangasarian, 1995. UCI Machine Learning Repository, record 17,
[archive.ics.uci.edu/dataset/17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
569 samples, 30 features. Released under CC BY 4.0. Distributed inside
scikit-learn as `sklearn.datasets.load_breast_cancer`, which is the copy read
here.

## Papers behind the datasets

Strack, B., J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios and
J. N. Clore, 2014. Impact of HbA1c measurement on hospital readmission rates:
analysis of 70,000 clinical database patient records. *BioMed Research
International* 2014, 781670. The paper released with the readmission data. Its
ICD-9 grouping of the primary diagnosis is the grouping reproduced in
`src/data.icd9_group`.

Hoffmann, G., A. Bietenbeck, R. Lichtinghagen and F. Klawonn, 2018. Using
machine learning techniques to generate laboratory diagnostic pathways, a case
study. *Journal of Laboratory and Precision Medicine* 3, 58.
[doi:10.21037/jlpm.2018.06.01](https://doi.org/10.21037/jlpm.2018.06.01). The
paper the HCV record names as its introductory paper.

Street, W. N., W. H. Wolberg and O. L. Mangasarian, 1993. Nuclear feature
extraction for breast tumor diagnosis. *Proceedings of SPIE* 1905, 861-870. The
paper describing how the thirty cell nucleus features were computed.

Forsyth, R. S. and R. Rada, 1986. *Machine Learning: Applications in Expert
Systems and Information Retrieval.* Ellis Horwood. The donor's own use of the
liver disorders data, in which `drinks` is the dependent variable.

## Methods

Agrawal, R., T. Imielinski and A. Swami, 1993. Mining association rules between
sets of items in large databases. *ACM SIGMOD Record* 22(2), 207-216. The
Apriori formulation, and the source of support, confidence and the downward
closure property this analysis relies on.

Brin, S., R. Motwani, J. D. Ullman and S. Tsur, 1997. Dynamic itemset counting
and implication rules for market basket data. *ACM SIGMOD Record* 26(2),
255-264. The definition of conviction, which is the rule metric thresholded
here.

Calinski, T. and J. Harabasz, 1974. A dendrite method for cluster analysis.
*Communications in Statistics* 3(1), 1-27. The internal index the coursework
selected clusterings by, and the one shown here to prefer k-means over the
methods that agree better with the withheld labels.

Ester, M., H.-P. Kriegel, J. Sander and X. Xu, 1996. A density-based algorithm
for discovering clusters in large spatial databases with noise. *Proceedings of
the Second International Conference on Knowledge Discovery and Data Mining*,
226-231. DBSCAN, including the outlier label this analysis reports separately.

Hubert, L. and P. Arabie, 1985. Comparing partitions. *Journal of
Classification* 2, 193-218. The adjusted Rand index, the chance-corrected
agreement used to compare a clustering against the withheld diagnostic
categories.

Vinh, N. X., J. Epps and J. Bailey, 2010. Information theoretic measures for
clusterings comparison: variants, properties, normalization and correction for
chance. *Journal of Machine Learning Research* 11, 2837-2854. Adjusted mutual
information, reported beside the adjusted Rand index.

Rousseeuw, P. J., 1987. Silhouettes: a graphical aid to the interpretation and
validation of cluster analysis. *Journal of Computational and Applied
Mathematics* 20, 53-65.

Davies, D. L. and D. W. Bouldin, 1979. A cluster separation measure. *IEEE
Transactions on Pattern Analysis and Machine Intelligence* 1(2), 224-227.

Breiman, L., 2001. Random forests. *Machine Learning* 45(1), 5-32. The
permutation importance measure and the forest regressor used as the nonlinear
comparison in model 4.

Youden, W. J., 1950. Index for rating diagnostic tests. *Cancer* 3(1), 32-35.
The statistic the readmission decision threshold maximizes on the out-of-fold
training scores.

Saito, T. and M. Rehmsmeier, 2015. The precision-recall plot is more informative
than the ROC plot when evaluating binary classifiers on imbalanced datasets.
*PLoS ONE* 10(3), e0118432. The reason average precision is reported beside
ROC-AUC for the readmission model.

Brier, G. W., 1950. Verification of forecasts expressed in terms of probability.
*Monthly Weather Review* 78(1), 1-3.

Jarque, C. M. and A. K. Bera, 1980. Efficient tests for normality,
homoscedasticity and serial independence of regression residuals. *Economics
Letters* 6(3), 255-259. The test applied to the least squares residuals in
model 4.

## Software

Harris, C. R., K. J. Millman, S. J. van der Walt and others, 2020. Array
programming with NumPy. *Nature* 585, 357-362.

Hunter, J. D., 2007. Matplotlib: a 2D graphics environment. *Computing in
Science and Engineering* 9(3), 90-95.

McKinney, W., 2010. Data structures for statistical computing in Python.
*Proceedings of the 9th Python in Science Conference*, 56-61.

Pedregosa, F., G. Varoquaux, A. Gramfort and others, 2011. Scikit-learn: machine
learning in Python. *Journal of Machine Learning Research* 12, 2825-2830.

Raschka, S., 2018. MLxtend: providing machine learning and data science
utilities and extensions to Python's scientific computing stack. *Journal of
Open Source Software* 3(24), 638.

Seabold, S. and J. Perktold, 2010. statsmodels: econometric and statistical
modeling with Python. *Proceedings of the 9th Python in Science Conference*,
92-96.

Virtanen, P., R. Gommers, T. E. Oliphant and others, 2020. SciPy 1.0:
fundamental algorithms for scientific computing in Python. *Nature Methods* 17,
261-272.
