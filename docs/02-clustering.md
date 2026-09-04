# Clustering: recovering hepatitis C categories without labels

## The question

The HCV data (Lichtinghagen et al. 2020) holds ten laboratory values for 615
subjects together with a diagnostic category for each: blood donor, hepatitis,
fibrosis or cirrhosis. This analysis measures how much of that four-way
structure the laboratory values carry on their own. The labels are withheld
from every clustering and used only afterwards, to score agreement.

A cluster index and a class index are arbitrary names, so a clustering cannot
be scored by comparing one to the other as a value. Agreement is measured with
four chance-corrected or renaming-invariant quantities: the adjusted Rand index
(Hubert and Arabie 1985), adjusted mutual information (Vinh et al. 2010),
homogeneity and completeness. Three internal indices are reported beside them,
computed without any reference to the labels, because a method meant for use
where labels are unavailable has to be judged by what is available there.

## Preparation

Of the 615 released subjects, 7 carry the category `0s=suspect Blood Donor`,
which marks a donor whose values were questioned. Seven subjects cannot form a
group of their own and they are not donors either, so they are removed and
counted. The remaining 608 are all analyzed.

Twenty-six of them are missing at least one of the ten assays, 31 cells of the
6,080 in the panel, or 0.51 percent. Eighteen of the missing cells are alkaline
phosphatase and ten are cholesterol. A complete-case rule would have removed
those 26 subjects, and it would have removed them unevenly.

| Category | Released | Analyzed | Of which imputed |
|---|---|---|---|
| Blood donor | 533 | 533 | 7 |
| Hepatitis | 24 | 24 | 4 |
| Fibrosis | 21 | 21 | 9 |
| Cirrhosis | 30 | 30 | 6 |

Dropping the incomplete rows would have cost 43 percent of the fibrosis cases
and 20 percent of the cirrhosis cases against 1 percent of the donors, which
thins exactly the categories the analysis wants to recover. The missing values
are imputed instead, and the imputation method matters. The median of alkaline
phosphatase is 66.7 among donors and 34.6 among hepatitis cases, so a single
overall median would pull a hepatitis case toward the donor center on the
assay most often missing. Each missing value is taken from the five nearest
subjects on the nine assays that subject does hold, measured on the
standardized scale, and the category is never read. The imputed rows are
flagged, and the sensitivity section below reports the two headline results
with those rows removed.

The release names gamma-glutamyl transferase `CGT`; the column is renamed
`GGT`. The cleaned panel, with the imputed values in place and the flag beside
them, is in `data/processed/hcv_panel.csv`.

## Standardization, decided before the labels were consulted

The ten assays are measured on different scales and their unstandardized
variances differ by four orders of magnitude. Gamma-glutamyl transferase and
creatinine carry 66.7 percent of the total variance between them, so an
unstandardized Euclidean distance is very nearly the distance in those two
assays alone.

![Each assay's share of the total unstandardized variance across the 608 subjects.](../figures/fig04_hcv_variance.png)

Both clusterings were computed. The unstandardized one agrees better with the
withheld labels, at an adjusted Rand index of 0.533 against 0.132, and it also
scores higher on Calinski-Harabasz, at 311.9 against 102.5. The standardized
version is used for every result below regardless. I made that choice on the
scales, before looking at the agreement, and kept it afterwards. Selecting the
preprocessing by how well the clusters match the withheld labels would put
those labels back into an analysis whose only claim is that it never saw
them. Both numbers are recorded so the cost of the decision is visible.

The unstandardized result does say something about the data. A distance
dominated by GGT and creatinine recovers the categories better than a distance
weighting all ten assays equally, which means the diagnostic signal is
concentrated in a few assays and equal weighting dilutes it. That is a reason
to expect the two-cluster result below.

## Three methods at the four clusters the labels imply

| Method | Clusters | Unassigned | Calinski-Harabasz | Silhouette | Davies-Bouldin | Adjusted Rand | Adjusted MI |
|---|---|---|---|---|---|---|---|
| k-means | 4 | 0 | 102.5 | 0.165 | 1.509 | 0.132 | 0.197 |
| DBSCAN, eps 4.0, min_samples 3 | 3 | 12 | 45.5 | 0.639 | 0.431 | 0.310 | 0.205 |
| Agglomerative, complete linkage | 4 | 0 | 63.8 | 0.664 | 0.703 | 0.249 | 0.194 |

The DBSCAN neighborhood parameters were searched over 36 values of eps and 6
minimum sample counts. A configuration was admitted only when it formed at
least two clusters and assigned at least four fifths of the subjects, because
a run that labels most points as outliers scores well on an internal index
while describing little of the data. Forty-five configurations formed two or
more clusters. The best admitted one, chosen by Calinski-Harabasz alone,
assigns 98.0 percent of subjects to three clusters and leaves 12 as outliers.
DBSCAN does not take a cluster count as a parameter, and no admitted
configuration with four clusters scored higher, so the three-cluster solution
is the one reported.

Calinski-Harabasz, the index the coursework used to select its clustering,
ranks k-means first by a factor of two. Agreement with the withheld labels
ranks k-means last. The index is the ratio of between-cluster to
within-cluster dispersion, and within-cluster dispersion is the quantity
k-means minimizes directly. On the same data k-means will usually score higher
on it than a method optimizing something else, whether or not its clusters
match anything outside the data. Silhouette and Davies-Bouldin
measure separation and compactness per point, and on this panel both rank
k-means last of the three, as the labels do. Where labels are unavailable,
more than one internal index should be reported for that reason.

![Calinski-Harabasz (a) and adjusted Rand index against the withheld labels (b) as the number of clusters varies, for k-means and for agglomerative clustering with complete linkage. The dotted line marks the four categories the labels hold.](../figures/fig05_hcv_cluster_sweep.png)

The k-means contingency table shows where the four-cluster solution fails.

| Cluster | Blood donor | Hepatitis | Fibrosis | Cirrhosis |
|---|---|---|---|---|
| 0 | 257 | 13 | 12 | 4 |
| 1 | 276 | 7 | 7 | 3 |
| 2 | 0 | 0 | 0 | 3 |
| 3 | 0 | 4 | 2 | 20 |

Clusters 0 and 1 split the donors into two groups of comparable size and hold
nearly every hepatitis and fibrosis case between them, so two of the four
available clusters are spent partitioning the healthy majority. Cluster 3 is a
cirrhosis group with a few advanced cases from the other categories, and
cluster 2 is three more cirrhosis cases. Hepatitis and fibrosis are not
recovered at all. Their 45 subjects sit inside the donor clusters with no
cluster holding a majority of either. Homogeneity of 0.277 against
completeness of 0.162 says the same thing numerically: a cluster tends to hold
one category, and a category is spread across clusters.

![The 608 subjects in the first two principal components of the standardized panel, colored by k-means cluster (a) and by the withheld category (b). The two components hold 42 percent of the standardized variance.](../figures/fig06_hcv_projection.png)

The projection is a picture of two components, not the ten-dimensional
geometry the clustering worked in. It does show the shape of the problem. The
cirrhosis cases lie far from the donor mass along the first component, while
hepatitis and fibrosis lie inside it.

## Where the recovery is strong

The coursework also compared a two-cluster solution against cirrhosis held
apart from every other category, and that comparison is the one that succeeds.

| Cluster | Every other category | Cirrhosis |
|---|---|---|
| 0 | 576 | 7 |
| 1 | 2 | 23 |

Two clusters, fitted with no reference to any label, recover cirrhosis against
everything else with an adjusted Rand index of 0.815, adjusted mutual
information of 0.665, homogeneity of 0.624 and completeness of 0.716.
Twenty-three of the 30 cirrhosis cases fall in a cluster of 25. This solution
also carries the highest Calinski-Harabasz score of any k tested, 114.7, so on
this dataset the internal index does point at the right number of clusters
even while it points at the wrong method.

![Median and quartiles of each assay, on a log scale, under the two-cluster k-means solution (a) and under the cirrhosis label (b).](../figures/fig07_hcv_group_profile.png)

The two panels of that figure are close to identical, which is the same result
the contingency table gives. The release states no units for any assay, so the
values below are quoted as released. The 25-subject cluster has a median
albumin of 32.0 against 42.2, a median bilirubin of 40.0 against 7.1, a median
cholinesterase of 2.47 against 8.39, a median aspartate aminotransferase of
99.0 against 25.6 and a median gamma-glutamyl transferase of 101.1 against
22.6. Albumin and cholinesterase are made by the liver and fall when it fails;
bilirubin is cleared by the liver and rises. Together they describe failing
synthetic liver function, and that is a large enough displacement in ten
dimensions for an unsupervised method to find.

## Sensitivity to the imputed rows and to the seed

Both headline results were refitted on the 582 complete cases alone, with the
26 imputed subjects removed.

| Comparison | All 608 | 582 complete cases |
|---|---|---|
| k-means, k = 4, against four categories, adjusted Rand | 0.132 | 0.137 |
| k-means, k = 2, against cirrhosis, adjusted Rand | 0.815 | 0.906 |

The four-cluster result is indifferent to the imputed rows. The two-cluster
result is not. On the complete cases 22 of 24 cirrhosis cases fall in a
cluster of 24, and on the full set 23 of 30 fall in a cluster of 25. The six
cirrhosis cases that were missing an assay make the difference, and five of
the six are placed with the donors once their missing value is filled in from
their neighbors. That is a limitation of the imputation and of the
complete-case result at once. The complete-case figure of 0.906 was measured
on the cirrhosis cases that happened to have a full panel, and a full panel is
itself a fact about how a case was worked up. The figure of 0.815 is measured
on every cirrhosis case in the release, with 31 of the 6,080 values in the
panel estimated.

k-means is run from ten initializations at one seed, and the two solutions
were refitted at ten seeds. The two-cluster solution against cirrhosis takes
one of two values, 0.815 at seven seeds and 0.743 at three, so the
initialization decides whether one or two more cirrhosis cases sit with the
donors. The four-cluster agreement ranges from 0.087 to 0.145 over the same
seeds, a range as wide as the value itself.

## What this establishes

Cirrhosis is separable from a routine laboratory panel without labels, at an
adjusted Rand index between 0.74 and 0.91 depending on which cases are counted
and where the initialization lands. The intermediate stages are not separable
by these methods at this sample size. Hepatitis and fibrosis differ from
healthy donors by less than the spread among donors themselves, so a
distance-based method partitions the donors before it isolates either disease
group.

## Limitations

The categories are not balanced and the imbalance decides the outcome. 533 of
608 subjects are donors, so a method minimizing within-cluster dispersion
spends its clusters where the points are. Any four-cluster comparison against
these labels is measuring a method's willingness to form a cluster of twenty
against a background of five hundred.

The 45 hepatitis and fibrosis subjects are too few to support a claim in either
direction about whether those stages are separable in principle. What is
established is that they are not separable at this sample size by these three
methods.

The DBSCAN parameters were selected by an internal index on the same data the
clustering was scored on. That is standard practice for an unsupervised method
with no held-out partition, and it is still selection on the evaluation data,
so DBSCAN's reported indices are the best of 45 configurations. The k-means
and agglomerative results take no such selection, having only the fixed
cluster count.

Complete linkage produces a single-subject cluster at every k from 2 to 6, so
the agglomerative agreement of 0.249 is achieved with one of its four clusters
holding one point. The linkage was kept because it is the one the coursework
specified; a different linkage would be a different comparison.

The four categories are a staging of one disease, so they are ordered. Every
measure used here treats them as unordered, which discards the information
that a fibrosis case misassigned to cirrhosis is a smaller error than one
misassigned to donor.
