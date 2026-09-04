# Notebooks

One notebook per analysis, in the order the coursework took each one: the
data and its cleaning, the inputs, the training, the results, and a short
discussion. The write-up with every table, figure and limitation is the
corresponding file in `docs/`; the notebook shows the steps and quotes the
headline numbers.

| Notebook | Analysis | Write-up |
|---|---|---|
| [01_classification.ipynb](01_classification.ipynb) | Readmission inside thirty days | [docs/01](../docs/01-supervised-classification.md) |
| [02_clustering.ipynb](02_clustering.ipynb) | Hepatitis C categories without labels | [docs/02](../docs/02-clustering.md) |
| [03_association_rules.ipynb](03_association_rules.ipynb) | Co-occurring diagnostic features | [docs/03](../docs/03-association-rules.md) |
| [04_regression.ipynb](04_regression.ipynb) | Daily alcohol intake from a liver panel | [docs/04](../docs/04-regression.md) |

The notebooks call the same functions the pipeline calls, `data.load_*`,
`splits.*`, `evaluate.*` and the helpers in `analysis/`, so the cleaning and
the model definitions cannot drift from the recorded results. Each notebook
redirects every output directory to a temporary location before importing the
pipeline, so running one writes nothing into `results/`, `figures/` or
`data/processed/`. Where a step takes longer than a notebook should, the grid
search and the permutation importance in the classification, the notebook
reads the committed table from `results/` and says so.

The committed notebooks carry their executed outputs. To execute them again:

```
.venv/Scripts/python -m pip install -r requirements-notebooks.txt
.venv/Scripts/python -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

The classification notebook fits three models on 72,597 encounters and takes
a few minutes; the other three finish in seconds.
