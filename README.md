[![PyPI version](https://badge.fury.io/py/be-great.svg)](https://badge.fury.io/py/be-great) [![Downloads](https://static.pepy.tech/badge/be-great)](https://pepy.tech/project/be-great)

[//]: # (![Screenshot]&#40;https://github.com/kathrinse/be_great/blob/main/imgs/GReaT_logo.png&#41;)
<p align="center">
<img src="https://github.com/kathrinse/be_great/raw/main/imgs/GReaT_logo.png" width="326"/>
</p>

<p align="center">
<strong>Generation of Realistic Tabular data</strong>
<br> with pretrained Transformer-based language models
</p>

&nbsp;
&nbsp;
&nbsp;

Our GReaT framework leverages the power of advanced pretrained Transformer language models to produce high-quality synthetic tabular data. Generate new data samples effortlessly with our user-friendly API in just a few lines of code. Please see our [publication](https://openreview.net/forum?id=cEygmQNOeI) for more details. 

## GReaT Installation

The GReaT framework can be easily installed using with [pip](https://pypi.org/project/pip/) - requires a Python version >= 3.9: 
```bash
pip install be-great
```



## GReaT Quickstart

In the example below, we show how the GReaT approach is used to generate synthetic tabular data for the California Housing dataset.
```python
from be_great import GReaT
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

model = GReaT(llm='distilgpt2', batch_size=32,  epochs=50, fp16=True)
model.fit(data)
synthetic_data = model.sample(n_samples=100)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kathrinse/be_great/blob/main/examples/GReaT_colab_example.ipynb)

### Imputing a sample
GReaT also features an interface to impute, i.e., fill in, missing values in arbitrary combinations. This requires a trained ``model``, for instance one obtained using the code snippet above, and a ```pd.DataFrame``` where missing values are set to NaN.
A minimal example is provided below:
```python
# test_data: pd.DataFrame with samples from the distribution
# model: GReaT trained on the data distribution that should be imputed

# Drop values randomly from test_data
import numpy as np
for clm in test_data.columns:
    test_data[clm]=test_data[clm].apply(lambda x: (x if np.random.rand() > 0.5 else np.nan))

imputed_data = model.impute(test_data, max_length=200)
```


### Sampling and imputing from Variable Row-numbers Under One Cluster (VRUOC)
For example, the locations of all housings in the Californian table are geographically clusterable.
A `cluster` can be viewed a geographical unit, or even a county.
Therefore, a possible scenario occurs that requires sampling and conditional imputing in real life can be: **how can I know the number of housing at sales in a county? how is the current market distributed in the measure of the geomgraphy of california?** ...
In this repo, I set `cluster_string` to the indicator of cluster in the dimension of geographical distance.
Also, I append columns e.g. `cluster_string_count` to indicate the total amount of a value of `cluster_string`, and `cluster_string_index` to indicate sequential informatio to the game.

This section I do the following 2 tasks:
- test if the configuration for VRUOC scenario tackles the challenge of `counts` and `index` in clustering issue. (cf.`examples/cal_dataframe_area_clusters_in_letters_counted_indexed.csv`).
- if given a cluster_stirng with total amounts of its items, can it return correctly with index column？

If you're interested in such scenario (which I is in, too), please go to examples/Example_California_Housing_VRUOC.ipynb.

## GReaT Citation 
If you use GReaT, please link or cite our work:

``` bibtex
@inproceedings{borisov2023language,
  title={Language Models are Realistic Tabular Data Generators},
  author={Vadim Borisov and Kathrin Sessler and Tobias Leemann and Martin Pawelczyk and Gjergji Kasneci},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=cEygmQNOeI}
}
```

## GReaT Acknowledgements

We sincerely thank the [HuggingFace](https://huggingface.co/) :hugs: framework. 
