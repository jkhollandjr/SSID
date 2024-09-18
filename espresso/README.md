# ESPRESSO Implementation

### Description

This folder contains an implemenation of the ESPRESSO flow correlation technique, recently presented at APNet `24 (see https://dl.acm.org/doi/10.1145/3663408.3665824). 

This approach uses transformers while aggregating the feature representation to better encode traffic patterns, outperformaing DeepCoFFEA on the DeepCoFFEA and stepping-stone datasets.

In order to use the base approach, see the sequence of scripts below:

```
python3 preprocessing_transformer.py
python3 train_espresso.py
python3 test_batched_drift_espresso.py
python3 nn_classifier.py
```

Note that the first script outputs the preprocessed data, while the second trains the feature embedder. Then, test\_batched\_drift_espresso.py compares the embeddings of the test set flows and nn\_classifier makes the final classification determination.


To use the 'live' training method, while allows the flows to be modified during the training process (while stored as Torch tensors), the instead use the following sequence of scripts:

```
python3 preprocessing_simple.py
python3 train_espresso_live.py
python3 test_espresso_batched_live.py
python3 nn_classifier.py
```
