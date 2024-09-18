# DeepCoFFEA Implementation

### Description

This folder contains a modified implementation of **DeepCoFFEA: Improved Flow Correlation Attacks on Tor via Metric Learning and Amplification**, which can be found at: https://github.com/traffic-analysis/deepcoffea.

Improvements include faster training options (online triplet loss and randomly selected triplets), streamlined preprocessing code, and more sophisticated voting mechanisms (see embedding\_combination folder). 

For a usefule online triplet learning description, see: https://omoindrot.github.io/triplet-loss. 


### Running the Attack 

To run the attack, follow the instructions from the DeepCoFFEA repository (reproduced below):

1. Gather flow pairs whose packet counts > threshold per window. Consider the arguments regarding the data path, the output path (which will save a text file with metadata) and the threshold. 

```
python filter.py (--data_path <your_data_path> --output_path <output_text_file> --threshold <thresh_value>)
```

2. Create formatted-input pickles to feed them to triplet network. Arguments include the data path, the metadata file list from the previous step, and an output folder to save the preprocessed data. Then, run the code. This code will create 11 pickle files in which each file carries the partial trace for each window.

```
python new_dcf_parse.py (--data_path <your_data_path> --file_list_path <previous_text_file> --prefix_pickle_output <preprocessed_data_folder>)
```

3. Train FENs using pickle files created by new\_dcf\_parse.py. Configure the arguments as needed. For example,

```bash
python train_fens.py (--input <preprocessed_data_folder> --model <your_model_path> --test <test_set_path>)
```

This script will save testing npz file in <test_set_path>, and trained models in <your_model_path>. Note that it defaults to randomly picking the negative example for the triplet selection, while the original technique picked semi-hard negatives.j

We stopped training when loss = 0.006 (DeepCorr set) and loss = 0.002 (our new set).

4. Evaluate the two trained FENs and test dataset (eval\_dcf.py). Configure arguments as needed. For example,

```bash
python eval_dcf.py (--test <your_input_path> --model1 <your_model1_path> --model2 <your_model2_path> --output <your_output_path>)
```

The script above will generate TPRs, FPRs, and BDRs when using 9 out 11 window results.


*However*, if you'd like to use the more sophisticated window combinations, then execute the following commands (this is done instead of running eval\_dcf.py). 

```
python eval_dcf_save_dataset.py (--test <your_input_path> --model1 <your_model1_path> --model2 <your_model2_path> --output <your_output_path>)
python embedding_combination/nn_embedding_classifier.py # Or, choose another script
```

This will save the window distance dataset to file and then uses the distances to train the classifier. Note that this technique was not used in the original DeepCoFFEA paper, but it will likely increase performance. 






