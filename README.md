# SSID: Stepping Stone Intrusion Detection (In Progress)

### Description

The code and techniques in the parent repository are modifications of code from **DeepCoFFEA: Improved Flow Correlation Attacks on Tor via Metric Learning and Amplification**. 

Improvements include faster training options (online triplet loss and randomly selected triplets), streamlined preprocessing code and more sophisticad voting mechanisms (see embedding\_combination folder). 

For an online triplet learning description, see: https://omoindrot.github.io/triplet-loss. 

For an adapted PyTorch implementation, see the 'pytorch\_implementation' folder. 


### Running the Attack 

To run the attack, follow the instructions from the DeepCoFFEA repository (copied below):

```
data_path = '/data/website-fingerprinting/datasets/CrawlE_Proc_100/' #input data path
out_file_path = '/data/seoh/CrawlE_Proc_100_files.txt' #output text file path to record flow pairs (i.e.,file_names)
threshold=20 # min number of packets per window in both ends
create_overlap_window_csv(data_path, out_file_path, threshold, 5, 11, 2) # We're using 11 windows and each window lasts 5 sec
```
Then, run the code
```
python filter.py
```
2. Create formatted-input pickles to feed them to triplet network.  Before running code, modify line 206-209 as follows.
```
data_path = '/data/website-fingerprinting/datasets/CrawlE_Proc/' #input data path
file_list_path = '/data/seoh/greaterthan50_final_burst.txt' # the path to txt file (we got from filter.py)
prefix_pickle_output = '/data/website-fingerprinting/datasets/new_dcf_data/crawle_new_overlap_interval' #output path
create_overlap_window_csv(data_path, file_list_path, prefix_pickle_output, 5, 11, 2) # We're using 11 windows and each window lasts 5 sec
```
Then, run the code. This code will create 11 pickle files in which each file carries the partial trace for each window.
```
python new_dcf_parse.py
```

3. Train FENs using pickle files created by new\_dcf\_parse.py. Configure the arguments as needed. For example,

```bash
python train_fens.py (--input <your_input_path> --model <your_model_path> --test <test_set_path>)
```

This script will save testing npz file in <test_set_path>, and trained models in <your_model_path>.

We stopped training when loss = 0.006 (DeepCorr set) and loss = 0.002 (our new set).

4. Evaluate trained FENs using trained two models and test dataset (eval\_dcf.py). Configure arguments as needed. For example,

```bash
python eval_dcf.py (--input <your_input_path> --model1 <your_model1_path> --model2 <your_model2_path> --output <your_output_path>)
```

The script above will generate TPRs, FPRs, and BDRs when using 9 out 11 window results.


However, if you'd like to use the more sophisticated window combinations, then run:

```
python eval_dcf_save_dataset.py
python embedding_combination/nn_embedding_classifier.py #Or, choose another script
```

This will save the window distance dataset to file and then uses the distances to train the classifier.






