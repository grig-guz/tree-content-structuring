# Domain-Independent Neural Text Structuring 

## Requirements

* Python (3.6+)
* [Pytorch](https://pytorch.org/) (1.3.0+)
* [dgl](https://www.dgl.ai/) (0.4.2 strictly)
* [Transformers](https://huggingface.co/transformers/) (3.0.2)

## Running experiments
1. Create the folder named "data".
2. Download the pickled versions of MEGA-DT [here](https://www.todo) (100k train, 250k train, 5k val, 15k test), and place it in the "data" folder.
3. Run the train/testing script as described below. Each scripts accepts a single numeric (1 or 2) indicating whether the model should be trained on 100k or 250k version of MEGA-DT.

#### Dependency Model
To train/evaluate the dependency model, 
 ```bash
bash scripts/train_dep.sh dataset_id
bash scripts/eval_dep.sh dataset_id
```
#### Pointer Model
 ```bash
bash scripts/train_pointer.sh dataset_id
bash scripts/eval_pointer.sh dataset_id
 ```
 #### Dependency no-pointer Baseline
 ```bash
bash scripts/train_dep_treetrain_baseline.sh  dataset_id
bash scripts/eval_dep_treetrain_baseline.sh dataset_id
 ```
#### Language Model Decoding Baseline
 ```bash
bash scripts/eval_lm_baseline.sh
 ```

## Configuration
You can set hyperparameters and device type in the training/testing scripts for each model individually. The parameter values used in our experiments are already specified there.

## Citation
 ```
@inproceedings{guz-carenini-2020-towards,
    title = "Towards Domain-Independent Text Structuring Trainable on Large Discourse Treebanks",
    author = "Guz, Grigorii  and
      Carenini, Giuseppe",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.281",
    pages = "3141--3152",
}
 ```
