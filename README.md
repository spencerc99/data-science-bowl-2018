# data-science-bowl-2018
## Setup
Download the data from https://www.kaggle.com/c/data-science-bowl-2018/data. Download training data, and create folder structure as follows:
- data
  - train
    - input
      - unzip stage1_train in here
    - labels
      - unzip stage1_train_labels.csv
  - test
    - unzip test in here

## Approach

## Running code
### Training
run `python train.py <model_name> <weights_checkpoint>`
### Prediction
run `python create_submission.py <path_to_weights_file>`

## References
* https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3