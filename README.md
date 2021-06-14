## Introduction

This is a TensorFlow 2.0 implementation of our paper.
Inside the Data folder we have a demo exercise video of Kimore dataset exercise 3, corresponding skeleton data namely `input.csv` and the actual score `label.csv`.

## Dependencies/Setup

### Requirements
- Python3 (>3.5)
- Install Tensorflow 2.0 from https://www.tensorflow.org/install
- Other Python libraries can be installed by `pip install -r requirements.txt`

## Running the demo
You can use the following commands to run the demo.

```shell
python demo.py [--skeleton data ${PATH_TO_DATA}] [--label ${PATH_TO_Label}]

# Alternative way
python demo.py
```
The output is the predicted label for the demo exercise.

## Dataset

We used the [KIMORE](https://vrai.dii.univpm.it/content/kimore-dataset) dataset, and [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/):</br>

    

# Training
To train the model you have to first download the dataset from above link. The data and labels of an exercise have to be inside a folder. Then run the train.py file to train the model. You can change the optimizer, learning rate and other parameters by editing `train.py`. The total number of training epoch is 2000; the learning rate is initialized as 0.0001.
You can train the model following command.
```shell
python train.py
```
