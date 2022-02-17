# Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises

This code is the official implementation of the following works (train + eval):

* S. Deb, M. F. Islam, S. Rahman and S. Rahman, "Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, doi: 10.1109/TNSRE.2022.3150392.

![sk-1](https://user-images.githubusercontent.com/55605296/154412915-6039717f-1070-400e-a8df-e20c3e751195.png)

### Data Preparation

We experimented on two skeleton based rehabilitation datasts: [KIMORE](https://vrai.dii.univpm.it/content/kimore-dataset) and [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/).
Before training and testing, for the convenience of fast data loading,
the datasets should be converted to the proper format.
Please download the pre-processed data from
[GoogleDrive]()
and extract files with
```
cd st-gcn
unzip <path to processed-data.zip>
```

## Requirements

- Python3 (>3.5)
- Install Tensorflow 2.0 from https://www.tensorflow.org/install
- To install other libraries simply run `pip install -r requirements.txt`

## Files


Inside the Data folder we have a demo exercise video of Kimore dataset exercise 3, corresponding skeleton data `input.csv` and the actual score `label.csv`.



## Running instructions
You can use the following commands to run the demo.

```shell
python demo.py [--skeleton data ${PATH_TO_DATA}] [--label ${PATH_TO_Label}]

# Alternative way
python demo.py
```
The output is the predicted label for the demo exercise.

<!--## Dataset

We used the [KIMORE](https://vrai.dii.univpm.it/content/kimore-dataset) dataset, and [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/):</br>
-->

To train the model you have to first download the dataset from above link. The data and labels of an exercise have to be inside a folder. Then run the train.py file to train the model. You can change the optimizer, learning rate and other parameters by editing `train.py`. The total number of training epoch is 2000; the learning rate is initialized as 0.0001.
You can train the model following command.
```shell
python train.py
```
## Notes on experiment
![sk-1](https://raw.githubusercontent.com/fokhruli/STGCN-rehab/main/Figure/guidence_vis-1.png)

## Citation
If you use this code and model and dataset splits for your research, please consider citing:

```
@ARTICLE{9709340,
  author={Deb, Swakshar and Islam, Md Fokhrul and Rahman, Shafin and Rahman, Sejuti},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNSRE.2022.3150392}}
  ```
## Acknowledgment
We thank the authors and contributors of original [ST-GCN implementation](https://github.com/fizyr/keras-retinanet).
