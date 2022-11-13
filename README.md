# Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises

This code is the official implementation of the following works (train + eval):
* S. Deb, M. F. Islam, S. Rahman and S. Rahman, "[Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9709340)," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 410-419, 2022, doi: 10.1109/TNSRE.2022.3150392.

![intro-1](https://user-images.githubusercontent.com/55605296/155846897-2d1d14b3-b09f-4f8f-b73b-17ec347a3de0.png)

Figure 1: Overview of existing vs. the proposed method. (a) The existing deep learning method applies CNN to the grid structure of stacked skeleton (body-joints) data. It performs consistently only with fixed-length input and ignores spatio-temporal topological structure from interaction among neighborhood joints. (b) Our proposed method employs STGCN to address the issues mentioned above. We offer extensions to STGCN using LSTM to extract rich spatio-temporal features and attend to different body-joints (as illustrated in colored joints) based on their role in the given exercise. It enables our method to guide users for better assessment scores.


![sk-1](https://user-images.githubusercontent.com/55605296/154412915-6039717f-1070-400e-a8df-e20c3e751195.png)

Figure 2: GCN based end-to-end models using (a-b) vanilla STGCN and (c-d) extended STGCN for rehabilitation exercise assessment. 'TC', \oplus and \odot denote temporal convolution, concatenation and element-wise multiplication, respectively. (b) and (d) illustrate the detailed components of the <font color="green">green</font> STGCN block of (a) and (c), respectively. 


### Data Preparation

We experimented on two skeleton based rehabilitation datasts: [KIMORE](https://vrai.dii.univpm.it/content/kimore-dataset) and [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/).
Before training and testing, for the convenience of fast data loading,
the datasets should be converted to the proper format.
Please download the pre-processed data from
[GoogleDrive](https://drive.google.com/drive/folders/1Vok-_HpLoqjKMybj9DHNxeG9C9yY44kM?usp=sharing)
and extract files with
```
cd st-gcn
unzip <path to Dataset.zip>
```

## Requirements

- Python3 (>3.5)
- Install Tensorflow 2.0 from https://www.tensorflow.org/install
- To install other libraries simply run `pip install -r requirements.txt`

## Files
* `train.py` : to perform training on Physical rehabilitation exercise
* `data_preprocessing.py` : preproces the data collected from dataset. It is mandatory to do some preprocessing before feeding it network.
* `graph.py` : It will generate skeleton graph from given data
* `stgcn_lstm.py` : build propose ST-GCN method 
* `demo.py` : perform a demo inference for given sample.


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
python train.py --ex Kimore_ex5 --epoch 2000 --batch_size 10
```

## Notes on experiment

![guidence_vis-1](https://user-images.githubusercontent.com/55605296/155735706-fbe9291f-b438-45bc-ad84-63f4b826cb00.jpg)

## Citation
If you use this code and model and dataset splits for your research, please consider citing:

```
@ARTICLE{deb-2022-graph,
  author={Deb, Swakshar and Islam, Md Fokhrul and Rahman, Shafin and Rahman, Sejuti},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises}, 
  year={2022},
  volume={30},
  number={},
  pages={410-419},
  doi={10.1109/TNSRE.2022.3150392}}
  ```
## Acknowledgment
We thank the authors and contributors of original [GCN implementation](https://github.com/tkipf/gcn).

## Contact
For any question, feel free to contact @
```
Swakshar Deb     : swakshar.sd@gmail.com
Md Fokhrul Islam : fokhrul.rmedu@gmail.com
```
