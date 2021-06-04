## Introduction

This is a TensorFlow 2.0 implementation of Graph Convolutional Networks for Assessment of Physical Rehabilitation Exercises.

## Dependencies/Setup

### Requirements
- Install Tensorflow 2.0 from https://www.tensorflow.org/install

```bash
pip install -r requirements.txt
```


## Running the demo
```
git clone https://github.com/nywang16/Pixel2Mesh.git
cd Data/
```
Download the pre-trained model and unzip to the `Data/` folder.
* https://drive.google.com/file/d/1gD-dk-XrAa5mfrgdZSunjaS6pUUWsZgU/view?usp=sharing(link model)
```
unzip checkpoint.zip
```

#### predict score of exercise
    python demo.py --image Data/examples/plane.png


#### Input image, output mesh(input data)
<img src="./Docs/images/plane.png" width = "330px" /><img src="./Docs/images/plane.gif" />



## Dataset

We used the [KIMORE](https://vrai.dii.univpm.it/content/kimore-dataset) dataset, and [UI-PRMD](https://webpages.uidaho.edu/ui-prmd/):</br>

The training/testing split can be found in supplymentary material. </br>
    

# Training
    python train.py
You can change the training data, learning rate and other parameters by editing `train.py`

The total number of training epoch is 2000; the learning rate is initialized as ***.

# Evaluation
The evaluation code was released, please refer to `eval_testset.py` for more details.



# Contact

