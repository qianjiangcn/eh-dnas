# EH-DNAS: End-to-End Hardware-aware Differentiable Neural Architecture Search
## Overview
This repository contains code for Paper [EH-DNAS: End-to-End Hardware-aware Differentiable Neural Architecture Search](https://arxiv.org/abs/2111.12299)

We now provide our collected hardware performance dataset using E2E-Perf, the pre-trained hardware loss, and the final searched model. We also provide code for training hardware loss, searching architecture and retraining. All on DARTS search space.


## Setup
To install the dependencies, change cudatoolkit version and run the following,
```bash
conda create --name myenv
conda activate myenv
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c nvidia
```
## Datasets
The datasets used in our experiments are availiable at the following links:
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](https://www.image-net.org/)


## Experiments

### The following commands run the experiments conducted in our paper.

#### Sample architectures
* We provide the sampled architectures as pickle file in dnas/DARTS/hwdataset_100w/ for directly use<br/>
* To resample and store input as pickle file in dnas/DARTS/hwdataset_100w/ (1000K train, 200K val, 200K test):<br/>
`python dnas/DARTS/darts_sampler.py`

#### Collect hardware performance dataset
* We provide the dataset output as pickle file in dnas/DARTS/hwdataset_100w/ for directly use<br/>

#### Train the hardware loss model
* We also provide the trained hardware loss models in dnas/acc/ and dnas/pip for directly use<br/>
* Train the hardware loss model with pipeline paradigm<br/>
`python hw_loss/main.py --config hw_loss/config_pip.yaml --para=pip`
* Train the hardware loss model with generic paradigm<br/>
`python hw_loss/main.py --config hw_loss/config_acc.yaml --para=acc`

#### Search on CIFAR10
* This will save the searching log in search-save-time/log.txt<br/>
* Search with hardware loss for pipeline paradigm<br/>
`python dnas/DARTS/train_search.py --hw_loss_type acc --hw_loss_rate 0.0001 --save acc --epochs 25`<br/>
* Search with hardware loss for generic paradigm<br/>
`python dnas/DARTS/train_search.py --hw_loss_type pip --hw_loss_rate 0.0005 --save pip --epochs 25`<br/>

#### Evaluate on CIFAR10
* Look through the search log and save searched architecture genotype in dnas/DARTS/genotypes.py with a $arch_name
* We saved our searched architectures in the paper in dnas/DARTS/genotypes.py as OURS_PIP and OURS_ACC for dirctly use
* Construct model for CIFAR10 and train from scratch to evaluate classification performance:<br/>
`python dnas/DARTS/train.py --arch $arch_name --cutout --auxiliary --save $arch_name`<br/>

#### Evaluate on ImageNet
* Train from scratch to evaluate classification performance:<br/>
`python dnas/DARTS/train_imagenet.py --batch_size 512 --epochs 150 --data $path_to_ImageNet --arch $arch_name --auxiliary --save img-$arch_name --parallel --report_freq 1000`<br/>

## Citation
If find this code helpful, please kindly cite the following paper <br/>
```
@article{jiang2021eh,
  title={EH-DNAS: End-to-End Hardware-aware Differentiable Neural Architecture Search},
  author={Jiang, Qian and Zhang, Xiaofan and Chen, Deming and Do, Minh N and Yeh, Raymond A},
  journal={arXiv preprint arXiv:2111.12299},
  year={2021}
}
```

## Acknowledgement
DARTS code originate from [DARTS repo](https://github.com/quark0/darts).
