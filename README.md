# BudgetCL

This repository contains the code for the paper:

**Computationally Budgeted Continual Learning: What Does Matter?, CVPR 2023**  
[Ameya Prabhu*](https://drimpossible.github.io), [Hasan Abed Al Kader Hammoud*](https://github.com/hammoudhasan), [Puneet Dokania](https://puneetkdokania.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Ser-Nam Lim](https://sites.google.com/site/sernam), [Bernard Ghanem](https://www.bernardghanem.com/), [Adel Bibi](https://www.adelbibi.com/)

[[Arxiv](https://arxiv.org/abs/2303.11165)]
[[PDF](https://github.com/drimpossible/drimpossible.github.io/raw/master/documents/BudgetCL.pdf)]
[[Bibtex](https://github.com/drimpossible/BudgetCL/#citation)]

## Getting started

Running our code requires 1 80GB A100 GPU with Pytorch 2.0 but is backward compatible.

* Install all requirements required to run the code by:
 ```	
# First, activate a new virtual environment
$ pip3 install -r requirements.txt
 ```

## Setting up the Datasets

### Creating the Continual ImageNet2K dataset


### Creating the Continual Google Landmarks V2 dataset

To download Continual Google Landmarks V2 (CGLM), please follow instructions from repository for [ACM](https://github.com/drimpossible/ACM).

### Directory structure

After setting up the datasets and the environment, the project root folder should look like this:
```
BudgetCL/
|–– data/
|–––– cglm/
|–––– clim2k/
|–– src/
|–––– clean.sh
|–– scripts/
|–– README.md
|–– requirements.txt
|–– .gitignore
|–– LICENSE

```
## Running a Model



## Reproducing All Experiments

- TBA

##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

## Citation

We hope our benchmark and contributions are valuable to advance your work in continual learning! To cite our work:

```
@inproceedings{prabhu2023computationally,
  title={Computationally Budgeted Continual Learning: What Does Matter?},
  author={Prabhu, Ameya and Hammoud, Hasan Abed Al Kader and Dokania, Puneet and Torr, Philip HS and Lim, Ser-Nam and Ghanem, Bernard and Bibi, Adel},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```
