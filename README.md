# BudgetCL 

This repository contains the code for the paper:

**Computationally Budgeted Continual Learning: What Does Matter?, CVPR 2023**  
[Ameya Prabhu*](https://drimpossible.github.io), [Hasan Abed Al Kader Hammoud*](https://scholar.google.com/citations?user=Plf1JSIAAAAJ&hl=en&oi=ao), [Puneet Dokania](https://puneetkdokania.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Ser-Nam Lim](https://sites.google.com/site/sernam), [Bernard Ghanem](https://www.bernardghanem.com/), [Adel Bibi](https://www.adelbibi.com/)

[[Arxiv](https://arxiv.org/abs/2303.11165)]
[[PDF](https://github.com/drimpossible/drimpossible.github.io/raw/master/documents/BudgetCL.pdf)]
[[Bibtex](https://github.com/drimpossible/BudgetCL/#citation)]

<p align="center">
  <img src="https://github.com/drimpossible/BudgetCL/assets/74360386/5af6d831-a8ea-40f9-a213-15855fc9d509" width="250" alt="Figure which describes our conclusions">
</p>

## Getting started

Running our code requires 1x80GB A100 GPU with PyTorch >=1.13.

- Install all requirements required to run the code by:
 ```	
# First, activate a new virtual environment
$ pip install -r requirements.txt
 ```

## Setting up the Datasets

-  We provide a fast, direct mechanism to download and use our datasets in [this repository](https://github.com/hammoudhasan/CLDatasets).
-  Input the directory where the dataset was downloaded into `data_dir` field in `src/opts.py`.

## Recreating the dataset

### ImageNet2K

ImageNet2K is a dataset introduced by us, consists of 1K classes from the original dataset and 1K additional classes from ImageNet21K.

To create ImageNet2K dataset: 
- Download ImageNet1K `train` and `val` set from [here](https://www.image-net.org/download.php). Copy them to the `ImageNet2K` folder in `train` and `test` subdirectories respectively.
- Download ImageNetV2 dataset from [here](https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz), and copy this to the `ImageNet2K` folder as `val` subdirectory.
- Download ImageNet21K dataset from this [webpage](https://www.image-net.org/download.php).
- Now, to select the subset of ImageNet21K dataset and generate ordering files, go to `scripts` and run the script:
```
python select_subset_imagenet21k.py PATH_TO_IMAGENET21K PATH_TO_IMAGENET1K 1000 1200 ../clim2k/
```
- Finally, copy the new files from Imagenet21K to be included in ImageNet2K over to the folder by running:
```
python copy_imnet21k_to_imnet2k.py PATH_TO_IMAGENET21K PATH_TO_IMAGENET2K ../clim2k/
```

### Continual Google Landmarks V2 (CGLM)

- This dataset was introduced in [ACM](https://github.com/drimpossible/ACM), please follow instructions in that repository for curation details.

### Directory structure

- After setting up the datasets and the environment, the project root folder should look like this:

```
BudgetCL/
|–– data/
|–––– cglm/
|–––– clim2k/
|–– src/
|–– scripts/
|–– README.md
|–– requirements.txt
|–– .gitignore
|–– LICENSE
```

## Usage

To run any model specified in the paper one needs to simply modify the arguments, an example command below (reproduces our Naive baseline on CI-ImageNet2K):

```
python main.py --log_dir='../logs/sampling/' \
              --order_file_dir=../data/clim2k/order_files/ \
              --train_batch_size 1500 \
              --test_batch_size 1500 \
              --crop_size 224 \
              --dset_mode='class_incremental' \
              --num_classes_per_timestep=50 \
              --num_timesteps=20 \
              --increment_size=0 \
              --optimizer="SGD" \
              --model='resnet50' \
              --sampling_mode='uniform' \
              --model_type='normal' \
              --maxlr=0.1 \
              --total_steps=400 \
              --seed=1 \
              --weight_decay=0 \
              --clip=2.0 \
              --num_workers=8 \
              --momentum 0.9
```

Arguments you can tweak for your new cool CL pipeline/formulation/method:
- Model (`--model`)
- Dataset (`--order_file_dir`)
- Total optimization steps changing the compute budget (`--total_steps`)

To vary the number of timesteps change:

In DI-ImageNet2K and CGLM:
- Number of samples per timestep (`--num_samples_per_timestep`)

In CI-ImageNet2K, increase one and decrease the other:
- Number of classes per timestep (`--num_classes_per_timestep`)
- Number of tasks (`--num_timesteps`)
    

### Extension to New Datasets

- Create `train.txt`, `val.txt` and `test.txt` data orders for your new dataset.
- Add the dataset details in `src/datasets.py`
- Add the dataset folder name exactly to `src/opts.py`
- Run you model with `--dataset your_fav_dataset`!

## Reproducing Results

To replicate the complete set of experiments, 

- TBA



##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

Discovered mistakes:

- ACE Loss implemented by us deviated from the original work. The correct ACE loss function is unsuitable for our setting along with uniform sampling, being practically equivalent to CrossEntropy. However, we shall include the deviated ACE loss function in our code repository as it gave interesting results on CGLM and DI-ImageNet2K.

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
