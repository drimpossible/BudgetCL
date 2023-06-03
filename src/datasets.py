import argparse
import copy
import pickle
import random
from os.path import exists

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

import opts
from sampling import select_samples
from training import train


class CLImageFolder(object):
    """
    Loading a continual learning dataloader from ordering specified in the .txt files for a given dataset
      ------
      opt: configuration file
      ------
      """

    def __init__(self, opt):
        super(CLImageFolder, self).__init__()
        self.kwargs = {
            'num_workers': opt.num_workers,
            'batch_size': opt.train_batch_size,
            'pin_memory': True,
            'shuffle': False,
            'drop_last': False
        }
        self.opt = opt
        self.curr_val_size, self.current_sample = 0, 0  # The total number of val/total samples

        # [self._classes, self._pretrain_paths, self._cl_{train/test}_paths] contain the full set of paths and classes -- do not use these for learning/sampling/anything. These are there for checks and ease-of-use.
        # Only self.curr_classes, self.curr_paths and self.{val/test}_paths should be exposed to the model.

        if opt.dataset == 'Imagenet2K':
            with open("../scripts/imagenet_folder_to_cls.pkl", 'rb') as f:
                self._classes = pickle.load(f)  #{}

            self._pretrain_paths, self._preval_paths, self._pretest_paths, self._pretrain_labels, self._preval_labels, self._pretest_labels = self.populate_paths(
                trainpath=opt.order_file_dir + '/pretrain.txt',
                valpath=opt.order_file_dir + '/preval.txt',
                testpath=opt.order_file_dir + '/pretest.txt')
            self.curr_classes = copy.deepcopy(list(self._classes.values(
            )))  # At timestep 0, curr_paths is pretrain_paths

            assert (opt.dset_mode in ['class_incremental', 'data_incremental'])

            self._cltrain_paths, self._clval_paths, self._cltest_paths, self._cltrain_labels, self._clval_labels, self._cltest_labels = self.populate_paths(
                trainpath=opt.order_file_dir + '/ordering_' + opt.dset_mode +
                '.txt',
                valpath=opt.order_file_dir + '/val.txt',
                testpath=opt.order_file_dir + '/test.txt')
            self.class_sizes = np.load(opt.order_file_dir + '/class_sizes.npy')

            # We usually split the curr_paths into train and val, excluding pretrain data. Currently 0 new data as everything is pretraining.
            self.curr_paths, self.val_paths, self.test_paths, self.fulltest_paths, self.curr_labels, self.val_labels, self.test_labels, self.fulltest_labels = copy.deepcopy(
                self._pretrain_paths), copy.deepcopy(
                    self._preval_paths), [], copy.deepcopy(
                        self._cltest_paths), copy.deepcopy(
                            self._pretrain_labels), copy.deepcopy(
                                self._preval_labels), [], copy.deepcopy(
                                    self._cltest_labels)
            self.classweights = np.bincount(np.array(self.curr_labels))
            assert (not torch.any(
                torch.eq(torch.from_numpy(self.classweights),
                         torch.zeros(len(self.classweights)))).item()
                    ), 'Class cannot have 0 samples'

            # Each timestep will have its {train/val/test/mem}loader (timestep 0 won't have memloader)
            assert (len(self.curr_paths) > 0 and len(self.val_paths) > 0
                    and len(self._pretest_paths) > 0
                    and len(self.test_paths) == 0), 'Path list cannot be empty'
            assert (len(self.curr_paths) == len(self.curr_labels)
                    and len(self.val_paths) == len(self.val_labels)
                    and len(self._pretest_paths) == len(self._pretest_labels)
                    and len(self.fulltest_paths) == len(self.fulltest_labels))

            self.trainloader = self.get_loader(opt,
                                               paths=self.curr_paths,
                                               labels=self.curr_labels,
                                               train=True)
            self.trainloader_eval = self.get_loader(opt,
                                                    paths=self.curr_paths,
                                                    labels=self.curr_labels,
                                                    train=False)
            self.valloader = self.get_loader(opt,
                                             paths=self.val_paths,
                                             labels=self.val_labels,
                                             train=False)
            self.pretestloader = self.get_loader(
                opt,
                paths=copy.deepcopy(self._pretest_paths),
                labels=copy.deepcopy(self._pretest_labels),
                train=False)
            self.fulltestloader = DataLoader(
                ListFolder(data_dir=opt.data_dir,
                           imgpaths=self.fulltest_paths,
                           labels=self.fulltest_labels,
                           transform=ClassificationPresetEval(
                               crop_size=self.opt.crop_size,
                               resize_size=self.opt.crop_size + 32)),
                **self.kwargs)
            self.trainlen, self.vallen, self.pretestlen, self.fulltestlen = len(
                self.curr_paths), len(self.val_paths), len(
                    self._pretest_paths), len(self.fulltest_paths)

        elif opt.dataset == 'CGLM':
            assert (opt.dset_mode == 'time_incremental')
            self._classes = {}
            self._cltrain_paths, _, self._test_paths, self._cltrain_labels, _, self._test_labels = self.populate_paths(
                trainpath=opt.order_file_dir + '/train.txt',
                valpath=opt.order_file_dir + '/test.txt',
                testpath=opt.order_file_dir + '/test.txt')
            self.curr_classes = copy.deepcopy(list(self._classes.values(
            )))  # At timestep 0, curr_paths is pretrain_paths

            # We usually split the curr_paths into train and val, excluding pretrain data. Currently 0 new data as everything is pretraining.
            self.curr_paths, self.curr_labels = [], []

            # Each timestep will have its {train/val/test/mem}loader (timestep 0 won't have memloader)
            assert (len(self._test_paths) > 0), 'Path list cannot be empty'
            self.testloader = DataLoader(
                ListFolder(imgpaths=copy.deepcopy(self._test_paths),
                           labels=self._test_labels,
                           transform=ClassificationPresetEval(
                               crop_size=self.opt.crop_size,
                               resize_size=self.opt.crop_size + 32)),
                **self.kwargs)
            self.testlen = len(self._test_paths)

    def populate_paths(self, trainpath, valpath, testpath):
        # Add training paths (order important -- never shuffle) along with mapping to ensure dataloader always gets 0-n as class indexes
        filename = trainpath
        fp = open(filename, 'r')
        trainimgs = fp.readlines()
        trainlabels = []
        for i in range(len(trainimgs)):
            if self.opt.dataset == 'Imagenet2K':
                clsname = trainimgs[i].strip().split('/')[-2]
                if clsname not in self._classes:
                    self._classes[clsname] = len(self._classes)
                trainlabels.append(self._classes[clsname])
                trainimgs[i] = trainimgs[i].strip()
            elif self.opt.dataset == 'CGLM':
                clsname = trainimgs[i].strip().split('\t')[0]
                if clsname not in self._classes:
                    self._classes[clsname] = len(self._classes)
                trainlabels.append(self._classes[clsname])
                trainimgs[i] = trainimgs[i].strip().split('\t')[1]
        fp.close()

        # Add val paths (order important -- never shuffle) and assert that classes for all samples are in the mapping function
        filename = valpath
        fp = open(filename, 'r')
        valimgs = fp.readlines()
        vallabels = []
        for i in range(len(valimgs)):
            if self.opt.dataset == 'Imagenet2K':
                clsname = valimgs[i].strip().split('/')[-2]
                vallabels.append(self._classes[clsname])
                valimgs[i] = valimgs[i].strip()
            elif self.opt.dataset == 'CGLM':
                clsname = valimgs[i].strip().split('\t')[0]
                vallabels.append(self._classes[clsname])
                valimgs[i] = valimgs[i].strip().split('\t')[1]
            assert (clsname
                    in self._classes), 'Class ' + str(clsname) + ' not found!'
        fp.close()

        # Add testing paths (order important -- never shuffle) and assert that classes for all samples are in the mapping function
        filename = testpath
        fp = open(filename, 'r')
        testimgs = fp.readlines()
        testlabels = []
        for i in range(len(testimgs)):
            if self.opt.dataset == 'Imagenet2K':
                clsname = testimgs[i].strip().split('/')[-2]
                testlabels.append(self._classes[clsname])
                testimgs[i] = testimgs[i].strip()
            elif self.opt.dataset == 'CGLM':
                clsname = testimgs[i].strip().split('\t')[0]
                testlabels.append(self._classes[clsname])
                testimgs[i] = testimgs[i].strip().split('\t')[1]
            assert (clsname
                    in self._classes), 'Class ' + str(clsname) + ' not found!'
        fp.close()

        assert (len(trainlabels) == len(trainimgs)
                and len(vallabels) == len(valimgs)
                and len(testlabels) == len(testimgs))
        return trainimgs, valimgs, testimgs, trainlabels, vallabels, testlabels

    def get_loader(self, opt, paths, labels, train=True):
        # drop_last is False and shuffle is True while training, vice-versa in other cases (batchnorm cries if last batch has 1 sample)
        if train:
            self.kwargs['batch_size'] = self.opt.train_batch_size
            self.kwargs['shuffle'] = True
        else:
            self.kwargs['batch_size'] = self.opt.test_batch_size
            self.kwargs['shuffle'] = False

        if train:
            return DataLoader(
                ListFolder(data_dir=opt.data_dir,
                           imgpaths=paths,
                           labels=labels,
                           transform=ClassificationPresetTrain(
                               crop_size=self.opt.crop_size)), **self.kwargs)
        else:
            return DataLoader(
                ListFolder(data_dir=opt.data_dir,
                           imgpaths=paths,
                           labels=labels,
                           transform=ClassificationPresetEval(
                               crop_size=self.opt.crop_size,
                               resize_size=self.opt.crop_size + 32)),
                **self.kwargs)

    def get_next_timestep_dataloader(self, opt):
        assert (
            (self.opt.num_classes_per_timestep > 0
             and self.opt.increment_size == 0)
            or (self.opt.num_classes_per_timestep == 0
                and self.opt.increment_size > 0)
        ), 'Either increment by class sizes or by constant number of samples'

        # Load new samples into the curr_paths -- either by incrementing per-class or per-sample (different as classes may have very unequal distribution of samples)
        assert (self.opt.dset_mode
                in ['class_incremental', 'data_incremental'])
        if self.opt.dset_mode == 'class_incremental':
            assert (self.opt.increment_size == 0)
            increment_size, start = 0, (self.opt.timestep -
                                        1) * self.opt.num_classes_per_timestep
            for i in range(self.opt.num_classes_per_timestep):
                increment_size += self.class_sizes[start + i]
        else:
            assert (self.opt.num_classes_per_timestep == 0)
            increment_size = self.opt.increment_size

        new_paths = self._cltrain_paths[self.
                                        current_sample:self.current_sample +
                                        increment_size]
        new_labels = self._cltrain_labels[self.
                                          current_sample:self.current_sample +
                                          increment_size]
        self.current_sample = self.current_sample + increment_size

        # Update the curr_classes dictionary to the new samples
        newclasses = []
        for i in range(len(new_labels)):
            clsname = new_labels[i]
            if clsname not in self.curr_classes:
                self.curr_classes.append(clsname)
                newclasses.append(clsname)
        self.expand_size = len(newclasses)

        # Add val images for all the new classes
        if self.opt.dataset == 'Imagenet2K':
            for i in range(len(self._clval_paths)):
                clsname = self._clval_labels[i]
                if clsname in newclasses:
                    self.val_paths.append(self._clval_paths[i])
                    self.val_labels.append(self._clval_labels[i])

            # Add test images for all the new classes
            for i in range(len(self._cltest_paths)):
                clsname = self._cltest_labels[i]
                if clsname in newclasses:
                    self.test_paths.append(self._cltest_paths[i])
                    self.test_labels.append(self._cltest_labels[i])

        self.curr_paths.extend(new_paths)
        self.curr_labels.extend(new_labels)
        assert (len(self.curr_labels) == len(self.curr_paths))
        print(len(self.curr_paths), len(self.curr_labels))

        if (not exists(self.opt.log_dir + '/' + self.opt.exp_name + '/' +
                       str(self.opt.timestep + 1) + '/last.ckpt')):
            self.classweights = np.bincount(np.array(self.curr_labels))
            assert (not torch.any(
                torch.eq(torch.from_numpy(self.classweights),
                         torch.zeros(len(self.classweights)))).item()
                    ), 'Class cannot have 0 samples'
            curr_paths, curr_labels = self.sample_data()

            # Each timestep will have its {train/val/test}loader
            if self.opt.dataset == 'Imagenet2K':
                assert (len(self.curr_paths) > 0 and len(self.val_paths) > 0
                        and len(
                            self.test_paths) > 0), 'Path list cannot be empty'
                self.trainloader = self.get_loader(opt,
                                                   paths=self.curr_paths,
                                                   labels=self.curr_labels,
                                                   train=True)
                self.trainloader_eval = self.get_loader(
                    opt,
                    paths=self.curr_paths,
                    labels=self.curr_labels,
                    train=False)
                self.valloader = self.get_loader(
                    opt,
                    paths=copy.deepcopy(self.val_paths),
                    labels=copy.deepcopy(self.val_labels),
                    train=False)
                self.testloader = self.get_loader(
                    opt,
                    paths=copy.deepcopy(self.test_paths),
                    labels=copy.deepcopy(self.test_labels),
                    train=False)
                self.trainlen, self.vallen, self.testlen = len(
                    curr_paths), len(self.val_paths), len(self.test_paths)
            elif self.opt.dataset == 'CGLM':
                assert (len(curr_paths) > 0 and len(curr_labels)
                        == len(curr_paths)), 'Path list cannot be empty'
                self.trainloader = self.get_loader(opt,
                                                   paths=curr_paths,
                                                   labels=curr_labels,
                                                   train=True)
                self.trainloader_eval = self.get_loader(
                    opt,
                    paths=self.curr_paths,
                    labels=self.curr_labels,
                    train=False)
                self.trainlen = len(curr_paths)

    def sample_data(self):
        if len(self.curr_paths) < (
                self.opt.total_steps * self.opt.train_batch_size
        ):  # Edge case-- if sampling size is higher than the all stored data then we don't select, simply add all samples. Won't be the case in most experiments.
            remaining = self.opt.total_steps * self.opt.train_batch_size
            curr_paths = copy.deepcopy(self.curr_paths)
            curr_labels = copy.deepcopy(self.curr_labels)
            remaining -= len(self.curr_paths)

            while remaining > len(self.curr_paths):
                curr_paths += copy.deepcopy(self.curr_paths)
                curr_labels += copy.deepcopy(self.curr_labels)
                remaining -= len(self.curr_paths)

            # Use lastk for the rest of the samples (likely negligible in size compared to the sampling done before, tested as using lastk performs similar to everything else)
            curr_paths += copy.deepcopy(self.curr_paths[len(self.curr_paths) -
                                                        (remaining):])
            curr_labels += copy.deepcopy(
                self.curr_labels[len(self.curr_paths) - (remaining):])
        else:
            if self.opt.sampling_mode == 'recency_biased':
                remaining = self.opt.total_steps * self.opt.train_batch_size
                curr_paths = copy.deepcopy(self.curr_paths[:(remaining)])
                curr_labels = copy.deepcopy(self.curr_labels[:(remaining)])
                for idx in range(
                    (self.opt.total_steps * self.opt.train_batch_size),
                        len(self.curr_paths)):
                    dice = random.randint(0, idx)
                    if dice < (self.opt.reservoir_alpha *
                               self.opt.total_steps *
                               self.opt.train_batch_size):
                        curr_paths[dice] = self.curr_paths[idx]
                        curr_labels[dice] = self.curr_labels[idx]
            elif self.opt.sampling_mode == 'lastk':
                remaining = self.opt.total_steps * self.opt.train_batch_size
                curr_paths = copy.deepcopy(
                    self.curr_paths[len(self.curr_paths) - (remaining):])
                curr_labels = copy.deepcopy(
                    self.curr_labels[len(self.curr_paths) - (remaining):])
            elif self.opt.sampling_mode == 'class_balanced':
                # Weights set to inverse frequency of samples per class in this timesteps.
                weights = np.array([
                    self.classweights[self.curr_labels[i]].item()
                    for i in range(len(self.curr_labels))
                ])
                probabilities = weights / np.sum(weights)
                idxes = np.random.choice(range(0, probabilities.shape[0]),
                                         size=self.opt.total_steps *
                                         self.opt.train_batch_size,
                                         p=probabilities,
                                         replace=True)
            elif self.opt.sampling_mode == 'uniform':
                probabilities = np.ones(len(self.curr_labels)) / len(
                    self.curr_labels)
                idxes = np.random.choice(range(0, probabilities.shape[0]),
                                         size=self.opt.total_steps *
                                         self.opt.train_batch_size,
                                         p=probabilities,
                                         replace=True)
            elif self.opt.sampling_mode in [
                    'herding', 'kmeans', 'unc_lc', 'max_loss'
            ]:  ## These still being cleaned
                idxes = select_samples(
                    opt=self.opt,
                    num_samples=self.opt.total_steps *
                    self.opt.train_batch_size,
                    class_balanced=True
                )  # Without class balancing performance is even crappier

            curr_paths = np.copy(np.array(self.curr_paths))
            curr_labels = np.copy(np.array(self.curr_labels))
            curr_paths = curr_paths[idxes]
            curr_labels = curr_labels[idxes]
            curr_paths = list(curr_paths)
            curr_labels = list(curr_labels)

        return curr_paths, curr_labels


class ListFolder(Dataset):

    def __init__(self,
                 data_dir,
                 imgpaths,
                 labels,
                 transform=None,
                 return_paths=False):
        super(ListFolder, self).__init__()
        # Get image list and weights per index
        self.data_dir = data_dir
        self.image_paths = imgpaths
        self.labels = torch.from_numpy(np.array(labels))

        # Check for correct sizes
        assert (len(imgpaths) == len(labels))
        self.transform = transform
        self.return_paths = return_paths

    def __getitem__(self, index):
        path = self.data_dir + '/' + self.image_paths[index]
        label = self.labels[index]
        sample = pil_loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        assert (len(self.image_paths) == self.labels.size(0)
                ), 'Length of image path array and labels different'
        return len(self.image_paths)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ClassificationPresetTrain:

    def __init__(self,
                 crop_size,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 interpolation=InterpolationMode.BILINEAR,
                 hflip_prob=0.5,
                 auto_augment_policy=None,
                 random_erase_prob=0.0):
        trans = [
            transforms.RandomResizedCrop(crop_size,
                                         interpolation=interpolation)
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(
                    autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(
                        interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(policy=aa_policy,
                                            interpolation=interpolation))
        trans.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:

    def __init__(
            self,
            crop_size,
            resize_size=256,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)


if __name__ == '__main__':
    # Test dataloading
    opt = opts.parse_args()
    d = CLImageFolder(opt=opt)
    d.get_next_timestep_dataloader(opt=opt)
    for (inputs, labels) in d.valloader:
        print(labels, inputs.size())
