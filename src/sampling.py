import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F


def select_samples(opt, num_samples, class_balanced=False):
    """
    Selects a subset of samples from the training data for active learning.

    Args:
        opt (argparse.Namespace): The command-line arguments.
        num_samples (int): The number of samples to select.
        class_balanced (bool): Whether to balance the samples across classes.

    Returns:
        np.ndarray: The indices of the selected samples.
    """
    if opt.timestep > 1:
        if not class_balanced:
            if opt.sampling_mode in ['herding', 'kmeans']:
                feats = np.load(opt.log_dir + '/' + opt.exp_name + '/feats_' +
                                str(opt.timestep - 1) + '_train.npy')
                labels = np.load(opt.log_dir + '/' + opt.exp_name +
                                 '/labels_' + str(opt.timestep - 1) +
                                 '_train.npy')
                if opt.sampling_mode == 'herding':
                    sampled_idxes = herding(feats, num_samples)
                elif opt.sampling_mode == 'kmeans':
                    sampled_idxes = kmeans(feats, num_samples)
            elif opt.sampling_mode in ['unc_lc', 'max_loss']:
                probs, labels = np.load(
                    opt.log_dir + '/' + opt.exp_name + '/predprobs_' +
                    str(opt.timestep - 1) +
                    '_train.npy'), np.load(opt.log_dir + '/' + opt.exp_name +
                                           '/labels_' + str(opt.timestep - 1) +
                                           '_train.npy')
                sampled_idxes = samplewise_losses(opt, probs, labels,
                                                  num_samples)
        else:
            labels = np.load(opt.log_dir + '/' + opt.exp_name + '/labels_' +
                             str(opt.timestep - 1) + '_train.npy')
            if opt.sampling_mode in ['herding', 'kmeans']:
                feats = np.load(opt.log_dir + '/' + opt.exp_name + '/feats_' +
                                str(opt.timestep - 1) + '_train.npy')
            elif opt.sampling_mode in ['unc_lc', 'max_loss']:
                probs = np.load(opt.log_dir + '/' + opt.exp_name +
                                '/predprobs_' + str(opt.timestep - 1) +
                                '_train.npy')

            classweights = np.bincount(np.array(labels))
            num_samples_per_class = (
                num_samples // len(classweights)
            ) + 1  # If not divisible, then store slightly more than num_samples
            sampled_idxes = []
            for cls in len(classweights):
                idx = np.where(labels == cls)[0]
                if idx.shape[
                        0] < num_samples_per_class:  # Can't do anything but undersample
                    sampled_idxes += idx.tolist()
                else:
                    if opt.sampling_mode == 'herding':
                        cls_idx = herding(feats[idx], num_samples_per_class)
                    elif opt.sampling_mode == 'kmeans':
                        cls_idx = kmeans(feats[idx], num_samples_per_class)
                    elif opt.sampling_mode in ['unc_lc', 'max_loss']:
                        cls_idx = samplewise_losses(opt, probs[idx],
                                                    labels[idx],
                                                    num_samples_per_class)
                    sampled_idxes += cls_idx.tolist()
            remaining = num_samples - len(sampled_idxes)
            if remaining > 0:
                add_idx = np.random.permutation(len(labels))[:remaining]
                sampled_idxes += add_idx.tolist()
    else:
        sampled_idxes = np.random.permutation(len(labels))[:num_samples]
    return sampled_idxes


### Note: Code borrowed from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/4991787c2ca19b364a5769e2c6afda53eed74020/inclearn/lib/herding.py
### Please refer there for details, license and updates


def closest_to_mean(features, nb_examplars):
    """
    Selects the `nb_examplars` features that are closest to the mean feature vector.

    Args:
        features (np.ndarray): The feature vectors to select from.
        nb_examplars (int): The number of feature vectors to select.

    Returns:
        np.ndarray: The indices of the selected feature vectors.
    """
    features = features / (np.linalg.norm(features, axis=0) + 1e-8)
    class_mean = np.mean(features, axis=0)

    return _l2_distance(features, class_mean).argsort()[:nb_examplars]


def samplewise_losses(opt, probs, labels, num_samples):
    """
    Selects a subset of samples from the input data based on the sampling mode.

    Args:
        opt (argparse.Namespace): The command-line arguments.
        probs (np.ndarray): The predicted probabilities for each sample.
        labels (np.ndarray): The true labels for each sample.
        num_samples (int): The number of samples to select.

    Returns:
        np.ndarray: The indices of the selected samples.
    """
    if opt.sampling_mode == 'unc_lc':
        var_ratio = np.max(probs, axis=1)
        indexes = var_ratio.argsort()
    elif opt.sampling_mode == 'max_loss':
        prob_gt = np.array(
            [probs[j, labels[j]] for j in range(probs.shape[0])])
        indexes = prob_gt.argsort()
    return indexes[:num_samples]


def herding(features, nb_examplars):
    """
    Selects the `nb_examplars` features that are most representative of the input feature vectors using the herding algorithm.

    Args:
        features (np.ndarray): The feature vectors to select from.
        nb_examplars (int): The number of feature vectors to select.

    Returns:
        np.ndarray: The indices of the selected feature vectors.
    """
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0], ))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
            np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 100000000:  # 10M iters are way too high, expected to converge before that. Converges in <100 iters in checks.
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(
        herding_matrix == 0)[0]] = 100000000  # Some high number

    return herding_matrix.argsort()[:nb_examplars]


def kmeans(features, nb_examplars, k=5):
    """Samples examplars for memory according to KMeans.

    :param features: The image features of a single class.
    :param nb_examplars: Number of images to keep.
    :param k: Number of clusters for KMeans algo, defaults to 5
    :return: A numpy array of indexes.
    """
    model = KMeans(n_clusters=k)
    cluster_assignements = model.fit_predict(features)

    nb_per_clusters = nb_examplars // k
    indexes = []
    for c in range(k):
        c_indexes = np.random.choice(np.where(cluster_assignements == c)[0],
                                     size=nb_per_clusters)
        indexes.append(c_indexes)

    return np.concatenate(indexes)


def _l2_distance(x, y):
    """
    Computes the squared L2 distance between two arrays.

    Args:
        x (np.ndarray): The first array.
        y (np.ndarray): The second array.

    Returns:
        np.ndarray: The squared L2 distance between `x` and `y`.
    """
    return np.power(x - y, 2).sum(-1)
