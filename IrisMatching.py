import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from scipy.spatial import distance


def dim_reduction(train_set, train_label, test_set, dim, method='LDA'):
    """
        reduce the dimension of the feature vectpr

        Args:
            train_set: feature vectors of train set
            train_label: label for feature vecotr in train set
            test_set: feature vectors of test set
            dim: the expected dimension
            method: LDA/NULL/PCA

        Returns:
           train_set_transform, test_set_transform(train and test dataset after dimension reduction)
    """
    train = np.array(train_set)
    label = np.array(train_label)
    if method == 'LDA':
        clf = LinearDiscriminantAnalysis(n_components=dim)
        clf.fit(train, label)

        train_set_transform = clf.transform(train_set)
        test_set_transform = clf.transform(test_set)

    if method == 'PCA':
        pca = PCA(n_components=dim)
        pca.fit(train)

        train_set_transform = pca.transform(train_set)
        test_set_transform = pca.transform(test_set)

    return train_set_transform, test_set_transform


def minimal_offset_distance(fi, f, offset, mode):
    """
        reduce the dimension of the feature vector

        Args:
            train_set: feature vectors of train set
            train_label: label for feature vecotr in train set
            test_set: feature vectors of test set
            dim: the expected dimension
            method: LDA/NULL/PCA

        Returns:
           train_set_transform, test_set_transform(train and test dataset after dimension reduction)
    """

    fi = np.array(fi)
    f = np.array(f)
    distance = []
    if mode == 'L1':
        for i in range(len(offset)):
            roll = offset[i]
            f_roll = np.roll(f, roll)
            distance.append(sum(abs(f_roll - fi)))
            # distance.append(distance.cityblock(fi, f_roll))

    if mode == 'L2':
        for i in range(len(offset)):
            roll = offset[i]
            f_roll = np.roll(f, roll)
            distance.append(sum(np.power(f_roll - fi, 2)))
            # distance.append(distance.euclidean(fi, f_roll))

    if mode == 'cosine':
        for i in range(len(offset)):
            roll = offset[i]
            f_roll = np.roll(f, roll)
            result = 1 - sum(f_roll.T * fi) / (math.sqrt(sum(np.power(f_roll, 2))) * math.sqrt(sum(np.power(fi, 2))))
            distance.append(result)
            # distance.append(distance.cosine(fi, f_roll))

    return min(distance)


def nearest_center_classifier(train_set, train_label, test_set, dim=100, offset=[-9, -6, -3, 0, 3, 6, 9],
                              SAMPLE_SIZE=108, method='LDA'):
    """
        match test feature vector to those in train set with smallest distance

        Args:
            train_set: feature vectors of train set
            train_label: label for feature vecotr in train set
            test_set: feature vectors of test set
            dim: the expected dimension
            offset: angels rotation
            SAMPLE_SIZE: constant - number of clusters
            method: LDA/NULL/PCA

        Returns:
           minimal distance and result under different distance measurement
    """

    if dim < SAMPLE_SIZE and method == 'LDA':
        train_set, test_set = dim_reduction(train_set, train_label, test_set, dim)

    if dim < len(train_set[1]) and method == 'PCA':
        train_set, test_set = dim_reduction(train_set, train_label, test_set, dim, method='PCA')

    if method == 'NULL':
        True

    train_set = np.array(train_set)
    test_set = np.array(test_set)

    L1_result = []
    L2_result = []
    L3_result = []

    L1_min_distance = []
    L2_min_distance = []
    L3_min_distance = []

    # match every test vector to train vector
    for i in range(len(test_set)):
        L1_distance = []
        L2_distance = []
        L3_distance = []
        fi = test_set[i]
        # calculate distance under different angel rotations and choose the smallest
        for j in range(len(train_set)):
            f = train_set[j]
            L1_distance.append(minimal_offset_distance(fi, f, offset, 'L1'))
            L2_distance.append(minimal_offset_distance(fi, f, offset, 'L2'))
            L3_distance.append(minimal_offset_distance(fi, f, offset, 'cosine'))

        # for every test vector get the train vector that is the closest
        L1_min_distance.append(min(L1_distance))
        L2_min_distance.append(min(L2_distance))
        L3_min_distance.append(min(L3_distance))

        L1_result.append(train_label[L1_distance.index(min(L1_distance))])
        L2_result.append(train_label[L2_distance.index(min(L2_distance))])
        L3_result.append(train_label[L3_distance.index(min(L3_distance))])

    return L1_min_distance, L2_min_distance, L3_min_distance, L1_result, L2_result, L3_result
