import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from IrisNormalization import normalization
from ImageEnhancement import enhancement
# from IrisLocalization import localization

from IrisMatching import nearest_center_classifier
from PerformanceEvaluation import crr, roc
from FeatureExtraction import featureVector
from tabulate import tabulate

F1 = 1 / 3
F2 = 1 / 4.5
BLOCK_SIZE = 9
THRESHOLD = 48

train_set = []
train_label = []

test_set = []
test_label = []

train_file_name = glob.glob('CASIA Iris Image Database (version 1.0)/*/1/*.bmp')

for item in train_file_name:
    train_label.append(int(item.split('/')[1]))

    image_train = cv2.imread(item)
    train_img = normalization(image_train)
    enhanced_train_img = enhancement(train_img, SIZE=32)
    feature_train = featureVector(img=enhanced_train_img, threshold_x=THRESHOLD, block=BLOCK_SIZE, f_1=F1, f_2=F2)
    # assert len(feature_train) == 1536
    train_set.append(feature_train)

print('Finished!')

test_file_name = glob.glob('CASIA Iris Image Database (version 1.0)/*/2/*.bmp')

for item in test_file_name:
    test_label.append(int(item.split('/')[1]))

    image_test = cv2.imread(item)
    test_img = normalization(image_test)
    enhanced_test_img = enhancement(test_img, SIZE=32)
    feature_test = featureVector(img=enhanced_test_img, threshold_x=THRESHOLD, block=BLOCK_SIZE, f_1=F1, f_2=F2)
    # assert len(feature_test) == 1536
    test_set.append(feature_test)

print('Finished!')

# create table 3
L1_min_distance, L2_min_distance, L3_min_distance, L1_result, L2_result, L3_result = nearest_center_classifier(
    train_set, train_label, test_set, dim=1536, method='NULL')
_, _, _, L1, L2, L3 = nearest_center_classifier(train_set, train_label, test_set, dim=107, method='LDA')

L1_original = crr(L1_result, test_label)
L2_original = crr(L2_result, test_label)
L3_origianl = crr(L3_result, test_label)

L1 = crr(L1, test_label)
L2 = crr(L2, test_label)
L3 = crr(L3, test_label)
print(tabulate([['L1 distance measure', L1_original, L1], ['L2 distance measure', L2_original, L2],
                ['Cosine similarity measure', L3_origianl, L3]],
               headers=['Similarity measure', 'Original feature set', 'Reduced feature set']))

L1_crr = []
L2_crr = []
L3_crr = []
feature_deduction = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for num in feature_deduction:
    L1_min_distance, L2_min_distance, L3_min_distance, L1_result, L2_result, L3_result = nearest_center_classifier(
        train_set, train_label, test_set, dim=num, method='LDA')
    L1_crr.append(crr(L1_result, test_label))
    L2_crr.append(crr(L2_result, test_label))
    L3_crr.append(crr(L3_result, test_label))

plt.plot(feature_deduction, L3_crr, marker="*")
plt.xlabel('Dimensionality of the feature vector')
plt.ylabel('CRR')
plt.title('CRR')
plt.savefig('figure_10.png')
plt.show()

threshold = np.arange(0.4, 0.5, 0.001)
fmrs, fnmrs = roc(L3_result, L3_min_distance, test_label, threshold)

# create table 4
print(tabulate([[threshold[20], fmrs[20], fnmrs[20]], [threshold[40], fmrs[40], fnmrs[40]],
                [threshold[50], fmrs[50], fnmrs[50]], [threshold[60], fmrs[60], fnmrs[60]]],
               headers=['Threshold', 'False match rate(%)', 'False non-match rate(%)']))

threshold = np.arange(0.1, 1.0, 0.01)
fmrs, fnmrs = roc(L3_result, L3_min_distance, test_label, threshold)
plt.plot(fmrs, fnmrs)
plt.xlabel('False Match Rate(%)')
plt.ylabel('False Non-match Rate(%)')
plt.title('ROC')
plt.savefig('fig_11.png')
plt.show()
