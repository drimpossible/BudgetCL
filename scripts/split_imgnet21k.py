# python split_imgnet21k.py <full path to imagenet21k extracted folder> <full path to imagenet1k folder> <num_classes> <imgs_per_cls_threshold>
# Reproduce ours by: python select_subset_imagenet21k.py PATH_TO_IMAGENET21K PATH_TO_IMAGENET1K PATH_TO_IMAGENET2K 1000 750 ../clim2k/

import copy, os, random, shutil, sys
from os.path import isdir, isfile, join
import numpy as np

random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

imnet21k_dir = sys.argv[1]
save_dir = sys.argv[6]
classes = [f for f in os.listdir(root_dir) if isdir(join(root_dir, f))]
cnt = 0
class_sizes, class_order, cls_list = [], [], []
order1, order2 = [], []
pretrainf, prevalf, pretestf, valf, testf = [], [], [], [], []

imnet1k_train_dir_1k = sys.argv[2] + '/train/'
imnet1k_val_dir_1k = sys.argv[2] + '/test/'
imnet1k_test_dir_1k = sys.argv[2] + '/val/'
imnet2k_dir = sys.argv[3]

classes_1k = [
    f for f in os.listdir(root_dir_1k) if isdir(join(root_dir_1k, f))
]

for cls in classes_1k:
    folder = join(root_train_dir_1k, cls)
    imgs = [
        imnet2k_dir + '/' + cls + '/' + f for f in os.listdir(folder)
        if (isfile(join(folder, f)))
    ]
    pretrainf.extend(imgs)

for cls in classes_1k:
    folder = join(root_val_dir_1k, cls)
    imgs = [
        imnet2k_dir + '/' + cls + '/' + f for f in os.listdir(folder)
        if (isfile(join(folder, f)))
    ]
    prevalf.extend(imgs)

for cls in classes_1k:
    folder = join(root_test_dir_1k, cls)
    imgs = [
        imnet2k_dir + '/' + cls + '/' + f for f in os.listdir(folder)
        if (isfile(join(folder, f)))
    ]
    pretestf.extend(imgs)

problematic = [
    'n13867492', 'n15102894', 'n09450163', 'n10994097', 'n11196627',
    'n11318824'
]
for cls in problematic:
    assert (cls not in classes_1k), 'Problematic classes in Imagenet1k'
classes_1k.extend(problematic)

for cls in classes:
    folder = join(imnet21k_dir, cls)
    imgs = [
        imnet2k_dir + '/' + cls + '/' + f for f in os.listdir(folder)
        if (isfile(join(folder, f)))
    ]
    if len(imgs) < int(sys.argv[5]) or cls in classes_1k:
        pass
    else:
        cnt += 1
        if cnt > int(sys.argv[4]): break
        cls_list.extend([cls])
        random.shuffle(imgs)
        testf.extend(imgs[:50])
        valf.extend(imgs[50:60])
        order1.extend(imgs[60:])
        class_sizes.append(len(imgs) - 60)
        class_order.append(cls)

order2 = copy.deepcopy(order1)
random.shuffle(order2)

print(len(pretrainf), len(pretestf), len(order1), len(order2))

# Create order files
f = open(save_dir + '/pretrain.txt', 'w')
for line in pretrainf:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/preval.txt', 'w')
for line in prevalf:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/pretest.txt', 'w')
for line in pretestf:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/class_incremental_ordering.txt', 'w')
for line in order1:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/data_incremental_ordering.txt', 'w')
for line in order2:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/val.txt', 'w')
for line in valf:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/test.txt', 'w')
for line in testf:
    f.write(line + '\n')
f.close()

f = open(save_dir + '/class_order.txt', 'w')
for line in class_order:
    f.write(line + '\n')
f.close()

class_sizes = np.array(class_sizes)
np.save(save_dir + '/class_sizes.npy', class_sizes)