import numpy as np
from dataGenerator import Generator

# in the file, we generate different data set to do experiments

## 1. Algorithm time complexity
#### (1) M workers
# for num_worker in [10, 100, 500, 1000, 1500]:
#     g = Generator(num_worker=num_worker, num_item=1000)
#     CM = g.confusionMatrix()
#     truth = g.generate_item_label()
#     label = g.generate_worker_label(CM, truth)
#     np.savetxt(f'synthetic_data/number/M={num_worker}_N=1000_truth.txt', truth, delimiter=' ', fmt='%d')
#     np.savetxt(f'synthetic_data/number/M={num_worker}_N=1000_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')
#### (2) N items
# for num_item in [50, 100, 500, 1500]:
#     g = Generator(num_worker=100, num_item=num_item)
#     CM = g.confusionMatrix()
#     truth = g.generate_item_label()
#     label = g.generate_worker_label(CM, truth)
#     np.savetxt(f'synthetic_data/number/M=100_N={num_item}_truth.txt', truth, delimiter=' ', fmt='%d')
#     np.savetxt(f'synthetic_data/number/M=100_N={num_item}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')
## 2. Sparsity of label matrix,
#### alter alpha and beta, to see the effect of different sparsity of label matrix.
#### from beta distribution plot, we can find proper alpha and beta values

#### (1) normal like, most workers label nearly 50% items, few workers label high or low percent
# for value in [2, 4, 6]:
#     g = Generator(num_worker=100, num_item=1000, alpha=value, beta=value)
#     CM = g.confusionMatrix()
#     truth = g.generate_item_label()
#     label = g.generate_worker_label(CM, truth)
#     np.savetxt(f'synthetic_data/sparsity/alpha={value}_beta={value}_truth.txt', truth, delimiter=' ', fmt='%d')
#     np.savetxt(f'synthetic_data/sparsity/alpha={value}_beta={value}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')
#### (2) right tail, most workers label low percent of items
# for alpha in [1, 2]:
#     for beta in [5, 7, 9]:
#         g = Generator(num_worker=100, num_item=1000, alpha=value, beta=value)
#         CM = g.confusionMatrix()
#         truth = g.generate_item_label()
#         label = g.generate_worker_label(CM, truth)
#         np.savetxt(f'synthetic_data/sparsity/alpha={alpha}_beta={beta}_truth.txt', truth, delimiter=' ', fmt='%d')
#         np.savetxt(f'synthetic_data/sparsity/alpha={alpha}_beta={beta}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')
#### (3) left tail, most workers label high percent of items
# for beta in [1, 2]:
#     for alpha in [5, 7, 9]:
#         g = Generator(num_worker=100, num_item=1000, alpha=value, beta=value)
#         CM = g.confusionMatrix()
#         truth = g.generate_item_label()
#         label = g.generate_worker_label(CM, truth)
#         np.savetxt(f'synthetic_data/sparsity/alpha={alpha}_beta={beta}_truth.txt', truth, delimiter=' ', fmt='%d')
#         np.savetxt(f'synthetic_data/sparsity/alpha={alpha}_beta={beta}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')
#### (4) low in the middle, most workers label either high or low percent of items
# for value in [0.2, 0.5, 0.8]:
#     g = Generator(num_worker=100, num_item=1000, alpha=value, beta=value)
#     CM = g.confusionMatrix()
#     truth = g.generate_item_label()
#     label = g.generate_worker_label(CM, truth)
#     np.savetxt(f'synthetic_data/sparsity/alpha={value}_beta={value}_truth.txt', truth, delimiter=' ', fmt='%d')
#     np.savetxt(f'synthetic_data/sparsity/alpha={value}_beta={value}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')

## 3. Worker labeling quarlity
#### normal worker and abnormal worker, how confusion matrix's change influent algorithm performance

#### (1) alter true_prob and false_prob
# for low_val in [0.3, 0.5, 0.7]:
#     for high_val in [0.3, 0.5, 0.7]:
#         g = Generator(num_worker=100, num_item=1000)
#         CM = g.confusionMatrix(true_prob=np.array([low_val, 0.9]), false_prob=np.array([0.0, high_val]))
#         truth = g.generate_item_label()
#         label = g.generate_worker_label(CM, truth)
#         np.savetxt(f'synthetic_data/quality/low={low_val}_high={high_val}_truth.txt', truth, delimiter=' ', fmt='%d')
#         np.savetxt(f'synthetic_data/quality/low={low_val}_high={high_val}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')
#### (2) alter proportion of abnormal workers
# for val1 in [0.02, 0.1, 0.25, 0.4]:
#     for val2 in [0.02, 0.1, 0.25, 0.4]:
#         g = Generator(num_worker=100, num_item=1000)
#         CM = g.confusionMatrix(all_same_prop=val1, all_false_prop=val2)
#         truth = g.generate_item_label()
#         label = g.generate_worker_label(CM, truth)
#         np.savetxt(f'synthetic_data/quality/same={val1}_false={val2}_truth.txt', truth, delimiter=' ', fmt='%d')
#         np.savetxt(f'synthetic_data/quality/same={val1}_false={val2}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')

## 4. true label distribution
#### to check whether the label ditribution will influent the algorithm performance

#### (1) unbalanced label
# for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     g = Generator(num_worker=100, num_item=1000)
#     CM = g.confusionMatrix()
#     truth = g.generate_item_label(prob=value)
#     label = g.generate_worker_label(CM, truth)
#     np.savetxt(f'synthetic_data/label/prob={value}_truth.txt', truth, delimiter=' ', fmt='%d')
#     np.savetxt(f'synthetic_data/label/prob={value}_crowd.txt', label, delimiter=' ', fmt='%d')
# print('ok')


## add experiment

for num_item in [500, 1000, 1500, 2000, 2500]:
    g = Generator(num_worker=10, num_item=num_item)
    CM = g.confusionMatrix(all_same_prop=0.25)
    truth = g.generate_item_label()
    label = g.generate_worker_label(CM, truth)
    np.savetxt(f'synthetic_data/them5/M=10_N={num_item}_truth.txt', truth, delimiter=' ', fmt='%d')
    np.savetxt(f'synthetic_data/them5/M=10_N={num_item}_crowd.txt', label, delimiter=' ', fmt='%d')
print('ok')


