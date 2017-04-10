from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

# # # # # # #之前跑的出的错Blas SGEMM launch failed : a.shape=(120, 4), b.shape=(4, 10), m=120, n=10, k=4
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))
  # # # # # #
# 定义数据集的路径
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 加载数据集
# # 加载训练集
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TRAINING,
    target_dtype = np.int,
    features_dtype = np.float32
    )

# # 加载测试集
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TEST,
    target_dtype = np.int,
    features_dtype = np.float32
    )

# Specifiy that all features have real-balue data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = 4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units = [10, 20, 10],
                                            n_classes = 3,
                                            model_dir = "iris_model")

# Fit model
classifier.fit(x = training_set.data, y = training_set.target, steps = 2000)
accuracy_score = classifier.evaluate(x = test_set.data, y = test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable = True))
print('Predictions: {}'.format(str(y)))
