import tensorflow as tf
import sys

if not sys.warnoptions:
    import warnings
    warnings.filterwarnings('ignore')

GPU_is_available=tf.test.is_built_with_cuda()
GPU_list_devices=tf.config.list_physical_devices()
if GPU_is_available:
    # use the first device
    tf.config.set_visible_devices(GPU_list_devices[0], 'GPU')

print(45)