import torch
print("PyTorch GPU:", torch.cuda.is_available())

import tensorflow as tf
print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))
