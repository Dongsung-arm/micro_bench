import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import plot_model

# load model
model = InceptionV3(weights='imagenet', include_top=True)

# save svg file
# to display svg file, please use your browser,jupyter or vcode
# the svg file failed to display within your terminal
plot_model(model,
           to_file='inceptionv3_model.svg',
           show_shapes=True,
           show_layer_names=True,
           dpi=70)