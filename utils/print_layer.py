import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input
from collections import Counter
import math

# Load model with fixed batch size = 1
model = InceptionV3(weights='imagenet', include_top=True,
                    input_tensor=Input(batch_shape=(1, 299, 299, 3)))

# Print header for the table
# showing layer index, name, input/output shapes, kernel size, stride, padding,
# and GEMM transformation details if applicable
print(f"{'Idx':>3} {'Layer Name':15} {'Input Shape':22} {'Output Shape':22} {'Kernel':20} {'Stride':10} {'Padding':8} {'GEMM Input Tile':35} {'GEMM Kernel':35} {'GEMM Output':22} {'(M,N,K)'}")
print("=" * 220)
num =0
mnk_counter = Counter()

# iterate through each layer in the model
# and print the relevant information
# including input/output shapes, kernel size, stride, padding,
# and GEMM transformation details if applicable
for idx, layer in enumerate(model.layers):
    def get_shape(val):
        if isinstance(val, (list, tuple)):
            return [v.shape if hasattr(v, 'shape') else v for v in val]
        return val.shape if hasattr(val, 'shape') else val

    input_shape = get_shape(layer.input)
    output_shape = get_shape(layer.output)

    kernel = "-"
    stride = "-"
    padding = "-"
    gemm_input_tile = "-"
    gemm_kernel = "-"



    if isinstance(layer, tf.keras.layers.Conv2D):
        if layer.kernel is not None:
            kernel = layer.kernel.shape  # (kh, kw, Cin, Cout)
            stride = layer.strides
            padding = layer.padding.upper()

            kh, kw, Cin, Cout = layer.kernel.shape

            _, H, W, _ = input_shape

            # number of tiles estimation with given informationl (number of tiles = H / stride_h × W / stride_w)
            tile_h = math.ceil(H / stride[0]) if stride[0] != 0 else 0
            tile_w = math.ceil(W / stride[1]) if stride[1] != 0 else 0
            num_tiles = tile_h * tile_w if tile_h and tile_w else "?"

            # GEMM transformation
            # input: (num_tiles, kh*kw, Cin)
            # weights: (kh*kw, Cin, Cout)
            k_eff = kh * kw
            gemm_input_tile = f"({num_tiles}, {k_eff}, {Cin}) -> ({num_tiles}, {k_eff * Cin})"
            gemm_kernel = f"({k_eff}, {Cin}, {Cout}) -> ({k_eff * Cin}, {Cout})"
            gemm_output = f"({num_tiles}, {Cout})"
            m_n_k = f"({num_tiles}, {Cout}, {k_eff * Cin})"
            print(f"{idx:3d} {layer.name:15} {str(input_shape):22} {str(output_shape):22} {str(kernel):20} {str(stride):10} {padding:8} {gemm_input_tile:35} {gemm_kernel:35} {gemm_output:22} {m_n_k}")
            num+= 1
            if m_n_k in mnk_counter.keys():
                mnk_counter[m_n_k] += 1
            else:
                mnk_counter[m_n_k] = 1
        
print('total conv2d:', num)
print(mnk_counter)
print(mnk_counter.total())