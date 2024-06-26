# Import libraries for super resolution

import time
from typing import Tuple
import os
# from winreg import REG_RESOURCE_LIST
from PIL import Image
import numpy as np
import onnxruntime
from torch import nn
import torch.utils.model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
from resizeimage import resizeimage
import numpy as np

from module_options import ModuleOptions

# print(torch.__version__)

# Super Resolution Model Definition in Pytorch. The model comes directly from 
# PyTorch’s examples without modification:
class SuperResolutionNet(nn.Module):

    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


# Create the super-resolution model by using the above model definition.
assets_path = ""
use_CUDA    = False
torch_model = SuperResolutionNet(upscale_factor=3)

def init_superres(weights_path: str, use_CUDA_gpu: bool):

    global assets_path
    global use_CUDA

    assets_path = weights_path
    use_CUDA    = use_CUDA_gpu

# Example of Exporting a Model from PyTorch Pretrained Model Weights to ONNX
def export_pytorch_to_onnx():

    model_url = os.path.normpath(assets_path + '/superres_epoch100-44c6958e.pth')
    batch_size = 1    # just a random number

    # Initialize model with the pretrained weights
    if ModuleOptions.enable_GPU and torch.cuda.is_available():
        map_location = None     # default GPU
    # elif use_MPS:
    #    map_location = torch.device('mps')
    else:
        map_location = torch.device('cpu') 

    torch_model.load_state_dict(torch.load(model_url, map_location=map_location))

    # set the model to inference mode. This is required since operators like 
    # dropout or batchnorm behave differently in inference and training mode.
    torch_model.eval()
    
    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    torch_model.eval()

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


# Preprocessing Image
def pre_process_image(orig_img: Image):

    #orig_img = Image.open("FILE_PATH_TO_IMAGE")
    img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)

    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_norm = img_4.astype(np.float32) / 255.0

    return (img_norm, img_cb, img_cr)


# Run Model on Onnxruntime
def run_onnx(assets_path: str, processed_img: Image, use_CUDA: bool) -> any:

    # Start from ORT 1.10, ORT requires explicitly setting the providers 
    # parameter if you want to use execution providers other than the default 
    # CPU provider (as opposed to the previous behavior of providers getting 
    # set/registered by default based on the build flags) when instantiating 
    # InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built 
    # with CUDA, then call API as following:
    # onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
    model_path = os.path.normpath(assets_path + "/super-resolution-10.onnx")

    providers = onnxruntime.get_available_providers()
    # We need to check if CUDAExecutionProvider is in the list of providers
    # if use_CUDA and torch.cuda.is_available():
    #     ort_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    # else:
    #     ort_session = onnxruntime.InferenceSession(model_path)
    ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
        
    ort_inputs  = {ort_session.get_inputs()[0].name: processed_img} 
    ort_outputs = ort_session.run(None, ort_inputs)   
    output      = ort_outputs[0]

    return output


# Post processing Image
def post_process_image(img_out_y: any, img_cb: any, img_cr: any) -> Image:

    img_output: Image = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img: Image = Image.merge(
        "YCbCr", [
            img_output,
            img_cb.resize(img_output.size, Image.BICUBIC),
            img_cr.resize(img_output.size, Image.BICUBIC),
        ]).convert("RGB")

    return final_img


def superresolution(img: Image) -> Tuple[any, int]: # Tuple[Image, int]

    (img_norm, img_cb, img_cr) = pre_process_image(img)

    start_time = time.perf_counter()
    output = run_onnx(assets_path, img_norm, use_CUDA)
    inferenceMs : int = int((time.perf_counter() - start_time) * 1000)

    result = post_process_image(output, img_cb, img_cr)

    return result, inferenceMs