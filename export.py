import os
import hashlib
import json
import bioimageio.core
import numpy as np
import torch
import torch.nn as nn

from core import util
from tifffile import imread, imwrite
from models.UniEMIR_network import Network
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.transforms import functional as tf
from bioimageio.core.build_spec import build_model, add_weights

class Network2D(Network):
    def chop_forward(self, image):
        # pad to 256x256
        _, _, h, w = image.size()
        mod_pad_h = (256 - h % 256) % 256
        mod_pad_w = (256 - w % 256) % 256
        image_pad = F.pad(image, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        image_process = []
        for i in range(0, image.shape[2], 256):
            for j in range(0, image.shape[3], 256):
                image_process.append(image_pad[:, :, i:i+256, j:j+256])
        image_process = torch.cat(image_process, dim=0)

        results = []
        for i in range(0, image_process.shape[0], 16):
            model_input = image_process[i:i+16]
            results.append(self.model(model_input))
        results = torch.cat(results, dim=0)

        output = torch.zeros(image_pad.shape if self.model.task != 3 else [image_pad.shape[0], results.shape[1], image_pad.shape[2], image_pad.shape[3]]).to(image.device)
        for i in range(0, image.shape[2], 256):
            for j in range(0, image.shape[3], 256):
                output[:, :, i:i+256, j:j+256] = results[(i//256*(image.shape[3]//256)+j//256)*image.shape[0]: (i//256*image.shape[3]//256+j//256+1)*image.shape[0]]

        output = output[:, :, :h, :w]
        return output

    def forward(self, x):
        x = x[0] / 255 # only support batch size 1
        x = x.unsqueeze(1)
        x = tf.normalize(x, [0.5], [0.5])
        if self.model.task == 1:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return (self.chop_forward(x).swapaxes(0, 1).clamp(-1,1) + 1) * 127.5
    
class Network3D(Network2D):
    def forward(self, x):
        x = x[0] / 255 # only support batch size 1
        x = torch.stack([x[:-1], x[1:]], dim=1)
        x = tf.normalize(x, [0.5], [0.5])
        y = self.chop_forward(x)
        y = torch.cat([x[:,0:1], y.flip(1)], dim=1)
        y = y.reshape(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3])
        y = torch.cat([y, x[-2:-1,1:2]], dim=0)
        return (y.swapaxes(0, 1).clamp(-1,1) + 1) * 127.5
        

# create a temporary directory to store intermediate files
os.makedirs("imagej-plugin", exist_ok=True)

# define the model
import sys
task_name = sys.argv[1]
assert task_name in ["super-resolution", "denoising", "isotropic_reconstruction"]
match task_name:
    case "super-resolution":
        config = 'config/UniEMIR-zoom.json'
        model_path = 'experiments/train_UniEMIR-zoom/checkpoint/300_Network.pth'
        model_name = "UniEMIRSuperResolution"
        task = 1
        input_ = imread('example/Super-resolution Example.tif')[None, None, ...]
        network = Network2D
    case "denoising":
        config = 'config/UniEMIR-denoise.json'
        model_path = 'experiments/train_UniEMIR-denoise/checkpoint/300_Network.pth'
        model_name = "UniEMIRDenoise"
        task = 2
        input_ = imread('example/Denoising Example.tif')[None, None, ...]
        network = Network2D
    case "isotropic_reconstruction":
        config = 'config/UniEMIR-isotropic.json'
        model_path = 'experiments/train_UniEMIR-isotropic/checkpoint/300_Network.pth'
        model_name = "UniEMIRIsotropic"
        task = 3
        input_ = imread('example/Isotropic Reconstruction Example.tif')[None, ...]
        network = Network3D

json_str = ''
with open(config, 'r') as f:
    for line in f:
        line = line.split('//')[0] + '\n'
        json_str += line
opt = json.loads(json_str, object_pairs_hook=OrderedDict)
util.set_seed(opt['seed'])

model = network(opt["model"]["which_networks"][0]["args"]["unimodel"])
model.load_state_dict(torch.load(model_path), strict=False)
model.model.task = task
model.eval()
model = torch.jit.script(model).cuda()
model.save("imagej-plugin/weights.pt")

# define the input
np.save("imagej-plugin/test-input.npy", input_)
imwrite("imagej-plugin/test-input.tif", input_[0].astype(np.uint8))
print(input_.shape, input_.max(), input_.min(), input_.dtype)
with torch.no_grad():
    output = model(torch.from_numpy(input_).cuda()).cpu().numpy()
np.save("imagej-plugin/test-output.npy", output)
output = np.load("imagej-plugin/test-output.npy")
print(output.max(), output.min())
print(input_.shape, output.shape)

# save output
imwrite("imagej-plugin/test-output.tif", output[0].astype(np.uint8))
input_ = np.load("imagej-plugin/test-input.npy")
model = torch.jit.load("imagej-plugin/weights.pt")
with torch.no_grad():
    output_2 = model(torch.from_numpy(input_).cuda()).cpu().numpy()
print(np.testing.assert_array_almost_equal(output,output_2,decimal=4))

# create documentation
with open("imagej-plugin/doc.md", "w") as f:
    f.write("# Pushing high-quality ultrastructural imaging limits with a foundational restoration model for volume electron microscopy")
    f.write(f'''This model is an electron microscopy {task_name} model. It is derivated from our foundational model, UniEMIR. \n The input must be an image with {"ZYX" if task == 3 else "YX"} axes and 8-bit uint format. Large images are cropped into 256 patches for processing. \n Please check the Github repository [UniEMIR](github.com/ryanhe312/UniEMIR) for details.\n''')

# create the model zip
build_model(
    # the weight file and the type of the weights
    weight_uri="imagej-plugin/weights.pt",
    weight_type="torchscript",
    covers=["example/cover.png"],
    # the test input and output data as well as the description of the tensors
    # these are passed as list because we support multiple inputs / outputs per model
    test_inputs=["imagej-plugin/test-input.npy"],
    test_outputs=["imagej-plugin/test-output.npy"],
    input_axes=["bcyx"],
    input_names=["input"],
    input_min_shape=[[1, 1, 128, 128]] if task == 1 else ([[1,1,256,256]] if task == 2 else [[1,2,256,256]]),
    input_step=[[0, 1, 0, 0]],
    output_axes=["bcyx"],
    output_reference=["input"],
    output_scale=[[1,1,2,2]] if task == 1 else ([[1,1,1,1]] if task == 2 else [[1,6,1,1]]),
    output_offset=[[0,0,0,0]] if task != 3 else [[1,-5,1,1]],
    input_data_range=[[0, 255]],
    # where to save the model zip, how to call the model and a short description of it
    output_path=f"imagej-plugin/{model_name}.zip",
    name=model_name,

    description="Pushing high-quality ultrastructural imaging limits with a foundational restoration model for volume electron microscopy",
    # additional metadata about authors, licenses, citation etc.
    authors=[{"name": "Ruian He", "affiliation": "Fudan University"},
             {"name": "Weimin Tan", "affiliation": "Fudan University"},
             {"name": "Chenxi Ma", "affiliation": "Fudan University"},
             {"name": "Bo Yan", "affiliation": "Fudan University"},],
    maintainers=[{"name": "Ruian He", "github_user": "ryanhe312"}],
    license="CC-BY-4.0",
    documentation="imagej-plugin/doc.md",
    tags=["electron-microscopy","2d","pytorch",'image-restoration'], 
    pytorch_version="1.13.1",
    cite=[{"text": "Ruian He, Weimin Tan, Chenxi Ma and Bo Yan. Pushing high-quality ultrastructural imaging limits with a foundational restoration model for volume electron microscopy. (2024).", "url": "github.com/ryanhe312/UniEMIR"}],
    add_deepimagej_config=True,
)

from bioimageio.core.resource_tests import test_model
my_model = bioimageio.core.load_resource_description(f"imagej-plugin/{model_name}.zip") 
test_model(my_model, devices=['cuda'])