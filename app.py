import torch
import json
import copy
import torchvision

import numpy as np
import gradio as gr

from PIL import Image
from core import util
from tqdm import tqdm
from collections import OrderedDict
from tifffile import imread, imwrite
from torch.nn import functional as F
from torchvision.transforms import functional as tf
from models.UniEMIR_network import Network
from classifier.classifier import add_gaussian_noise, add_gaussian_blur, fft_tensor, Classifier

DEVICES = ['CPU','CUDA','Paralleled CUDA']
TASKS = ['Super-Resolution','Denoising','Isotropic Reconstruction']
MODEL = None
ARGS = None
BATCH_SIZE = 16

class Args:
    chop = False
    task = None
    device = 'cpu'

def run_model(img_input, adaptive):
    global MODEL, ARGS

    if MODEL is None:
        gr.Error("Model not loaded!")
        return [None, None]

    if img_input is None:
        gr.Error("Image not loaded!")
        return [None, None]  
    
    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif') and not img_input.name.endswith('.tiff'):
        gr.Error("Image must be a tiff file!")
        return None
    
    image = imread(img_input.name) 
    image = image.astype(np.float32) / np.iinfo(image.dtype).max
    print(image.shape, image.max(), image.min())
    image = torch.tensor(image)
    print(image.shape, image.max(), image.min())

    if ARGS.task != 3:
        if len(image.shape) == 2:
            image = image[None, None, ...]
        elif len(image.shape) == 3:
            image = image[:, None, ...]
        else:
            gr.Error("Image must be 2 or 3 dimensional!")
            return [None, None] 
        
    else:
        if len(image.shape) != 3:
            gr.Error("Isotropic reconstruction only accepts 3D images!")
            return [None, None] 
        
        # split the image to 2 slices
        image = torch.stack([image[:-1], image[1:]], dim=1)

    # upscale
    # if ARGS.task == 1:
    #     image = F.interpolate(image, scale_factor=2, mode='bilinear', align_corners=False)

    image = tf.normalize(image, 0.5, 0.5)
    print(image.shape, image.max(), image.min())
        
    # chop nxcxhxw into 256x256 patches
    if ARGS.chop:
        # pad to 256x256
        _, _, h, w = image.size()
        mod_pad_h = (256 - h % 256) % 256
        mod_pad_w = (256 - w % 256) % 256
        image_pad = F.pad(image, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        image_process = torch.cat([image_pad[:, :, i:i+256, j:j+256] for i in range(0, image.shape[2], 256) for j in range(0, image.shape[3], 256)], dim=0)
    else:
        image_process = image

    print(f'Image shape: {image_process.shape}')

    # adapt
    MODEL.model.task = ARGS.task
    if adaptive and ARGS.task != 3:
        # setup the student model
        tea_model = MODEL.model
        stu_model = copy.deepcopy(tea_model)
        stu_model.train()
        optimizer = torch.optim.Adam(stu_model.parameters(), lr=5e-5)

        # generate crap image
        if ARGS.task == 1:
            classifier = Classifier(task='blur', device=ARGS.device)
            cond_image = np.clip((image[0,0].cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
            level = classifier(fft_tensor(cond_image))[0]
            print('Blur level:', level)
            crap_image = add_gaussian_blur(cond_image, level)
        elif ARGS.task == 2:
            classifier = Classifier(task='noise', device=ARGS.device)
            cond_image = np.clip((image[0,0].cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
            level = classifier(fft_tensor(cond_image))[0]
            print('Noise level:', level)
            crap_image = add_gaussian_noise(cond_image, level)
        else:
            raise ValueError('task not supported')
        
        print('reimaging:', crap_image.shape, crap_image.max(), crap_image.min())
        print('raw image:', cond_image.shape, cond_image.max(), cond_image.min())
        cond_image = torch.tensor(cond_image,device=ARGS.device).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1
        crap_image = torch.tensor(crap_image,device=ARGS.device).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1

        # optimize the student model
        for i in range(adaptive):
            optimizer.zero_grad()

            # random crop to 64 x 64
            lq, input_crap = [], []
            for j in range(32):
                # get random crop
                y = np.random.randint(0, cond_image.shape[-2]-64)
                x = np.random.randint(0, cond_image.shape[-1]-64)
                lq.append(cond_image[..., y:y+64, x:x+64])
                input_crap.append(crap_image[..., y:y+64, x:x+64])
            lq = torch.concat(lq, dim=0)
            input_crap = torch.concat(input_crap, dim=0)

            # forward pass
            output_stu = stu_model(lq)
            output_tea = tea_model(lq).detach()
            output_crap = stu_model(input_crap)

            # compute loss
            loss = torch.sqrt(torch.square(output_crap-output_tea)+1e-3).mean() + torch.sqrt(torch.square(output_stu-output_crap)+1e-3).mean()
            print('loss:',loss.item())

            loss.backward()
            optimizer.step()
        
        # eval student model
        stu_model.eval()

    if adaptive and ARGS.task == 3:
        # setup the student model
        tea_model = MODEL.model
        stu_model = copy.deepcopy(tea_model)
        stu_model.train()
        optimizer = torch.optim.Adam(stu_model.parameters(), lr=5e-5)

        cond_image = torch.cat([image[:,0], image[-1:,1]], dim=0).transpose(0,1).to(ARGS.device)
        print('reimaging:', cond_image.shape)
        
        if cond_image.shape[1] <= 64:
            gr.Warning("Image axial resolution too small for imaging-aware adaptation!")
            adaptive = 0

        # optimize the student model
        for i in range(adaptive):
            optimizer.zero_grad()

            # random crop to 64 x 64
            lq, hq = [], []
            for j in range(32):
                # get random crop
                y = np.random.randint(0, cond_image.shape[-2]-64)
                x = np.random.randint(0, cond_image.shape[-1]-64)
                z = np.random.randint(0, cond_image.shape[0]-6)
                lq.append(torch.stack([cond_image[z, y:y+64, x:x+64],cond_image[z+6, y:y+64, x:x+64]]))
                hq.append(cond_image[z+1:z+6, y:y+64, x:x+64].flip(0))
            lq = torch.stack(lq, dim=0)
            hq = torch.stack(hq, dim=0)
            print('lq:',lq.shape,'hq:',hq.shape)

            # forward pass
            output_stu = stu_model(lq)

            # compute loss
            loss = torch.sqrt(torch.square(output_stu-hq)+1e-3).mean()
            print('loss:',loss.item())

            loss.backward()
            optimizer.step()
        
        # eval student model
        stu_model.eval()

    # run the model
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, image_process.shape[0], BATCH_SIZE)):
            model_input = image_process[i:i+BATCH_SIZE].to(ARGS.device)
            if adaptive:
                results.append(stu_model(model_input).cpu().detach())
            else:
                results.append(MODEL.model(model_input).cpu().detach())
    results = torch.cat(results, dim=0)

    # merge the patches
    if ARGS.chop:
        output = torch.zeros(image_pad.shape if ARGS.task != 3 else (image_pad.shape[0], results.shape[1], image_pad.shape[2], image_pad.shape[3]))
        for i in range(0, image.shape[2], 256):
            for j in range(0, image.shape[3], 256):
                output[:, :, i:i+256, j:j+256] = results[(i//256*(image.shape[3]//256)+j//256)*image.shape[0]: (i//256*image.shape[3]//256+j//256+1)*image.shape[0]]

        output = output[:, :, :h, :w]
    else:
        output = results

    # image unstack axis 1, nx5xhxw -> 5nxhxw
    if ARGS.task == 3:
        output = torch.cat([image[:,0:1], output.flip(1)], dim=1)
        output = output.reshape(output.shape[0]*output.shape[1], *(output.size()[2:]))
        output = torch.cat([output,image[-1,1:2]], dim=0)

    # save the output
    print(f'Ouput shape: {output.shape}')
    imwrite('output.tif', ((output.squeeze().clamp_(-1, 1).numpy() + 1) * 127.5).round().astype(np.uint8))

    return ['output.tif', "Output Successfully Saved!"]

def visualize(img_input, progress=gr.Progress()):
    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif') and not img_input.name.endswith('.tiff'):
        gr.Error("Image must be a tiff file!")
        return None
    
    image = imread(img_input.name)
    shape = image.shape
    print(f'Image shape: {shape}')

    if len(shape) == 2:
        return [[image], f'2D image loaded with shape {shape}']
    elif len(shape) == 3:
        clips = []
        for i in range(min(shape[0],10)):
            clips.append(image[i])
        return [clips, f'3D image loaded with shape {shape}, only showing first 10 slices.']
    else:
        gr.Error("Image must be 2 or 3 dimensional!")
        return None
    
def load_model(type, device, chop, progress=gr.Progress()):
    global MODEL, ARGS

    ARGS = Args()
    ARGS.chop = chop == 'Yes'

    match type:
        case 'Super-Resolution':
            config = 'config/UniEMIR-zoom.json'
            model_path = 'experiments/train_UniEMIR-zoom/checkpoint/300_Network.pth'
            ARGS.task = 1
        case 'Denoising':
            config = 'config/UniEMIR-denoise.json'
            model_path = 'experiments/train_UniEMIR-denoise/checkpoint/300_Network.pth'
            ARGS.task = 2
        case 'Isotropic Reconstruction':
            config = 'config/UniEMIR-isotropic.json'
            model_path = 'experiments/train_UniEMIR-isotropic/checkpoint/300_Network.pth'
            ARGS.task = 3

    json_str = ''
    with open(config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    util.set_seed(opt['seed'])

    MODEL = Network(opt["model"]["which_networks"][0]["args"]["unimodel"])
    MODEL.load_state_dict(torch.load(model_path), strict=False)
    MODEL.eval()

    match device:
        case 'CPU':
            pass
        case 'CUDA':
            MODEL.cuda()
            ARGS.device = 'cuda'
        case 'Paralleled CUDA':
            MODEL.cuda()
            MODEL.model = torch.nn.DataParallel(MODEL.model)
            ARGS.device = 'cuda'

    return '%s model loaded on %s, %s chop' % (type, device, "w/ " if ARGS.chop else "w/o")


with gr.Blocks(title="UniEMIR Web Demo") as demo:

    gr.Markdown("# UniEMIR WebUI")
    gr.Markdown("This web UI allows you to run the models on your own images or the examples from the paper.")

    gr.Markdown("## Instructions")
    gr.Markdown("1. Select the model and options. We provide models for different tasks including super-resolution, denoising and isotropic reconstruction. The model supports CPU, GPU, and multiple GPUs. You can choose to chop the image into smaller patches to save GPU memory and enable parallel processing.")
    gr.Markdown("2. Click 'Load Model' to load the model.")
    gr.Markdown("3. Upload your tiff image or use the examples below. The model accepts 2 (xy) and 3 (zxy) dimensional images in uint8 or uint16 data type. Isotropic reconstruction only accepts 3D images and interpolates on z-axis.")
    gr.Markdown("4. (Optional) Click 'Check Input' to inspect your input image.")
    gr.Markdown("5. Click 'Restore Image' to run the model on the input image. Processing large 3D images will take several minutes to run. The output image will be saved as 'output.tif' for download.")
    gr.Markdown("6. (Optional) Click 'Check Output' to inspect the output image.")

    gr.Markdown("## Load Model")
    with gr.Row():
        type = gr.Dropdown(label="Task", choices=TASKS, value="Denoising", interactive=True)
        chop = gr.Dropdown(label="Chop", choices=['Yes','No'], value="Yes", interactive=True)
        device = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA", interactive=True)
        adaptive = gr.Slider(label="Adaptive", minimum=0, maximum=10, step=1, value=0, interactive=True)
        load_progress = gr.Textbox(label="Model Information", value="Model not loaded")
        load_btn = gr.Button("Load Model")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Image or Use Examples")
            img_input = gr.File(label="Input File", interactive=True)
            gr.Examples(
                label='Examples',
                examples=[
                    ["example/Denoising Example.tif"],
                    ["example/Super-resolution Example.tif"],
                    ["example/Isotropic Reconstruction Example.tif"],
                ],
                inputs=[img_input],
            )
            img_visual = gr.Gallery(label="Input Viusalization", interactive=False)

            with gr.Row():
                input_message = gr.Textbox(label="Image Information", value="Image not loaded")
                check_input = gr.Button("Check Input") 
                run_btn = gr.Button("Restore Image")

        with gr.Column():
            gr.Markdown("## Preview and Download Results")
            output_file = gr.File(label="Output File", interactive=False)
            img_output = gr.Gallery(label="Output Visualiztion", interactive=False)

            with gr.Row():
                output_message = gr.Textbox(label="Output Information", value="Image not loaded")
                display_btn = gr.Button("Check Output")
                
    check_input.click(visualize, inputs=img_input, outputs=[img_visual, input_message], queue=True)
    display_btn.click(visualize, inputs=output_file, outputs=[img_output, output_message], queue=True)
    load_btn.click(load_model,inputs=[type, device, chop],outputs=load_progress, queue=True)
    run_btn.click(run_model, inputs=[img_input, adaptive], outputs=[output_file, output_message], queue=True)

demo.queue().launch(server_name='0.0.0.0', server_port=7860)