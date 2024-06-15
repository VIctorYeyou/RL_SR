import sys
sys.dont_write_bytecode = True  # Prevent Python from generating .pyc bytecode files.

import argparse  # Used to parse command-line arguments.
from neuralnet import PixelRL_model 
from State import State
import torch
from utils.common import *
import matplotlib.pyplot as plt  # 用于展示图像
import os  # 用于文件操作
import numpy as np

torch.manual_seed(1)

# =====================================================================================
# arguments parser
# =====================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--scale",     type=int, default=4,  help='-')
parser.add_argument("--ckpt-path", type=str, default="", help='-')
parser.add_argument("--output-dir", type=str, default="output", help='Directory to save output images')
FLAG, unparsed = parser.parse_known_args()


# =====================================================================================
# Global variables
# =====================================================================================

SCALE = FLAG.scale
if SCALE not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3 or 4")

MODEL_PATH = FLAG.ckpt_path
if (MODEL_PATH == "") or (MODEL_PATH == "default"):
    MODEL_PATH = f"checkpoint/x{SCALE}/PixelRL_SR-x{SCALE}.pt"

OUTPUT_DIR = FLAG.output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU (if available) or CPU
N_ACTIONS = 7  # number of actions for the model
GAMMA = 0.95  # Discount factor, which is used to calculate the reward.
T_MAX = 5  # Maximum number of time steps.
SIGMA = 0.3 if SCALE == 2 else 0.2 
# The standard deviation used for the Gaussian blur, set according to the magnification scale factor.

LS_HR_PATHS = sorted_list(f"dataset/test/myx{SCALE}/labels")
LS_LR_PATHS = sorted_list(f"dataset/test/myx{SCALE}/data")
# to get the file path for high - and low-resolution images

# =====================================================================================
# Test each image
# =====================================================================================

def main():
    CURRENT_STATE = State(SCALE, DEVICE)  # Initialize the state object.

    MODEL = PixelRL_model(N_ACTIONS).to(DEVICE)  # Initialize and load the model.
    if exists(MODEL_PATH):
        MODEL.load_state_dict(torch.load(MODEL_PATH, torch.device(DEVICE)))
        # Load the model weights.
    MODEL.eval()  # Set the model to evaluation mode.
    # Test each image
    reward_array = []
    metric_array = []
    # The reward and PSNR values are stored separately.
    for i in range(0, len(LS_HR_PATHS)):
        # The paths of the high-resolution and low-resolution images are obtained separately.
        hr_image_path = LS_HR_PATHS[i]
        lr_image_path = LS_LR_PATHS[i]
        hr = read_image(hr_image_path)
        lr = read_image(lr_image_path)
        lr = gaussian_blur(lr, sigma=SIGMA)  # Gaussian blur is applied to the low-resolution image.
        bicubic = upscale(lr, SCALE)  # The low-resolution image is magnified by bicubic interpolation.

        bicubic = rgb2ycbcr(bicubic)
        lr = rgb2ycbcr(lr)
        hr = rgb2ycbcr(hr)  # Convert image color space to YCbCr.

        bicubic = norm01(bicubic).unsqueeze(0)  # Normalize the image and add a dimension.
        lr = norm01(lr).unsqueeze(0)
        hr = norm01(hr).unsqueeze(0)

        # In the case of not computing the gradient
        with torch.no_grad():
            CURRENT_STATE.reset(lr, bicubic)  # Resetting the current state
            sum_reward = 0
            for t in range(0, T_MAX):
                prev_img = CURRENT_STATE.sr_image.clone()  # Reset the current state to copy the current super-resolution image
                statevar = CURRENT_STATE.tensor.to(DEVICE)  # Transform the current state into a tensor and pass it to the model.
                actions, _, inner_state = MODEL.choose_best_actions(statevar)  # Select the best action and update the state.

                CURRENT_STATE.step(actions, inner_state)
                # Calculate reward on Y channel only
                reward = torch.square(hr[:,0:1] - prev_img[:,0:1]) - \
                         torch.square(hr[:,0:1] - CURRENT_STATE.sr_image[:,0:1])

                sum_reward += torch.mean(reward * 255) * (GAMMA ** t)

            sr = torch.clip(CURRENT_STATE.sr_image, 0.0, 1.0)  # The generated super-resolution image is cropped.
            sr_image = torch.clip(CURRENT_STATE.sr_image[0], 0.0, 1.0)
            sr_image = denorm01(sr_image)
            sr_image = sr_image.type(torch.uint8)
            sr_image = ycbcr2rgb(sr_image)

            
            psnr = PSNR(hr, sr)
            metric_array.append(psnr)
            reward_array.append(sum_reward)

            # 输出每张图片的 PSNR 和 reward
            print(f"Image {i+1}: PSNR: {psnr:.4f}, Reward: {sum_reward:.4f}")

            # 保存超分辨率后的图像
            #sr_image = sr.squeeze(0).permute(1,2,0).cpu().numpy() * 255
            #sr_image = sr_image.astype(np.uint8)
    
            output_path = os.path.join(OUTPUT_DIR, f"super_resolution_image_{i+1}.png")
            plt.imsave(output_path, sr_image)
            print(f"Saved super-resolution image to {output_path}")

    # 输出平均 PSNR 和 reward
    print(f"Average reward: {torch.mean(torch.tensor(reward_array) * 255):.4f}",
          f"- PSNR: {torch.mean(torch.tensor(metric_array)):.4f}")

if __name__ == '__main__':
    main()
