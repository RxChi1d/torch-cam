from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_layer_name(model):
    con2d = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            con2d.append(name)
        print(con2d[-1])
        
def image_processing(rgb_img, device):
    # preprocess_image to normalize the image and transfer to tensor
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    return input_tensor.to(device)

def get_cam(model, target_layers, rgb_img, device, input_tensor=None, targets=None, method='gradcam', aug_smooth=False, eigen_smooth=False, display=True, save_path=None):
    methods = \
        {"gradcam": GradCAM, 
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM, 
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "FullGrad": FullGrad}
    
    use_cuda = True if torch.cuda.is_available() else False
    if input_tensor == None:
        input_tensor = image_processing(rgb_img,device)
        
    with methods[method](model=model, target_layers=target_layers, use_cuda=use_cuda) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)
        
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.figure("image")
    plt.axis("off");plt.title(method)
    if display == True:
        plt.imshow(visualization)
        plt.show()
    if save_path == True:
        plt.imsave(save_path,visualization)


if __name__ == '__main__':
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    target_layers = [model.layer4[-1]]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rgb_img = Image.open(r'cat.bmp').convert('RGB')
    rgb_img = np.float32(rgb_img) / 255

    get_cam(model, target_layers, rgb_img, save_path=r'gradcam.png',device=device)
    
    