# Requirements
|Package|
|----|
|numpy|
|matplotlib|
|torch|
|Pillow|
|torchvision|
|pytorch_grad_cam|

# Codes
You can import get_cam from get_cam, and the function as following:  

    get_cam(model, target_layers, rgb_img, device, input_tensor=None, targets=None, method='gradcam', aug_smooth=False, eigen_smooth=False, display=True, save_path=None)

There are several methods to use:  

    methods = \
        {"gradcam": GradCAM, 
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM, 
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "FullGrad": FullGrad}

The range of the rgb_image to input into the function is must between 0 and 1.  
If you do not want to display the result, please set the `display=False`.  
To save the result, `save_path=your_path`