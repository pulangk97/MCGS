# %%
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# %%
def get_img_feature(img, model, if_cont_rgb=False, layeridx = [2,6,40,84]): #[2,6,40,84]
    def get_featuremap(layer_output, image_shape, idx=[2,6,40,84]):
        featuremap = torch.tensor([])
        for i in idx:
            out = layer_output[i]
            for key, value in out.items():
                scale_factor = (image_shape[0]/value.shape[-2],image_shape[1]/value.shape[-1])
                output_tensor = F.interpolate(value, scale_factor=scale_factor, mode='bilinear', align_corners=True)

                featuremap = torch.concat((featuremap, output_tensor), dim=1)

        return featuremap

    def forward_hook(module, input, output):

        class_name = module.__class__.__name__
        data = { class_name: output

        }
        layer_outputs.append(data)

    layer_outputs = []

    for name, layer in model.named_modules():
        layer.register_forward_hook(forward_hook)

    model = model
    output = model(img)
    shape = img.shape[-2:]
    featuremap = get_featuremap(layer_output=layer_outputs, image_shape=shape, idx=layeridx)   
    # cat_feature = torch.concat((img[None,...],featuremap),dim=1) 
    norm_featuremap = featuremap/torch.norm(featuremap,dim=1)
    # cat_feature = torch.concat((img[None,...],norm_featuremap),dim=1)
    # print("feature shape"+str(cat_feature.shape))

    return norm_featuremap.squeeze()

    # return featuremap.squeeze()

DIR_0 = "/media/xyr/data22/datasets/datasets/nerf_llff_data/fern/images_8/image000.png"
DIR_1 = "/media/xyr/data22/datasets/datasets/nerf_llff_data/fern/images_8/image001.png"
img0 = torch.tensor(np.array(Image.open(DIR_0))).permute(2,0,1)[None,...]/255.
img1 = torch.tensor(np.array(Image.open(DIR_1))).permute(2,0,1)[None,...]/255.
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
model = resnet50
feature0 = get_img_feature(img=img0,model=model)
feature1 = get_img_feature(img=img1,model=model)

attention_map = torch.sum(feature0[:,100,100][:,None,None]*feature1,dim=0)/(torch.norm(feature0[:,100,100])*torch.norm(feature1,dim=0))
# print(attention_map.shape)
# attention_map = attention_map/torch.max(attention_map)
# plt.imsave(attention_map.detach().numpy(),"/media/xyr/data11/code/3DGS/gaussian-splatting-wdepth/gaussian-splatting-diff-depth/MCGS/output/attention.png")
plt.imshow(attention_map.detach().numpy())
# %%
