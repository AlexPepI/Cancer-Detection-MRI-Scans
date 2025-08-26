import torch
import torchxrayvision as xrv
import torchvision.transforms as transforms
import skimage, torch, torchvision


path = "./images/example_xray.jpg"


# Prepare the image:
img = skimage.io.imread(path)
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img.mean(2)[None, ...] # Make single color channel



transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

img = transform(img)
img = torch.from_numpy(img)

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-rsna")
outputs = model(img[None,...]) # or model.features(img[None,...]) 

# Print results
x = dict(zip(model.pathologies,outputs[0].detach().numpy()))

# print(model.pathologies,outputs[0].detach().numpy())

print(x)
