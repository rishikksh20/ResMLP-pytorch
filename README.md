# ResMLP : Feedforward networks for image classification with data-efficient training
Pytorch implementaion of [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404).

![](model.PNG)
## Usage:
```python
import torch
import numpy as np
from resmlp import ResMLP

img = torch.ones([1, 3, 224, 224])

model = ResMLP(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                 dim=384, depth=12, mlp_dim=384*4)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)

out_img = model(img)

print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
```

## Citation:
```
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training}, 
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and Edouard Grave and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
