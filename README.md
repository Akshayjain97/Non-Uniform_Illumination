# Non-Uniform_Illumination
we introduce a novel attack technique, Non-Uniform Illumination (NUI), where images are subtly altered using varying NUI masks.

The (NUI) attack mask is created using several non-linear equations generating non-uniform variations of brightness and darkness exploiting the spatial structure of the image. We demonstrate how the proposed (NUI) attack degrades the performance of VGG, ResNet, MobilenetV3-small and InceptionV3 models on various renowned datasets, including CIFAR10, Caltech256, and TinyImageNet. We evaluate the proposed attack’s effectiveness on various classes of images.

## Overview

<div style="display: flex; justify-content: space-around; flex-wrap: nowrap;">

  <div style="padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/88e2d850-58ed-4991-9cb8-fd345143db02" alt="method_diag" width="600"/>
  </div>
</div>


Explanation of the Non-Uniform Illumination
(NUI) attack to fool the CNN models on the task
of image classification. The 1st row shows the training over
the original training data, the 2nd row shows the testing over
the test data and the 3rd row shows the testing over the
transformed test set.

## Method
<div style="display: flex; justify-content: space-around; flex-wrap: nowrap;">

  <div style="padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/6a57b512-ff1a-4610-b689-c35155d572d0" alt="method_diag" width="600"/>
  </div>
</div>

Here is the workflow of the proposed method and the experimental settings used for the training and testing to fool
the CNN models using the NUI attack technique.

# Result

<div style="display: flex; justify-content: space-around; flex-wrap: nowrap;">

  <div style="padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/310ca582-851d-43fe-b242-2f3c7c7ed26a" alt="method_diag" width="500"/>
  </div>
</div>

We have applied 12 different masks to perform the NUI attack. In the above figure, the effect of the NUI attack on different images is shown. Here the 1st row is the original images and the later rows
contain the transformed images after the NUI attack. The 2nd row contains images perturbed using mask function mask(1)
given above and similarly all the other rows contain images perturbed using the mask function from mask(2) to mask(12)
respectively

Below are the Accuracy, precision, recall and F1-score for the mobilenetV4-small model



<table>
  <tr>
    <td><img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/5f9a9098-c81a-412c-a72c-22c2aba41625" alt="method_diag" width="200"/></td>
    <td><img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/834f466a-ffe0-4b7f-91f8-de57fea9abf6" alt="method_diag" width="200"/></td>
    <td><img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/dd4f9700-f8c4-4771-84c5-8eff78acbbfe" alt="method_diag" width="200"/></td>
    <td><img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/bd2ae9b9-b1e1-4636-a96f-10e5899da84c" alt="method_diag" width="200"/></td>

  </tr>
</table>

To describe the changes in the data distribution due to the NUI attack, we have shown the TSNE graph.

The below graphs are the TSNE for the original data and data after applying the NUI attack using one of the mask, calculated using MobilenetV3-small

<table>
  <tr>
    <td><img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/13e5a5eb-998d-4289-8c82-edff67f6780b" alt="method_diag" width="300"/></td>
    <td><img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/8670105e-5808-4726-b670-854a51c2a238" alt="method_diag" width="300"/></td>

  </tr>
</table>

# Links and Citation
Find our work at https://ieeexplore.ieee.org/document/10916770 or access the **arxiv_paper** pdf in the repository.

**Cite our work using:**
A. Jain, S. R. Dubey, S. K. Singh, K. Santosh and B. B. Chaudhuri, "Non-Uniform Illumination Attack for Fooling Convolutional Neural Networks," IEEE Transactions on Artificial Intelligence, March 2025.

