# Non-Uniform_Illumination
we introduce an attack technique, which is the Non-Uniform Illumination (NUI) of images to fool the CNN models. In this attack, the images are perturbed by adding a NUI mask of different intensities.

The (NUI) attack mask is created using several non-linear equations generating non-uniform variations of brightness and darkness exploiting the spatial structure of the image. We demonstrate how the proposed (NUI) attack degrades the performance of VGG, ResNet, MobilenetV3-small and InceptionV3 models on various renowned datasets, including CIFAR10, Caltech256, and TinyImageNet. We evaluate the proposed attack’s effectiveness on various classes of images.

## Overview
![Screenshot from 2023-05-28 19-58-45](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/88e2d850-58ed-4991-9cb8-fd345143db02)

Explanation of the Non-Uniform Illumination
(NUI) attack in order to fool the CNN models on the task
of image classification. The 1st row shows the training over
the original training data, the 2nd row shows the testing over
the test data and the 3rd row shows the testing over the
transformed test set.

## Method
![method_diag](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/6a57b512-ff1a-4610-b689-c35155d572d0)
Here is the workflow of the proposed method and the experimental settings used for the training and testing to fool
the CNN models using the NUI attack technique.

# Result
![2](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/310ca582-851d-43fe-b242-2f3c7c7ed26a)

We have applied 12 different masks in order to perform the NUI attack. In the above figure, the effect of the NUI attack on different images is shown. Here the 1st row is the original images and the later rows
contain the transformed images after the NUI attack. The 2nd row contains images perturbed using mask function mask(1)
given above and similarly all the other rows contains images perturbed using the mask function from mask(2) to mask(12)
respectively

Below are the Accuracy, precision, recall and F1-score for the mobilenetV4-small model

<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">

  <div style="flex: 1; min-width: 200px; padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/5f9a9098-c81a-412c-a72c-22c2aba41625" alt="method_diag" width="200"/>
  </div>

  <div style="flex: 1; min-width: 200px; padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/834f466a-ffe0-4b7f-91f8-de57fea9abf6" alt="method_diag" width="200"/>
  </div>

  <div style="flex: 1; min-width: 200px; padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/dd4f9700-f8c4-4771-84c5-8eff78acbbfe" alt="method_diag" width="200"/>
  </div>

  <div style="flex: 1; min-width: 200px; padding: 10px;">
    <img src="https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/bd2ae9b9-b1e1-4636-a96f-10e5899da84c" alt="method_diag" width="200"/>
  </div>

</div>


![1](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/5f9a9098-c81a-412c-a72c-22c2aba41625)
![precision_1](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/834f466a-ffe0-4b7f-91f8-de57fea9abf6)
![recall_1](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/dd4f9700-f8c4-4771-84c5-8eff78acbbfe)
![F1_1](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/bd2ae9b9-b1e1-4636-a96f-10e5899da84c)



To describe the changes in the data distribution due to NUI attack, we have shown TSNE graph
![mob_unperturbed](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/13e5a5eb-998d-4289-8c82-edff67f6780b)
The above graph is the TSNE for the original data and the below graph is the TSNE after applying the NUI attack using one of the mask
![mob10_pos](https://github.com/Akshayjain97/Non-Uniform_Illumination/assets/131511513/8670105e-5808-4726-b670-854a51c2a238)

