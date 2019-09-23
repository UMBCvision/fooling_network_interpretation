# Fooling Network Interpretation in Image Classification
This is the PyTorch implementation for our ICCV 2019 paper - Fooling Network Interpretation in Image Classification [https://www.csee.umbc.edu/~hpirsiav/papers/fooling_iccv19.pdf].

Deep neural networks have been shown to be fooled rather easily using adversarial attack algorithms. Practical methods such as adversarial patches have been shown to be extremely effective in causing misclassification. However, these patches are highlighted using standard network interpretation algorithms, thus revealing the identity of the adversary. We show that it is possible to create adversarial patches which not only fool the prediction, but also change what we interpret regarding the cause of the prediction. Moreover, we introduce our attack as a controlled setting to measure the accuracy of interpretation algorithms. We show this using extensive experiments for Grad-CAM interpretation that transfers to occluding patch interpretation as well. We believe our algorithms can facilitate developing more robust network interpretation tools that truly explain the network’s underlying decision making process.

This code is based on https://github.com/jacobgil/pytorch-grad-cam

Please install PyTorch (https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed.

The script `gradcam_targeted_patch_attack.py` takes as argument the input image and the corresponding result directory to store the results. The script performs a targeted patch attack using the regular adversarial patch method as well as our adversarial patch method. The adversarial patch created using our method is able to fool both the classifier as well as the corresponding network interpretation for the target category.

Usage: `python gradcam_targeted_patch_attack.py --image-path ./examples/ILSVRC2012_val_00008855.JPEG --result-dir ./results`


