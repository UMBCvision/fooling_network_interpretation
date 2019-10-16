# Fooling Network Interpretation in Image Classification
This is the PyTorch implementation for our ICCV 2019 paper - [Fooling Network Interpretation in Image Classification][1]
Akshayvarun Subramanya*, Vipin Pillai*, Hamed Pirsiavash.

Deep neural networks have been shown to be fooled rather easily using adversarial attack algorithms. Practical methods such as adversarial patches have been shown to be extremely effective in causing misclassification. However, these patches are highlighted using standard network interpretation algorithms, thus revealing the identity of the adversary. We show that it is possible to create adversarial patches which not only fool the prediction, but also change what we interpret regarding the cause of the prediction. Moreover, we introduce our attack as a controlled setting to measure the accuracy of interpretation algorithms. We show this using extensive experiments for Grad-CAM interpretation that transfers to occluding patch interpretation as well. We believe our algorithms can facilitate developing more robust network interpretation tools that truly explain the network’s underlying decision making process.


![alt text][teaser]


### Bibtex
```
@article{fool_net_interp2019,
  title={Fooling Network Interpretation in Image Classification.},
  author={Akshayvarun Subramanya and Vipin Pillai and Hamed Pirsiavash},
  journal={International Conference on Computer Vision},
  year={2019}
}
```

### Pre-requisites

This code is based on https://github.com/jacobgil/pytorch-grad-cam

Please install PyTorch (https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed.

The script `gradcam_targeted_patch_attack.py` takes as argument the input image and the corresponding result directory to store the results. The script performs a targeted patch attack using the regular adversarial patch method as well as our adversarial patch method. The adversarial patch created using our method is able to fool both the classifier as well as the corresponding network interpretation for the target category.


### Usage
`python gradcam_targeted_patch_attack.py --image-path ./examples/ILSVRC2012_val_00008855.JPEG --result-dir ./results`

The mapping file for imagenet class labels to indices (0-999) can be found here - `misc/imagenet1000_clsidx_to_labels.txt` 

### Results

<table>
    <tr>
        <td align="center">
          Original Image
          <br/>
            <img src="https://github.com/UMBCvision/fooling_network_interpretation/blob/master/examples/ILSVRC2012_val_00008855.JPEG" width="160" height="160"/>
          <br/>
          Paddle
        </td>
        <td align="center">
          Adv. Patch
          <br/>
          <img src="https://github.com/UMBCvision/fooling_network_interpretation/blob/master/results/ILSVRC2012_val_00008855_reg_adv_patch_image.png" width="160" height="160"/>
          <br/>
          Box Turtle
        </td>
      <td align="center">
        Adv. Patch - GCAM
          <br/>
          <img src="https://github.com/UMBCvision/fooling_network_interpretation/blob/master/results/ILSVRC2012_val_00008855_reg_adv_patch_gcam.JPEG" width="160" height="160"/>
        <br/>
          Box Turtle
        </td>
      <td align="center">
        Our Patch
          <br/>
          <img src="https://github.com/UMBCvision/fooling_network_interpretation/blob/master/results/ILSVRC2012_val_00008855_our_adv_patch_image.png" width="160" height="160"/>
        <br/>
          Box Turtle
        </td>
      <td align="center">
        Our Patch - GCAM
          <br/>
          <img src="https://github.com/UMBCvision/fooling_network_interpretation/blob/master/results/ILSVRC2012_val_00008855_our_adv_patch_gcam.JPEG" width="160" height="160"/>
        <br/>
          Box Turtle
        </td>
    </tr>
</table>

### License
MIT

### Acknowledgement
This work was performed under the following financial assistance award: 60NANB18D279 from U.S. Department of Commerce, National Institute of Standards and Technology, funding from SAP SE, and also NSF grant 1845216.

[1]: https://arxiv.org/pdf/1812.02843.pdf
[teaser]: https://github.com/UMBCvision/fooling_network_interpretation/blob/master/misc/teaser.jpg
