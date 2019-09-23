import os
import argparse
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import models


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                # Store the features and register hook to save gradients for the target layer
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate targeted layers.
    3. Gradients from intermediate targeted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        # Retrieve the saved gradients for the target layer
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(input_image):
    """
    This method normalizes the input image and converts it to a torch Variable
    :param input_image: The input image to be pre-processed
    :return: torch Variable of the normalized input image
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    normalized_image = input_image.copy()[:, :, ::-1]
    for i in range(3):
        normalized_image[:, :, i] = normalized_image[:, :, i] - means[i]
        normalized_image[:, :, i] = normalized_image[:, :, i] / stds[i]
    normalized_image = \
        np.ascontiguousarray(np.transpose(normalized_image, (2, 0, 1)))
    normalized_image = torch.from_numpy(normalized_image)
    normalized_image.unsqueeze_(0)
    normalized_tensor = Variable(normalized_image, requires_grad=True).cuda()
    return normalized_tensor


def show_cam_on_image(input_image, mask, filename="cam.png"):
    """
    Converts the mask to a heatmap and overlays it with the input image.
    :param input_image: input image
    :param mask: gradcam mask to be used as heatmap
    :param filename: output path to write the image overlayed with the mask
    :return:
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(input_image)
    cam = cam / np.max(cam)
    cv2.imwrite(filename, np.uint8(255 * cam))


class GradCamAttack:
    """
    This class is responsible for creating a targeted adversarial patch such that the top predicted category is the
    target category and the Grad-CAM for the target category is hidden in the patch location.
    """
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model = model.cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)
        self.relu = torch.nn.ReLU()

    def __call__(self, image_tensor, index, target_class_index, lr=.005, eps=0.007, lambda_val=0.05, attack_iters=750):
        print('\n\nOur adversarial patch attack:\n\n')
        print('Before attack, Predicted class:{}\tTarget class:{}\n\n'.format(index, target_class_index))

        # Clone the original image for computing the adversarial image with the patch
        adv_image_tensor = image_tensor.clone()

        # Initialize the perturbation tensor
        dl_dx_cumulative = torch.zeros_like(image_tensor)

        # Specify the top-left co-ordinates and the size for the patch and create the corresponding mask
        # The mask will have ones at the patch location pixels and zeros at all other pixels
        start_pos = (0, 0)
        patch_size = 64
        mask = torch.zeros_like(image_tensor).cuda()
        mask.data[0, :, start_pos[1]:start_pos[1] + patch_size, start_pos[0]:start_pos[0] + patch_size] = 1.0

        # Means and std_devs used for pre-processing
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])

        # Compute the per channel clamp_min and clamp_max respectively
        channel_clamp_min = (0 - means) / stds
        channel_clamp_max = (1 - means) / stds

        loss_zero_counter = 0
        for i in range(attack_iters):
            adv_features, adv_output = self.extractor(adv_image_tensor)
            pred_index = np.argmax(adv_output.cpu().data.numpy())

            # Create a one-hot tensor for the target category
            one_hot = np.zeros((1, adv_output.size()[-1]), dtype=np.float32)
            one_hot[0][target_class_index] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot_tensor = torch.sum(one_hot.cuda() * adv_output)

            # Clear the gradients before loss computation
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()

            # Compute the gradients of the loss with respect to the feature layer for the adversarial image
            dy_dz, = torch.autograd.grad(one_hot_tensor, adv_features[0],
                                         grad_outputs=torch.ones(one_hot_tensor.size()).cuda(),
                                         retain_graph=True, create_graph=True)
            dy_dz_sum = dy_dz.sum(dim=2).sum(dim=2)

            # Compute gradient weighted class activations for the perturbed image
            grad_weighted_feats = dy_dz_sum.unsqueeze(-1).unsqueeze(-1) * adv_features[0]
            gcam = grad_weighted_feats.sum(dim=1).squeeze(0)
            gcam = self.relu(gcam)

            # Normalize the gradcam tensor
            gcam = gcam / (gcam.sum() + 1e-10)

            # Compute the loss for the patch location pixels in the gradcam tensor.
            # For a 224x224 image, the adversarial patch size is 64x64.
            # Since the gradcam tensor is 14x14 for VGG19 BN network, the corresponding gradcam patch size is 4x4
            gcam_loss = torch.sum(gcam[0:4, 0:4]).abs().cuda()
            gcam_loss = gcam_loss / 16.0

            # Add the cross entropy loss if target category is not the top predicted category
            if np.argmax(adv_output.cpu().data.numpy()) == target_class_index:
                xe_loss = 0.0
            else:
                xe_loss = self.criterion(adv_output, torch.tensor([target_class_index], dtype=torch.long).cuda())

            # We minimize both the gradcam loss and cross entropy loss
            total_loss = gcam_loss + (lambda_val * xe_loss)

            # Stop the attack once the loss is zero for 5 consecutive iterations
            if total_loss == 0.0:
                if loss_zero_counter > 5:
                    break
                else:
                    loss_zero_counter += 1
            else:
                loss_zero_counter = 0

            # Compute the gradient of the total loss with respect to the perturbed image
            dl_dx, = torch.autograd.grad(total_loss, adv_image_tensor)

            # Perform gradient ascent using the sign of dl_dx to compute the cumulative perturbation
            dl_dx_cumulative = dl_dx_cumulative - lr * torch.sign(dl_dx)
            adv_image_tensor = (1 - mask) * image_tensor.clone() + mask * dl_dx_cumulative

            # Clamp the adversarial image using per channel min and max respectively
            for c in range(3):
                adv_image_tensor[:, c, :, :] = adv_image_tensor[:, c, :, :].clamp(channel_clamp_min[c],
                                                                                  channel_clamp_max[c])

            if i % 10 == 0:
                print('Iteration:{}\tGradCAM Loss:{:.3f}\tCE Loss:{:.3f}\ttotal_pert.mean:{:.3f}\tOrig index:{}'
                      '\tTarget index:{}\tPred index:{}'.format(i, gcam_loss, xe_loss, dl_dx_cumulative.abs().mean(),
                                                                index, target_index, pred_index))

        # Store the resulting adversarial image tensor
        res_adv_tensor = image_tensor.clone()
        res_adv_tensor.data = adv_image_tensor.data

        # Get the top predicted category of the resulting adversarial image tensor
        _, adv_output = self.extractor(res_adv_tensor)

        print('\n\nAfter attack, Original class: {}\tPredicted class: {}\tTarget class: {}'.
              format(index, adv_output[0].argmax(), target_class_index))

        # Denormalize the adversarial image
        adv_img = res_adv_tensor.data[0].cpu().numpy()
        adv_img = np.transpose(adv_img, (1, 2, 0))
        for i in range(3):
            adv_img[:, :, i] = (adv_img[:, :, i] * stds[i]) + means[i]

        return adv_img, res_adv_tensor


class GradCamRegPatchAttack:
    """
    This class is responsible for creating a regular adversarial patch for a targeted attack.
    """
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model = model.cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def __call__(self, image_tensor, index, target_class_index, lr=.005, eps=0.007, lambda_val=0.05, attack_iters=750):
        print('\n\nRegular adversarial patch attack:\n\n')
        print('Before attack, Predicted class:{}\tTarget class:{}\n'.format(index, target_class_index))

        # Clone the original image for computing the perturbed adversarial image
        adv_image_tensor = image_tensor.clone() + torch.randn(image_tensor.size()).cuda() / 100

        # Initialize the perturbation tensor
        dl_dx_cumulative = torch.zeros_like(image_tensor)

        # Means and std_devs used for pre-processing
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])

        # Compute the per channel clamp_min and clamp_max respectively
        channel_clamp_min = (0 - means) / stds
        channel_clamp_max = (1 - means) / stds

        # Specify the top-left co-ordinates and the size for the patch and create the corresponding mask
        # The mask will have ones at the patch location pixels and zeros at all other pixels
        start_pos = (0, 0)
        patch_size = 64
        mask = torch.zeros_like(image_tensor).cuda()
        mask.data[0, :, start_pos[1]:start_pos[1] + patch_size, start_pos[0]:start_pos[0] + patch_size] = 1.0

        loss_zero_counter = 0
        target_flip_counter = 0
        for i in range(attack_iters):
            _, adv_output = self.extractor(adv_image_tensor)
            pred_index = np.argmax(adv_output.cpu().data.numpy())

            self.model.features.zero_grad()
            self.model.classifier.zero_grad()

            # Stop the attack once the target category is reached for 5 consecutive attack iterations
            if i > 250 and pred_index == target_class_index:
                if target_flip_counter > 5:
                    break
                else:
                    target_flip_counter += 1
            else:
                target_flip_counter = 0

            xe_loss = self.criterion(adv_output, torch.tensor([target_class_index], dtype=torch.long).cuda())

            # Stop the attack once the loss is zero for 5 consecutive attack iterations
            if xe_loss == 0.0:
                if loss_zero_counter > 5:
                    break
                else:
                    loss_zero_counter += 1
            else:
                loss_zero_counter = 0

            # Compute the gradient of the total loss with respect to the perturbed image
            dl_dx, = torch.autograd.grad(xe_loss, adv_image_tensor)

            # Perform gradient ascent using the sign of dl_dx to compute the cumulative perturbation
            dl_dx_cumulative = dl_dx_cumulative - lr * torch.sign(dl_dx)
            adv_image_tensor = image_tensor.clone() * (1 - mask) + dl_dx_cumulative * mask

            # Clamp the adversarial image using per channel min and max respectively
            for c in range(3):
                adv_image_tensor[:, c, :, :] = adv_image_tensor[:, c, :, :].clamp(channel_clamp_min[c],
                                                                                  channel_clamp_max[c])

            if i % 10 == 0:
                print('Iteration:{}\tCE Loss:{:.3f}\ttotal_pert.mean:{:.3f}\tOrig index:{}\tTarget index:{}'
                      '\tPred index:{}'.format(i, xe_loss, dl_dx_cumulative.abs().mean(),
                                               index, target_index, pred_index))

        # Store the resulting adversarial image tensor
        res_adv_tensor = image_tensor.clone()
        res_adv_tensor.data = adv_image_tensor.data

        # Get the top predicted category of the resulting adversarial image tensor
        _, adv_output = self.extractor(res_adv_tensor)

        print('\n\nAfter attack, Original class: {}\tPredicted class: {}\tTarget class: {}'.
              format(index, adv_output[0].argmax(), target_class_index))

        # Denormalize the adversarial image
        adv_img = res_adv_tensor.data[0].cpu().numpy()
        adv_img = np.transpose(adv_img, (1, 2, 0))
        for i in range(3):
            adv_img[:, :, i] = (adv_img[:, :, i] * stds[i]) + means[i]

        return adv_img, res_adv_tensor


class GradCam:
    """
    This class computes the Grad-CAM mask for the specified index.
    """
    def __init__(self, model, target_layer_names):
        self.model = model
        self.model.eval()
        self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def __call__(self, image_tensor, index=None):
        features, output = self.extractor(image_tensor)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        # Compute the one-hot tensor corresponding to the index
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # Get the gradients and features to compute Grad-CAM
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def forward_inference(model, input_tensor):
    """
    Computes forward inference on the input image tensor and
    returns the top prediction index and probability
    :param model:
    :param input_tensor:
    :return:
    """
    output = model(input_tensor)
    index = np.argmax(output.cpu().data.numpy())
    index_prob = torch.nn.functional.softmax(output)[0][index]
    return index, index_prob


def get_args():

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        exit()
    print("Using GPU for acceleration")

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./examples/ILSVRC2012_val_00017472.JPEG',
                        help='Input image path')
    parser.add_argument('--result-dir', type=str, default='./results', help='Path to store the results')
    return parser.parse_args()


if __name__ == '__main__':
    """ python gradcam_targeted_patch_attack.py --image-path <path_to_image> --result-dir <path_to_result_dir>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Setting the seed for reproducibility for demo
    # Comment the below 4 lines for the target category to be random across runs
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Can work with any model, but it assumes that the model has a feature method,
    # and a classifier method, as in the VGG models in torchvision.
    gradcam_attack = GradCamAttack(model=models.vgg19_bn(pretrained=True), target_layer_names=["51"])
    gradcam_reg_patch_attack = GradCamRegPatchAttack(model=models.vgg19_bn(pretrained=True), target_layer_names=["51"])
    gradcam = GradCam(model=models.vgg19_bn(pretrained=True), target_layer_names=["51"])
    pretrained_vgg_net = models.vgg19_bn(pretrained=True).cuda()
    pretrained_vgg_net = pretrained_vgg_net.eval()
    image_name = args.image_path.split('/')[-1].split('.')[0]

    # Create result directory if it doesn't exist
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Read the input image and preprocess to a tensor
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_img = preprocess_image(img)

    # Get the original prediction index and the corresponding probability
    orig_index, orig_prob = forward_inference(pretrained_vgg_net, preprocessed_img)

    # Pick a random target from the remaining 999 categories excluding the original prediction
    list_of_idx = np.delete(np.arange(1000), orig_index)
    rand_idx = np.random.randint(999)
    target_index = list_of_idx[rand_idx]

    # Compute the regular adv patch attack image and the corresponding GradCAM
    reg_patch_adv_img, reg_patch_adv_tensor = gradcam_reg_patch_attack(preprocessed_img, orig_index, target_index)
    reg_patch_pred_index, reg_patch_pred_prob = forward_inference(pretrained_vgg_net,
                                                                  preprocess_image(reg_patch_adv_img[:, :, ::-1]))
    cv2.imwrite(os.path.join(args.result_dir, image_name + '_reg_adv_patch_image.png'),
                np.uint8(255 * np.clip(reg_patch_adv_img[:, :, ::-1], 0, 1)))

    # Generate the GradCAM heatmap for the target category using the regular patch adversarial image
    reg_patch_adv_mask = gradcam(reg_patch_adv_tensor, target_index)
    show_cam_on_image(np.clip(reg_patch_adv_img[:, :, ::-1], 0, 1), reg_patch_adv_mask,
                      filename=os.path.join(args.result_dir, image_name + '_reg_adv_patch_gcam.JPEG'))

    # Compute the adv patch attack using our method and the corresponding GradCAM
    our_patch_adv_img, our_patch_adv_tensor = gradcam_attack(preprocessed_img, orig_index, target_index)
    our_patch_pred_index, our_patch_pred_prob = forward_inference(pretrained_vgg_net,
                                                                  preprocess_image(our_patch_adv_img[:, :, ::-1]))
    cv2.imwrite(os.path.join(args.result_dir, image_name + '_our_adv_patch_image.png'),
                np.uint8(255 * np.clip(our_patch_adv_img[:, :, ::-1], 0, 1)))

    # Generate the GradCAM heatmap for the target category using our patch adversarial image
    mask_adv_ = gradcam(our_patch_adv_tensor, target_index)
    show_cam_on_image(np.clip(our_patch_adv_img[:, :, ::-1], 0, 1), mask_adv_,
                      filename=os.path.join(args.result_dir, image_name + '_our_adv_patch_gcam.JPEG'))
