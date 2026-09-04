import torchio as tio
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy

from torchio import Subject
from torchio.constants import TYPE, INTENSITY
from torchio.transforms.augmentation.spatial.random_affine import RandomAffine, Affine, get_borders_mean
from torchio.data.io import nib_to_sitk
from numbers import Number

import warnings


def get_pixdim_from_affine(affine: np.array):
    rot = affine[:-1,:-1]
    return np.sqrt(np.sum(rot**2, axis=0))


class MyRescaleIntensity(tio.RescaleIntensity):
    
    def rescale(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        image_name: str,
    ) -> torch.Tensor:
        # The tensor is cloned as in-place operations will be used
        array = tensor.clone().float().numpy()
        mask_array = mask.numpy()
        if not mask_array.any():
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because the mask to compute the statistics is empty'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor

        values = array[mask_array]
        cutoff = np.percentile(values, self.percentiles)
        ###### Added by me ######
        '''
        Sometimes if we zoomed out a lot, the 98th percentile is background. 
        Use this is brain mask unavailable.
        '''
        if cutoff[0] == cutoff[1]:
            percentiles = list(copy.deepcopy(self.percentiles))
            while True:
                percentiles[0] = max(percentiles[0] - 0.25, 0.0)
                percentiles[1] = min(percentiles[1] + 0.25, 100.0)
                cutoff = np.percentile(values, percentiles)
                if (cutoff[0] != cutoff[1]) or (percentiles[0] == 0 and percentiles[1] == 100):
                    break
        ########################      
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]

        if self.in_min_max is None:
            in_min, in_max = array.min(), array.max()
        else:
            in_min, in_max = self.in_min_max
        in_range = in_max - in_min
        if in_range == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because all the intensity values are the same'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor

        out_range = self.out_max - self.out_min

        array -= in_min
        array /= in_range
        array *= out_range
        array += self.out_min
        return torch.as_tensor(array)
    

class MyRandomAffine(RandomAffine):
    
    def __init__(self, scales_are_spacings = False, update_affine: bool = False, no_scale_dims = None, no_translation_dims = None, no_rotation_dims = None, **kwargs):
        super().__init__(**kwargs)
        self.update_affine = update_affine
        self.scales_are_spacings = scales_are_spacings
        self.no_scale_dims = no_scale_dims
        self.no_translation_dims = no_translation_dims
        self.no_rotation_dims = no_rotation_dims
    
    def apply_transform(self, subject: Subject) -> Subject:
        scaling_params, rotation_params, translation_params = self.get_params(
            self.scales,
            self.degrees,
            self.translation,
            self.isotropic,
        )
        
        if self.scales_are_spacings:
            scaling_params = (torch.tensor(subject.spacing) / scaling_params).to(dtype=scaling_params.dtype)
        
        if self.no_scale_dims is not None:
            for d in self.no_scale_dims:
                scaling_params[d] = 1.0

        if self.no_translation_dims is not None:
            for d in self.no_translation_dims:
                translation_params[d] = 0.0
                
        if self.no_rotation_dims is not None:
            for d in self.no_rotation_dims:
                rotation_params[d] = 0.0
            
        arguments = {
            'scales': scaling_params,
            'degrees': rotation_params,
            'translation': translation_params,
            'center': self.center,
            'default_pad_value': self.default_pad_value,
            'image_interpolation': self.image_interpolation,
            'label_interpolation': self.label_interpolation,
            'check_shape': self.check_shape,
        }
        transform = MyAffine(**self.add_base_args(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed
    
class MyAffine(Affine):
    
    def __init__(self, update_affine: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.update_affine = update_affine
    
    def apply_transform(self, subject: Subject) -> Subject:
        if self.check_shape:
            subject.check_consistent_spatial_shape()
        default_value: float
        for image in self.get_images(subject):
            transform = self.get_affine_transform(image)
            transformed_tensors = []
            for tensor in image.data:
                sitk_image = nib_to_sitk(
                    tensor[np.newaxis],
                    image.affine,
                    force_3d=True,
                )
                if image[TYPE] != INTENSITY:
                    interpolation = self.label_interpolation
                    default_value = 0
                else:
                    interpolation = self.image_interpolation
                    if self.default_pad_value == 'minimum':
                        default_value = tensor.min().item()
                    elif self.default_pad_value == 'mean':
                        default_value = get_borders_mean(
                            sitk_image,
                            filter_otsu=False,
                        )
                    elif self.default_pad_value == 'otsu':
                        default_value = get_borders_mean(
                            sitk_image,
                            filter_otsu=True,
                        )
                    else:
                        assert isinstance(self.default_pad_value, Number)
                        default_value = float(self.default_pad_value)
                transformed_tensor = self.apply_affine_transform(
                    sitk_image,
                    transform,
                    interpolation,
                    default_value,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
            if self.update_affine:
                new_affine = self.get_new_affine_matrix(image)
                image.affine = new_affine
        return subject
    
    def get_new_affine_matrix(self, image: tio.Image) -> np.ndarray:
        # get the scaling, rotation and translation parameters
        scaling = np.asarray(self.scales).copy()
        rotation = np.asarray(self.degrees).copy()
        translation = np.asarray(self.translation).copy()
        # get the original affine matrix
        original_affine = image.affine
        # get matrix to offset voxel indices so that the voxel origin ([0,0,0] in voxel space) 
        # is at the location about which the transformation is applied
        if self.center == "image":
            voxel_origin = np.array(image.spatial_shape) / 2
        elif self.center == "origin":
            voxel_origin = (np.linalg.inv(original_affine) @ np.array([[0, 0, 0, 1]]).T).flatten()[:3]
        offset_voxels = np.eye(4)
        offset_voxels[:3,-1] = -voxel_origin
        reset_voxels = np.linalg.inv(offset_voxels)
        # forward transform of voxels
        rot = self.get_rotation_matrix(rotation)
        trans = translation / get_pixdim_from_affine(original_affine) # convert translation to voxel units
        rot_trans_mat = np.hstack([rot, trans.reshape(-1,1)])
        rot_trans_mat = np.vstack([rot_trans_mat, [0, 0, 0, 1]])
        rot_trans_mat = reset_voxels @ rot_trans_mat @ offset_voxels
        scale = np.eye(4)
        scale[:3,:3] = np.diag(scaling)
        scale_mat = reset_voxels @ scale @ offset_voxels
        forward_voxel_transform = scale_mat @ rot_trans_mat
        # inverse transform of voxels
        backward_voxel_transform = np.linalg.inv(forward_voxel_transform)
        new_affine = original_affine @ backward_voxel_transform
        return new_affine   
    
    @staticmethod
    def get_rotation_matrix(deg):
        x_deg, y_deg, z_deg = deg
        x_mat = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(x_deg)), -np.sin(np.deg2rad(x_deg))], [0, np.sin(np.deg2rad(x_deg)), np.cos(np.deg2rad(x_deg))]])
        y_mat = np.array([[np.cos(np.deg2rad(y_deg)), 0, np.sin(np.deg2rad(y_deg))], [0, 1, 0], [-np.sin(np.deg2rad(y_deg)), 0, np.cos(np.deg2rad(y_deg))]])
        z_mat = np.array([[np.cos(np.deg2rad(z_deg)), -np.sin(np.deg2rad(z_deg)), 0], [np.sin(np.deg2rad(z_deg)), np.cos(np.deg2rad(z_deg)), 0], [0, 0, 1]])
        return x_mat @ y_mat @ z_mat 


class ResampleWithinRange(tio.Transform):
    
    def __init__(self, target, resample_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        if len(target) != 6:
            target = 3 * target
            assert len(target) == 6
        target = [(mini, maxi) for mini, maxi in zip(target[::2], target[1::2])]
        self.target = target
        self.resample_kwargs = resample_kwargs
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        target = copy.deepcopy(list(subject.spacing))
        for i, (rng, s) in enumerate(zip(self.target, target)):
            if s >= rng[0] and s <= rng[1]:
                continue
            if abs(s - rng[0]) < abs(s - rng[1]):
                target[i] = rng[0]
            else:
                target[i] = rng[1]
        T = tio.Resample(target, **self.resample_kwargs)
        return T(subject)
    

class MyEnsureShapeMultiple(tio.EnsureShapeMultiple):
    
    def apply_transform(self, subject: Subject) -> Subject:
        excluded = {}
        for image_name in (self.exclude if self.exclude is not None else []):
            excluded[image_name] = subject.pop(image_name)
        source_shape = np.array(subject.spatial_shape, np.uint16)
        function: Callable = np.floor if self.method == 'crop' else np.ceil  # type: ignore[assignment]
        integer_ratio = function(source_shape / self.target_multiple)
        target_shape = integer_ratio * self.target_multiple
        target_shape = np.maximum(target_shape, 1)
        transform = tio.CropOrPad(target_shape.astype(int), **self.get_base_args())
        subject = transform(subject)  # type: ignore[assignment]
        for image_name, image in excluded.items():
            subject.add_image(image, image_name)
        return subject


class MyToCanonical(tio.ToCanonical):
    
    def apply_transform(self, subject):

        # Exlude images from ToCanonical if specified
        excluded = {}
        for image_name in (self.exclude if self.exclude is not None else []):
            excluded[image_name] = subject.pop(image_name)

        # Apply ToCanonical to the remaining images
        subject = super().apply_transform(subject)

        # Add the rest of the images to a new subject and apply ToCanonical
        for image_name, image in excluded.items():
            subject.add_image(image, image_name)
        return subject


class CopyAnyAffine(tio.Transform):

    def __init__(self):
        super().__init__()

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        keys = list(subject.get_images_names())
        reference = subject[keys[0]]
        affine = copy.deepcopy(reference.affine)
        for key in keys[1:]:
            image = subject[key]
            image.load()
            if not np.allclose(affine, image.affine, atol=1e-5):
                raise RuntimeError(
                    f"Not all affines for the subject are close. Found: \n\n{reference.path}:\n{affine}\n{image.path}:\n{image.affine}\n"
                    )
            image.affine = affine
        return subject


def get_min_max(subject):
        min_max = {}
        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
                min_max[key] = (min, max)
        return min_max
    

class Brightness(tio.transforms.Transform):

    def __init__(self, rng=(0.7, 1.3), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        x = random.uniform(*self.rng)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                value.set_data(value.data * x)

        return subject


class Contrast(tio.transforms.Transform):

    def __init__(self, rng=(0.65, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        x = random.uniform(*self.rng)

        min_max = get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                scaled_data = value.data * x
                clamped_data = torch.clamp(scaled_data, min_max[key][0], min_max[key][1])
                value.set_data(clamped_data)

        return subject
    
    
class Gamma(tio.transforms.Transform):

    def __init__(self, rng=(0.7, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        gamma = random.uniform(*self.rng)
        min_max = get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min, max = min_max[key][0], min_max[key][1]
                normalised_data = (value.data - min) / (max - min)
                if random.random() < 0.15:
                    augmented_data = 1 - (1 - normalised_data) ** gamma
                else:
                    augmented_data = normalised_data ** gamma
                rescaled_data = augmented_data * (max - min) + min
                value.set_data(rescaled_data)

        return subject