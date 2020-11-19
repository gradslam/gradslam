from .rgbdimages import RGBDImages
from .pointclouds import Pointclouds

__all__ = ["pointclouds_from_rgbdimages"]


def pointclouds_from_rgbdimages(
    rgbdimages: RGBDImages,
    *,
    global_coordinates: bool = True,
    filter_missing_depths: bool = True,
) -> Pointclouds:
    r"""Converts gradslam.RGBDImages containing batch of RGB-D images with sequence length of 1 to gradslam.Pointclouds

    Args:
        rgbdimages (gradslam.RGBDImages): Can contain a batch of RGB-D images but must have sequence length of 1.
        global_coordinates (bool): If True, will create pointclouds object based on :math:`(X, Y, Z)` coordinates
            in the global coordinates (based on `rgbdimages.poses`). Otherwise, will use the local frame coordinates.
        filter_missing_depths (bool): If True, will not include vertices corresponding to missing depth values
            in the output pointclouds.

    Returns:
        gradslam.Pointclouds: Output pointclouds
    """
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.".format(
                type(rgbdimages)
            )
        )
    if not rgbdimages.shape[1] == 1:
        raise ValueError(
            "Expected rgbdimages to have sequence length of 1. Got {0}.".format(
                rgbdimages.shape[1]
            )
        )

    B = rgbdimages.shape[0]
    rgbdimages = rgbdimages.to_channels_last()
    vertex_map = (
        rgbdimages.global_vertex_map if global_coordinates else rgbdimages.vertex_map
    )
    normal_map = (
        rgbdimages.global_normal_map if global_coordinates else rgbdimages.normal_map
    )

    if filter_missing_depths:
        mask = rgbdimages.valid_depth_mask.squeeze(-1)  # remove missing depth values
        points = [vertex_map[b][mask[b]] for b in range(B)]
        normals = [normal_map[b][mask[b]] for b in range(B)]
        colors = [rgbdimages.rgb_image[b][mask[b]] for b in range(B)]
    else:
        points = vertex_map.reshape(B, -1, 3).contiguous()
        normals = normal_map.reshape(B, -1, 3).contiguous()
        colors = rgbdimages.rgb_image.reshape(B, -1, 3).contiguous()

    return Pointclouds(points=points, normals=normals, colors=colors)
