import open3d as o3d
import pytest
import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import gradslam as gs
from gradslam.slam import fusionutils
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.structures.utils import pointclouds_from_rgbdimages

from tests.common import default_to_cpu_if_no_gpu, load_test_data


def rgbdimages_to_pointclouds(rgbdimages, sigma):
    pointclouds_global = pointclouds_from_rgbdimages(rgbdimages)
    pointclouds_local = pointclouds_from_rgbdimages(
        rgbdimages, global_coordinates=False
    )
    features = fusionutils.get_alpha(pointclouds_local.points_padded, sigma)
    pointclouds_global.features_padded = (
        features * pointclouds_global.nonpad_mask.to(features.dtype)
    ).unsqueeze(-1)
    return pointclouds_global


class TestGetAlpha:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_get_alpha(self, device):
        device = default_to_cpu_if_no_gpu(device)
        # Points to compute alpha for
        pts = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )

        sigma = 0.6
        eps = 1e-20
        alpha = fusionutils.get_alpha(pts, sigma, eps=eps)
        groundtruth = torch.tensor(
            [eps, 5.17e-17, 3.5924e-09, 3.5924e-09, 6.2177e-02, 1.0], device=device
        )

        assert alpha.shape == groundtruth.shape
        assert_allclose(alpha, groundtruth)
        assert alpha.gt(0).all().item()

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_gradcheck(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )

        sigma = torch.tensor(0.6, device=device)

        # evaluate function gradient
        pts = pts.requires_grad_().type(torch.float64)
        sigma = sigma.requires_grad_().type(torch.float64)
        assert gradcheck(fusionutils.get_alpha, (pts, sigma), raise_exception=True)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts = [1, 2, 3]
        sigma = 0.6
        with pytest.raises(TypeError):
            fusionutils.get_alpha(pts, sigma)

        pts = torch.tensor([1, 2, 3], device=device)
        sigma = (0.6, 0.7)
        with pytest.raises(TypeError):
            fusionutils.get_alpha(pts, sigma)

        pts = torch.tensor([1, 2, 3], device=device)
        sigma = 0.6
        fusionutils.get_alpha(pts, sigma)

        pts = torch.tensor([1, 2, 3], device=device)
        sigma = 0.6
        eps = 5
        with pytest.raises(TypeError):
            fusionutils.get_alpha(pts, sigma, eps=eps)

        pts = torch.tensor([1, 2, 3], device=device)
        sigma = 0.6
        eps = torch.tensor(1.5, device=device)
        with pytest.raises(TypeError):
            fusionutils.get_alpha(pts, sigma, eps=eps)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts = torch.tensor([1, 2, 3, 4], device=device)
        sigma = 0.6
        with pytest.raises(ValueError):
            fusionutils.get_alpha(pts, sigma)

        pts = torch.tensor([1, 2, 3], device=device)
        sigma = torch.tensor([0.6, 0.7], device=device)
        with pytest.raises(ValueError):
            fusionutils.get_alpha(pts, sigma)


class TestArePointsClose:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_are_points_close(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts1 = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )

        pts2 = torch.tensor(
            [
                [1.0, 3.0, 5.0],
                [3.0, 2.0, 2.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 1.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )
        dist_th = 2.0 ** 0.5
        is_close = fusionutils.are_points_close(pts1, pts2, dist_th)
        groundtruth = torch.tensor([0, 0, 1, 0, 0, 1], device=device, dtype=bool)
        assert_allclose(is_close.int(), groundtruth.int())

        dist_th = 2.01 ** 0.5
        is_close = fusionutils.are_points_close(pts1, pts2, dist_th)
        groundtruth = torch.tensor([0, 1, 1, 0, 0, 1], device=device, dtype=bool)
        assert_allclose(is_close.int(), groundtruth.int())

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts1 = ([3.0, 2.0, 1.0], [-1.0, 0.0, 1.0])
        pts2 = torch.tensor([[1.0, 2.0, 1.0], [1.0, 0.0, -1.0]], device=device)
        dist_th = 2.0 ** 0.5
        with pytest.raises(
            TypeError, match="Expected input tensor1 to be of type torch.Tensor"
        ):
            fusionutils.are_points_close(pts1, pts2, dist_th)
        with pytest.raises(
            TypeError, match="Expected input tensor2 to be of type torch.Tensor"
        ):
            fusionutils.are_points_close(pts2, pts1, dist_th)

        pts1 = torch.tensor([[3.0, 2.0, 1.0], [-1.0, 0.0, 1.0]], device=device)
        pts2 = torch.tensor([[1.0, 2.0, 1.0], [1.0, 0.0, -1.0]], device=device)
        dist_th = 2.0 ** 0.5
        fusionutils.are_points_close(pts1, pts2, dist_th)

        pts1 = torch.tensor([[3.0, 2.0, 1.0], [-1.0, 0.0, 1.0]], device=device)
        pts2 = torch.tensor([[1.0, 2.0, 1.0], [1.0, 0.0, -1.0]], device=device)
        dist_th = (2.0 ** 0.5,)
        with pytest.raises(TypeError, match="Expected input dist_th to be of type"):
            fusionutils.are_points_close(pts1, pts2, dist_th)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts1 = torch.tensor([[3.0, 2.0, 1.0]], device=device)
        pts2 = torch.tensor([[1.0, 2.0, 1.0], [1.0, 0.0, -1.0]], device=device)
        dist_th = 2.0 ** 0.5
        with pytest.raises(ValueError, match="tensor1 and tensor2 should have"):
            fusionutils.are_points_close(pts1, pts2, dist_th)

        pts1 = torch.tensor(
            [[3.0, 2.0, 1.0, 4.0], [-1.0, 0.0, 1.0, 3.0]], device=device
        )
        pts2 = torch.tensor(
            [[1.0, 2.0, 1.0, 2.0], [1.0, 0.0, -1.0, 4.0]], device=device
        )
        dist_th = 2.0 ** 0.5
        with pytest.raises(
            ValueError, match="Expected length of input tensors' dim-th"
        ):
            fusionutils.are_points_close(pts1, pts2, dist_th)


class TestAreNormalsSimilar:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_are_normals_similar(self, device):
        device = default_to_cpu_if_no_gpu(device)
        normals1 = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
            ],
            device=device,
        )

        normals2 = torch.tensor(
            [
                [1.0, 3.0, 5.0],
                [3.0, 2.0, 2.0],
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 1.0],
                [1.0, 0.0, -1.0],
            ],
            device=device,
        )
        normals1 = normals1 / (torch.sum((normals1 ** 2), -1, keepdim=True) ** 0.5)
        normals2 = normals2 / (torch.sum((normals2 ** 2), -1, keepdim=True) ** 0.5)
        dot_th = 0.879

        is_similar = fusionutils.are_normals_similar(normals1, normals2, dot_th)
        groundtruth = torch.tensor([0, 1, 1, 0, 0], device=device, dtype=bool)
        assert_allclose(is_similar.int(), groundtruth.int())

        dot_th = 0.878
        is_similar = fusionutils.are_normals_similar(normals1, normals2, dot_th)
        groundtruth = torch.tensor([1, 1, 1, 0, 0], device=device, dtype=bool)
        assert_allclose(is_similar.int(), groundtruth.int())

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        normals1 = ([3.0, 2.0, 1.0], [-1.0, 0.0, 1.0])
        normals2 = torch.tensor([[3.0, 2.0, 2.0], [1.0, 2.0, 3.0]], device=device)
        normals2 = normals2 / (torch.sum((normals2 ** 2), -1, keepdim=True) ** 0.5)
        dot_th = 0.879
        with pytest.raises(
            TypeError, match="Expected input tensor1 to be of type torch.Tensor"
        ):
            fusionutils.are_normals_similar(normals1, normals2, dot_th)
        with pytest.raises(
            TypeError, match="Expected input tensor2 to be of type torch.Tensor"
        ):
            fusionutils.are_normals_similar(normals2, normals1, dot_th)

        normals1 = torch.tensor([[3.0, 3.0, 3.0], [1.0, 2.0, 3.0]], device=device)
        normals2 = torch.tensor([[3.0, 2.0, 2.0], [1.0, 2.0, 3.0]], device=device)
        normals1 = normals1 / (torch.sum((normals1 ** 2), -1, keepdim=True) ** 0.5)
        normals2 = normals2 / (torch.sum((normals2 ** 2), -1, keepdim=True) ** 0.5)
        dot_th = 0.879
        fusionutils.are_normals_similar(normals1, normals2, dot_th)

        dot_th = (2.0,)
        with pytest.raises(TypeError, match="Expected input dot_th to be of type"):
            fusionutils.are_normals_similar(normals1, normals2, dot_th)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        normals1 = torch.tensor([[3.0, 3.0, 3.0]], device=device)
        normals2 = torch.tensor([[3.0, 2.0, 2.0], [1.0, 2.0, 3.0]], device=device)
        normals1 = normals1 / (torch.sum((normals1 ** 2), -1, keepdim=True) ** 0.5)
        normals2 = normals2 / (torch.sum((normals2 ** 2), -1, keepdim=True) ** 0.5)
        dot_th = 0.879
        with pytest.raises(ValueError, match="tensor1 and tensor2 should have"):
            fusionutils.are_normals_similar(normals1, normals2, dot_th)

        normals1 = torch.tensor(
            [[3.0, 3.0, 3.0, 4.0], [1.0, 2.0, 3.0, 2.0]], device=device
        )
        normals2 = torch.tensor(
            [[3.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 1.0]], device=device
        )
        normals1 = normals1 / (torch.sum((normals1 ** 2), -1, keepdim=True) ** 0.5)
        normals2 = normals2 / (torch.sum((normals2 ** 2), -1, keepdim=True) ** 0.5)
        dot_th = 0.879
        with pytest.raises(
            ValueError, match="Expected length of input tensors' dim-th"
        ):
            fusionutils.are_normals_similar(normals1, normals2, dot_th)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_warns(self, device):
        device = default_to_cpu_if_no_gpu(device)
        normals1 = torch.tensor([[3.0, 3.0, 3.0], [1.0, 2.0, 3.0]], device=device)
        normals2 = torch.tensor([[3.0, 2.0, 2.0], [1.0, 2.0, 3.0]], device=device)
        dot_th = 0.879
        with pytest.warns(RuntimeWarning, match="Max of dot product was "):
            fusionutils.are_normals_similar(normals1, normals2, dot_th)


class TestFindActiveMapPoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_find_active_map_points(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        # first frame: rgbdimages[:, 0]
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        # first frame: rgbdimages[:, 0]
        pc2im_bnhw = fusionutils.find_active_map_points(pointclouds, rgbdimages[:, 0])

        assert pc2im_bnhw.shape[0] == rgbdimages.valid_depth_mask[:, 0].sum()
        projected_colors = torch.zeros_like(colors.to(device))
        projected_colors[
            pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
        ] = pointclouds.colors_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]
        assert_allclose(
            projected_colors[:, 0:1],
            colors.to(device)[:, 0:1] * rgbdimages.valid_depth_mask[:, 0:1].float(),
        )

        # # sanity check visualization
        # import matplotlib
        # import matplotlib.pyplot as plt
        # import numpy as np

        # projected_colors = projected_colors.detach().cpu().numpy()
        # # colors = colors.detach().cpu().numpy()
        # colors = (colors * rgbdimages.valid_depth_mask[:, 0:1].float()).detach().cpu().numpy()
        # fig, axs = plt.subplots(2, 2, figsize=(10, 3))
        # fig.suptitle("Normal map")
        # axs[0][0].imshow(projected_colors[0, 0].squeeze().astype(np.uint8))
        # axs[0][1].imshow(projected_colors[1, 0].squeeze().astype(np.uint8))
        # axs[1][0].imshow(colors[0, 0].squeeze().astype(np.uint8))
        # axs[1][1].imshow(colors[1, 0].squeeze().astype(np.uint8))
        # plt.tight_layout()
        # plt.show()

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw = fusionutils.find_active_map_points(pointclouds, rgbdimages[:, 1])

        with pytest.raises(TypeError, match="Expected pointclouds to be of type"):
            fusionutils.find_active_map_points(rgbdimages, rgbdimages)

        with pytest.raises(TypeError, match="Expected rgbdimages to be of type"):
            fusionutils.find_active_map_points(pointclouds, pointclouds)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw = fusionutils.find_active_map_points(pointclouds, rgbdimages[:, 1])

        with pytest.raises(ValueError, match="Expected equal batch sizes"):
            fusionutils.find_active_map_points(pointclouds, rgbdimages[0, 1])

        with pytest.raises(
            ValueError, match="Expected rgbdimages to have sequence length of 1"
        ):
            fusionutils.find_active_map_points(pointclouds, rgbdimages[:, :])

    def test_visualize_normals(self):
        device = torch.device("cpu")
        pass

        # import matplotlib
        # import matplotlib.pyplot as plt
        # import numpy as np
        # channels_first = False
        # colors, depths, intrinsics, poses = load_test_data(channels_first)
        # rgbdimages = RGBDImages(colors.to(device), depths.to(device), intrinsics.to(device), poses.to(device), channels_first=channels_first).to(device)

        # sigma = 0.6
        # # first frame: rgbdimages[:, 0]
        # pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        # # second frame: rgbdimages[:, 1]
        # pc2im_bnhw = fusionutils.find_active_map_points(pointclouds, rgbdimages[:, 1])

        # # # sanity check visualization
        # projected_normals = torch.zeros_like(colors)
        # projected_normals[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]] = pointclouds.normals_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]

        # normal_map = projected_normals[:, 0].detach().cpu().numpy().squeeze() * -1
        # normal_map[..., 0], normal_map[..., 1] = normal_map[..., 1].copy(), normal_map[..., 0].copy()
        # normal_map = ((normal_map + 1) * 255 / 2).astype(np.uint8)

        # normal_map1 = rgbdimages.normal_map[:, 0].contiguous() * -1
        # normal_map1 = normal_map1.detach().cpu().numpy().squeeze()
        # normal_map1[..., 0], normal_map1[..., 1] = normal_map1[..., 1].copy(), normal_map1[..., 0].copy()
        # normal_map1 = ((normal_map1 + 1) * 255 / 2).astype(np.uint8)
        # fig, axs = plt.subplots(2, 2, figsize=(10, 3))
        # fig.suptitle("Normal maps")
        # axs[0][0].imshow(normal_map[0].squeeze())
        # axs[0][1].imshow(normal_map[1].squeeze())
        # axs[1][0].imshow(normal_map1[0].squeeze())
        # axs[1][1].imshow(normal_map1[1].squeeze())
        # plt.tight_layout()
        # plt.show()


class TestFindSimilarMapPoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_find_similar_map_points(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        # first frame: rgbdimages[:, 0]
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        # first frame: rgbdimages[:, 0]
        pc2im_bnhw_active = fusionutils.find_active_map_points(
            pointclouds, rgbdimages[:, 0]
        )
        pc2im_bnhw_similar, is_similar = fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 0], pc2im_bnhw_active, dist_th, dot_th
        )

        # Only points corresponding to zero normals (despite valid depths) will be removed
        pc2im_not_similar = pc2im_bnhw_active[is_similar == False]
        normal_maps = rgbdimages.normal_map
        frame_normals = torch.zeros_like(pointclouds.normals_padded)
        frame_normals[pc2im_not_similar[:, 0], pc2im_not_similar[:, 1]] = normal_maps[
            pc2im_not_similar[:, 0], 0, pc2im_not_similar[:, 2], pc2im_not_similar[:, 3]
        ]

        assert frame_normals.abs().max() == 0
        assert (
            pc2im_bnhw_active.shape[0] - pc2im_bnhw_similar.shape[0]
            == (
                pointclouds.normals_list[0].eq(0).all(-1).sum()
                + pointclouds.normals_list[1].eq(0).all(-1).sum()
            ).item()
        )
        assert pointclouds.points_list[0].eq(0).all(-1).sum().item() == 0
        assert pointclouds.points_list[1].eq(0).all(-1).sum().item() == 0

        # # sanity check visualization
        # import matplotlib
        # import matplotlib.pyplot as plt
        # import numpy as np

        # active_normals = torch.zeros_like(colors)
        # similar_normals = torch.zeros_like(colors)
        # active_normals[pc2im_bnhw_active[:, 0], 0, pc2im_bnhw_active[:, 2], pc2im_bnhw_active[:, 3]] = pointclouds.normals_padded[pc2im_bnhw_active[:, 0], pc2im_bnhw_active[:, 1]]
        # similar_normals[pc2im_bnhw_similar[:, 0], 0, pc2im_bnhw_similar[:, 2], pc2im_bnhw_similar[:, 3]] = pointclouds.normals_padded[pc2im_bnhw_similar[:, 0], pc2im_bnhw_similar[:, 1]]

        # active_normal_map = active_normals[:, 0].detach().cpu().numpy().squeeze() * -1
        # active_normal_map[..., 0], active_normal_map[..., 1] = active_normal_map[..., 1].copy(), active_normal_map[..., 0].copy()
        # active_normal_map = ((active_normal_map + 1) * 255 / 2).astype(np.uint8)

        # similar_normal_map = similar_normals[:, 0].detach().cpu().numpy().squeeze() * -1
        # similar_normal_map[..., 0], similar_normal_map[..., 1] = similar_normal_map[..., 1].copy(), similar_normal_map[..., 0].copy()
        # similar_normal_map = ((similar_normal_map + 1) * 255 / 2).astype(np.uint8)

        # # import pdb; pdb.set_trace();
        # normal_map = rgbdimages.normal_map[:, 0].contiguous() * -1
        # normal_map = normal_map.detach().cpu().numpy().squeeze()
        # normal_map[..., 0], normal_map[..., 1] = normal_map[..., 1].copy(), normal_map[..., 0].copy()
        # normal_map = ((normal_map + 1) * 255 / 2).astype(np.uint8)
        # fig, axs = plt.subplots(3, 2, figsize=(10, 3))
        # fig.suptitle("Normal maps")
        # axs[0][0].imshow(active_normal_map[0].squeeze().astype(np.uint8))
        # axs[0][1].imshow(active_normal_map[1].squeeze().astype(np.uint8))
        # axs[1][0].imshow(similar_normal_map[0].squeeze().astype(np.uint8))
        # axs[1][1].imshow(similar_normal_map[1].squeeze().astype(np.uint8))
        # axs[2][0].imshow(normal_map[0].squeeze().astype(np.uint8))
        # axs[2][1].imshow(normal_map[1].squeeze().astype(np.uint8))
        # plt.tight_layout()
        # plt.show()

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw_active = fusionutils.find_active_map_points(
            pointclouds, rgbdimages[:, 1]
        )
        pc2im_bnhw_similar, is_similar = fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_active, dist_th, dot_th
        )

        with pytest.raises(TypeError, match="Expected pointclouds to be of type"):
            fusionutils.find_similar_map_points(
                rgbdimages[:, 1], rgbdimages[:, 1], pc2im_bnhw_active, dist_th, dot_th
            )
        with pytest.raises(TypeError, match="Expected rgbdimages to be of type"):
            fusionutils.find_similar_map_points(
                pointclouds, pointclouds, pc2im_bnhw_active, dist_th, dot_th
            )

        with pytest.raises(TypeError, match="Expected input pc2im_bnhw to be of type"):
            fusionutils.find_similar_map_points(
                pointclouds,
                rgbdimages[:, 1],
                (
                    1,
                    2,
                    3,
                ),
                dist_th,
                dot_th,
            )

        fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_active.long(), dist_th, dot_th
        )
        with pytest.raises(
            TypeError, match="Expected input pc2im_bnhw to have dtype of"
        ):
            fusionutils.find_similar_map_points(
                pointclouds,
                rgbdimages[:, 1],
                pc2im_bnhw_active.short(),
                dist_th,
                dot_th,
            )
        with pytest.raises(
            TypeError, match="Expected input pc2im_bnhw to have dtype of"
        ):
            fusionutils.find_similar_map_points(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw_active.int(), dist_th, dot_th
            )
        with pytest.raises(
            TypeError, match="Expected input pc2im_bnhw to have dtype of"
        ):
            fusionutils.find_similar_map_points(
                pointclouds,
                rgbdimages[:, 1],
                pc2im_bnhw_active.float(),
                dist_th,
                dot_th,
            )

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw_active = fusionutils.find_active_map_points(
            pointclouds, rgbdimages[:, 1]
        )
        pc2im_bnhw_similar, is_similar = fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_active, dist_th, dot_th
        )

        fusionutils.find_similar_map_points(
            pointclouds[0],
            rgbdimages[0, 1],
            pc2im_bnhw_active[pc2im_bnhw_active[:, 0] == 0],
            dist_th,
            dot_th,
        )
        with pytest.raises(ValueError, match="Expected equal batch sizes for"):
            fusionutils.find_similar_map_points(
                pointclouds, rgbdimages[0, 1], pc2im_bnhw_active, dist_th, dot_th
            )

        pointclouds1 = pointclouds.clone()
        pointclouds1._normals_list = None
        pointclouds1._normals_padded = None
        pointclouds1._has_normals = False
        with pytest.raises(
            ValueError, match="Pointclouds must have normals for finding similar"
        ):
            fusionutils.find_similar_map_points(
                pointclouds1, rgbdimages[:, 1], pc2im_bnhw_active, dist_th, dot_th
            )

        with pytest.raises(
            ValueError, match="Expected rgbdimages to have sequence length of 1"
        ):
            fusionutils.find_similar_map_points(
                pointclouds, rgbdimages[:, :], pc2im_bnhw_active, dist_th, dot_th
            )

        with pytest.warns(RuntimeWarning, match="No similar map points "):
            fusionutils.find_similar_map_points(
                pointclouds + 10,
                rgbdimages[:, 1],
                pc2im_bnhw_active[0:1],
                dist_th,
                dot_th,
            )
        with pytest.raises(ValueError, match="Expected pc2im_bnhw.ndim of 2"):
            fusionutils.find_similar_map_points(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw_active[0], dist_th, dot_th
            )

        with pytest.raises(ValueError, match="Expected pc2im_bnhw.shape"):
            fusionutils.find_similar_map_points(
                pointclouds,
                rgbdimages[:, 1],
                pc2im_bnhw_active[:, :3],
                dist_th,
                dot_th,
            )


class TestFindBestUniqueCorrespondences:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_sorting_correspondences(self, device):
        device = default_to_cpu_if_no_gpu(device)
        sigma = 0.6
        pts1 = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [-0.5, -0.5, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        ).unsqueeze(0)
        pc2im_bnhw = torch.tensor(
            [
                [0, 4, 0, 0],
                [0, 0, 1, 1],
                [0, 5, 1, 0],
                [0, 1, 0, 0],
                [0, 2, 1, 1],
                [0, 3, 0, 0],
            ],
            device=device,
            dtype=torch.int64,
        )
        features = fusionutils.get_alpha(pts1, sigma, keepdim=True)
        features[0, 3] = 1e-12
        pointclouds = gs.structures.Pointclouds(points=pts1, features=features)
        image = (
            torch.tensor(
                [
                    [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    [[0.0, 5.0, 1.0], [8.0, 8.0, 8.0]],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        depths = torch.ones_like(image[..., 0:1])
        intrinsics = (
            torch.tensor(
                [
                    [2.0, 0.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        rgbdimages = RGBDImages(image, depths, intrinsics).to(device)
        # rgbdimages.vertex_map -> tensor(
        #     [
        #         [[-0.5, -0.5, 1.], [0., -0.5, 1.]],
        #         [[-0.5, 0., 1.], [0., 0., 1.]]
        #     ]
        # )
        pc2im_bnhw_unique = fusionutils.find_best_unique_correspondences(
            pointclouds, rgbdimages, pc2im_bnhw
        )

        groundtruth_pc2im_bnhw = torch.tensor(
            [
                [0, 4, 0, 0],
                [0, 5, 1, 0],
                [0, 2, 1, 1],
            ],
            device=device,
            dtype=torch.int64,
        )
        assert_allclose(pc2im_bnhw_unique, groundtruth_pc2im_bnhw)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_find_best_unique_correspondences(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        ).to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        # first frame: rgbdimages[:, 0]
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        # first frame: rgbdimages[:, 0]
        pc2im_bnhw_active = fusionutils.find_active_map_points(
            pointclouds, rgbdimages[:, 0]
        )
        pc2im_bnhw_similar, _ = fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 0], pc2im_bnhw_active, dist_th, dot_th
        )
        pc2im_bnhw_unique = fusionutils.find_best_unique_correspondences(
            pointclouds, rgbdimages[:, 0], pc2im_bnhw_similar
        )

        assert pc2im_bnhw_unique.shape[0] == pc2im_bnhw_similar.shape[0]
        assert_allclose(pc2im_bnhw_unique, pc2im_bnhw_similar)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        ).to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw_active = fusionutils.find_active_map_points(
            pointclouds, rgbdimages[:, 1]
        )
        pc2im_bnhw_similar, _ = fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_active, dist_th, dot_th
        )
        pc2im_bnhw_unique = fusionutils.find_best_unique_correspondences(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_similar
        )

        with pytest.raises(TypeError, match="Expected pointclouds to be of type"):
            fusionutils.find_best_unique_correspondences(
                pc2im_bnhw_similar, rgbdimages[:, 1], pc2im_bnhw_similar
            )

        with pytest.raises(TypeError, match="Expected input pc2im_bnhw to be of type"):
            fusionutils.find_best_unique_correspondences(
                pointclouds, rgbdimages[:, 1], pointclouds
            )

        with pytest.raises(TypeError, match="Expected input pc2im_bnhw to have dtype"):
            fusionutils.find_best_unique_correspondences(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw_similar.float()
            )

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        ).to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw_active = fusionutils.find_active_map_points(
            pointclouds, rgbdimages[:, 1]
        )
        pc2im_bnhw_similar, _ = fusionutils.find_similar_map_points(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_active, dist_th, dot_th
        )
        pc2im_bnhw_unique = fusionutils.find_best_unique_correspondences(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_similar
        )

        pointclouds1 = pointclouds.clone()
        pointclouds1._features_list = None
        pointclouds1._features_padded = None
        pointclouds1._has_features = False
        with pytest.raises(ValueError, match="Pointclouds must have features for"):
            fusionutils.find_best_unique_correspondences(
                pointclouds1, rgbdimages[:, 1], pc2im_bnhw_similar
            )
        with pytest.raises(ValueError, match="Expected rgbdimages to have "):
            fusionutils.find_best_unique_correspondences(
                pointclouds, rgbdimages, pc2im_bnhw_similar
            )
        fusionutils.find_best_unique_correspondences(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw_similar[0:1]
        )
        with pytest.raises(ValueError, match="Expected pc2im_bnhw.ndim of 2"):
            fusionutils.find_best_unique_correspondences(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw_similar[0]
            )

        with pytest.raises(ValueError, match="Expected pc2im_bnhw.shape"):
            fusionutils.find_best_unique_correspondences(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw_similar[:, :3]
            )


class TestFindCorrespondences:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_find_correspondences(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw = fusionutils.find_correspondences(
            pointclouds, rgbdimages[:, 0], dist_th, dot_th
        )

        # How many points will pc2im_bnhw have?
        num_valid_points = rgbdimages.valid_depth_mask[:, 0].sum()
        valid_zero_normals = (
            rgbdimages.normal_map[:, 0].eq(0).all(-1).int()
            - (rgbdimages.valid_depth_mask[:, 0] == False).squeeze(-1).int()
        )
        assert valid_zero_normals.abs().sum() == valid_zero_normals.sum()
        assert (
            rgbdimages.vertex_map[:, 0].eq(0).all(-1).int()
            - (rgbdimages.valid_depth_mask[:, 0] == False).squeeze(-1).int()
        ).abs().sum() == 0
        num_valid_zero_normals = valid_zero_normals.sum()
        assert pc2im_bnhw.shape[0] == num_valid_points - num_valid_zero_normals

        # Errors are handled by other functions used inside of fusionutils.find_correspondences()


class TestFuseWithMap:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_fuse_with_map(self, device):
        device = default_to_cpu_if_no_gpu(device)
        sigma = 0.6
        pts1 = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        pc2im_bnhw = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 2, 0, 1],
                [0, 5, 1, 0],
            ],
            device=device,
            dtype=torch.int64,
        )
        image = (
            torch.tensor(
                [
                    [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    [[0.0, 5.0, 1.0], [8.0, 8.0, 8.0]],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        depths = torch.ones_like(image[..., 0:1]) * 1e-20
        intrinsics = torch.rand(4, 4).unsqueeze(0).unsqueeze(0).to(device)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        features = torch.ones_like(pts1[..., 0:1])

        rgbdimages = RGBDImages(
            image, depths, intrinsics, poses, channels_first=False
        )  # .to(device)
        pointclouds = gs.structures.Pointclouds(
            points=pts1, normals=pts1, colors=pts1, features=features
        )
        pointclouds = fusionutils.fuse_with_map(
            pointclouds, rgbdimages, pc2im_bnhw, sigma
        )

        groundtruth_colors = torch.tensor(
            [
                [
                    [5.0000, 5.0000, 5.0000],
                    [1.5000, 2.0000, 1.5000],
                    [0.5000, 2.0000, 1.5000],
                    [3.0000, 2.0000, 1.0000],
                    [-1.0000, 0.0000, 1.0000],
                    [0.0000, 2.5000, 0.5000],
                    [8.0000, 8.0000, 8.0000],
                ]
            ],
            device=device,
            dtype=torch.float,
        )
        assert_allclose(groundtruth_colors, pointclouds.colors_padded)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_append_no_points(self, device):
        device = default_to_cpu_if_no_gpu(device)
        sigma = 0.6
        pts1 = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        pc2im_bnhw = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 2, 0, 1],
                [0, 4, 1, 1],
                [0, 5, 1, 0],
            ],
            device=device,
            dtype=torch.int64,
        )
        image = (
            torch.tensor(
                [
                    [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    [[0.0, 5.0, 1.0], [8.0, 8.0, 8.0]],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        depths = torch.zeros_like(image[..., 0:1])
        intrinsics = torch.rand(4, 4).unsqueeze(0).unsqueeze(0).to(device)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        features = torch.ones_like(pts1[..., 0:1])

        rgbdimages = RGBDImages(
            image, depths, intrinsics, poses, channels_first=False
        ).to(device)
        pointclouds = gs.structures.Pointclouds(
            points=pts1, normals=pts1, colors=pts1, features=features
        )
        pointclouds = fusionutils.fuse_with_map(
            pointclouds, rgbdimages, pc2im_bnhw, sigma
        )
        # This should not raise an error

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_type_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw = fusionutils.find_correspondences(
            pointclouds, rgbdimages[:, 1], dist_th, dot_th
        )
        pointclouds = fusionutils.fuse_with_map(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw, sigma
        )

        with pytest.raises(TypeError, match="Expected pointclouds to be of type"):
            fusionutils.fuse_with_map(
                rgbdimages[:, 1], rgbdimages[:, 1], pc2im_bnhw, sigma
            )

        with pytest.raises(TypeError, match="Expected rgbdimages to be of type"):
            fusionutils.fuse_with_map(pointclouds, pointclouds, pc2im_bnhw, sigma)

        with pytest.raises(TypeError, match="Expected input pc2im_bnhw to be of type"):
            fusionutils.fuse_with_map(pointclouds, rgbdimages[:, 1], (1, 2, 3), sigma)

        with pytest.raises(TypeError, match="Expected input pc2im_bnhw to have dtype"):
            fusionutils.fuse_with_map(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw.float(), sigma
            )

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_value_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        pc2im_bnhw = fusionutils.find_correspondences(
            pointclouds, rgbdimages[:, 1], dist_th, dot_th
        )
        pointclouds = fusionutils.fuse_with_map(
            pointclouds, rgbdimages[:, 1], pc2im_bnhw, sigma
        )

        pointclouds1 = pointclouds.clone()
        pointclouds1._normals_list = None
        pointclouds1._normals_padded = None
        pointclouds1._has_normals = False
        with pytest.raises(ValueError, match="Pointclouds must have normals"):
            fusionutils.fuse_with_map(pointclouds1, rgbdimages[:, 1], pc2im_bnhw, sigma)

        pointclouds1 = pointclouds.clone()
        pointclouds1._colors_list = None
        pointclouds1._colors_padded = None
        pointclouds1._has_colors = False
        with pytest.raises(ValueError, match="Pointclouds must have colors"):
            fusionutils.fuse_with_map(pointclouds1, rgbdimages[:, 1], pc2im_bnhw, sigma)

        pointclouds1 = pointclouds.clone()
        pointclouds1._features_list = None
        pointclouds1._features_padded = None
        pointclouds1._has_features = False
        with pytest.raises(ValueError, match="Pointclouds must have features"):
            fusionutils.fuse_with_map(pointclouds1, rgbdimages[:, 1], pc2im_bnhw, sigma)

        with pytest.raises(ValueError, match="Expected pc2im_bnhw.ndim of 2"):
            fusionutils.fuse_with_map(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw[0, :], sigma
            )

        with pytest.raises(ValueError, match="Expected pc2im_bnhw.shape"):
            fusionutils.fuse_with_map(
                pointclouds, rgbdimages[:, 1], pc2im_bnhw[:, :3], sigma
            )


class TestUpdateMapFusion:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_update_map_fusion(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        sigma = 0.6

        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        init_num_points = pointclouds.num_points_per_pointcloud
        pointclouds = fusionutils.update_map_fusion(
            pointclouds, rgbdimages[:, 1], dist_th, dot_th, sigma
        )
        updated_num_points = pointclouds.num_points_per_pointcloud
        assert updated_num_points.gt(init_num_points).all()

        # change params s.t. more points get fused
        dist_th = 0.4 ** 0.5
        dot_th = 0.5
        sigma = 0.6

        pointclouds2 = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        init_num_points2 = pointclouds2.num_points_per_pointcloud
        pointclouds2 = fusionutils.update_map_fusion(
            pointclouds2, rgbdimages[:, 1], dist_th, dot_th, sigma
        )
        updated_num_points2 = pointclouds2.num_points_per_pointcloud
        assert updated_num_points2.gt(init_num_points2).all()
        assert updated_num_points.gt(updated_num_points2).all()

    """
    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND)
    @pytest.mark.parametrize("device", ("cpu", ))
    # @pytest.mark.parametrize("device", ("cpu",  ))
    def test_update_map_fusion(self, device):
        channels_first = False
        dataset = Scannet(SCANNET_ROOT, SCANNET_META_ROOT, ("scene0333_00", "scene0636_00", ), start=0, end=4, 
            height=240, width=320, channels_first=channels_first)
        loader = DataLoader(dataset=dataset, batch_size=2)
        colors, depths, intrinsics, poses, *_ = next(iter(loader))
        rgbdimages = RGBDImages(colors.to(device), depths.to(device), intrinsics.to(device), poses.to(device), channels_first=channels_first).to(device)

        dist_th = 0.05 ** 0.5
        dot_th = 0.9
        sigma = 0.6

        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma).to(device)
        init_num_points = pointclouds.num_points_per_pointcloud
        pointclouds = fusionutils.update_map_fusion(pointclouds, rgbdimages[:, 0], dist_th, dot_th, sigma)
        updated_num_points = pointclouds.num_points_per_pointcloud
        assert (init_num_points == updated_num_points).all()
        # TODO: Ideally, if using same frames, init_num_points should be equal to 
        # updated_num_points, but it is not, because some valid depth points have zero 
        # normals so they get re-added to the map
    """
