import torch

from splat_cuda import (
    get_sorted_gaussian_list,
)


def get_splats(
    uvs,
    tiles,
    conic,
    xyz_camera_frame,
    mh_dist,
):
    # nan in xyz will cause an unrecoverable sorting failure
    if torch.any(~torch.isfinite(xyz_camera_frame)):
        print("xyz_camera_frame has NaN")
        exit()
    return get_sorted_gaussian_list(
        1024,
        uvs,
        xyz_camera_frame,
        conic,
        tiles.x_tiles_count,
        tiles.y_tiles_count,
        mh_dist,
    )
