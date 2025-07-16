from collections import OrderedDict

import numpy as np

from splat_py.structs import Gaussians


def export_model_as_ply(model: Gaussians, file_path):
    """ export functionality """
    model.verify_sizes()

    model_out = OrderedDict()

    xyz = model.xyz.detach().cpu().numpy()
    model_out["x"] = xyz[:, 0]
    model_out["y"] = xyz[:, 1]
    model_out["z"] = xyz[:, 2]

    num_elements = model_out["x"].shape[0]
    model_out["nx"] = np.zeros(num_elements, dtype=np.float32)
    model_out["ny"] = np.zeros(num_elements, dtype=np.float32)
    model_out["nz"] = np.zeros(num_elements, dtype=np.float32)

    # save colors (model.rgb represents the spherical harmonics dc coefficients for l=0)
    shs_0 = model.rgb.detach().cpu().numpy()
    for i in range(3):
        model_out[f"f_dc_{i}"] = shs_0[:, i, None]

    # save higher order spherical harmonics coefficients (stored in f_rest_{...})
    if model.sh is not None:
        sh = model.sh.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        for i in range(sh.shape[-1]):
            # Usually there are 3 x 15 (num_rgb_channels x num_spherical_coefficients) = 45
            model_out[f"f_rest_{i}"] = sh[:, i, None]

    # save opacities
    model_out["opacity"] = model.opacity.detach().cpu().numpy()

    # save scale (divided into scale_0, scale_1, scale_2)
    for i in range(3):
        scales = model.scale.detach().cpu().numpy()
        model_out[f"scale_{i}"] = scales[:, i, None]

    # save rotations (divided into rot_0, ..., rot_3; representing quaternions)
    for i in range(4):
        rot = model.quaternion.detach().cpu().numpy()
        model_out[f"rot_{i}"] = rot[:, i, None]

    write_ply_file(out_dict=model_out, file_path=file_path)

def write_ply_file(out_dict, file_path):

    num_elements = out_dict["x"].shape[0]
    with open(file_path, "wb") as ply_file:
        # PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(f"element vertex {num_elements}\n".encode())

        # Define vertex properties (ordered!)
        for key, tensor in out_dict.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())

        ply_file.write(b"end_header\n")

        # Write vertex data
        for i in range(num_elements):
            for tensor in out_dict.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())