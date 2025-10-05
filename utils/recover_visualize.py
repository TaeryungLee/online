# representation: 272 dim
# :2 local xz velocities of root, no heading, can recover translation
# 2:8  heading angular velocities, 6d rotation, can recover heading
# 8:8+3*njoint local position, no heading, all at xz origin
# 8+3*njoint:8+6*njoint local velocities, no heading, all at xz origin, can recover local postion
# 8+6*njoint:8+12*njoint local rotations, 6d rotation, no heading, all frames z+

import numpy as np
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import torch
import os
import visualization.plot_3d_global as plot_3d
import argparse
import tqdm
import imageio
from PIL import Image

from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, PerspectiveCameras, RasterizationSettings,
    TexturesVertex, look_at_view_transform, FoVPerspectiveCameras, Materials
)
from pytorch3d.structures import Meshes



def findAllFile(base, endswith='.npy'):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.endswith(endswith):
                file_path.append(fullname)
    return file_path

def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

    
def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1

    return -heading, axis

def accumulate_rotations(relative_rotations):
    """Accumulate relative rotations to get the overall rotation"""
    # Support both numpy arrays and torch tensors; prefer torch when given
    if isinstance(relative_rotations, np.ndarray):
        R_total = [relative_rotations[0]]
        for R_rel in relative_rotations[1:]:
            R_total.append(np.matmul(R_rel, R_total[-1]))
        return np.stack(R_total, axis=0)
    elif torch.is_tensor(relative_rotations):
        R_total = [relative_rotations[0]]
        for R_rel in relative_rotations[1:]:
            R_total.append(torch.matmul(R_rel, R_total[-1]))
        return torch.stack(R_total, dim=0)
    else:
        raise TypeError("relative_rotations must be a numpy.ndarray or torch.Tensor")

def recover_from_local_position(final_x, njoint):
    # take positions_no_heading: local position on xz ori, no heading
    # velocities_root_xy_no_heading: to recover translation
    # global_heading_diff_rot: to recover root rotation
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:,8:8+3*njoint].reshape(nfrm, -1, 3) # frames, njoints * 3
    velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
    global_heading_diff_rot = final_x[:,2:8] # frames, 6

    # recover global heading
    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), positions_no_heading[...,None]).squeeze(-1)

    # recover root translation
    # add heading to velocities_root_xy_no_heading

    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading


# add hip height to translation when recoverring from rotation
def recover_from_local_rotation(final_x, njoint):
    is_array = isinstance(final_x, np.ndarray)
    if is_array:
        final_x = torch.from_numpy(final_x)
    else:
        final_x = final_x.detach().cpu()
    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(final_x[:,8+6*njoint:8+12*njoint].reshape(nfrm, -1, 6))
    global_heading_diff_rot = final_x[:,2:8]
    velocities_root_xy_no_heading = final_x[:,:2]
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(global_heading_diff_rot))
    inv_global_heading_rot = torch.transpose(global_heading_rot, 2, 1)
    # recover root rotation
    rotations_matrix[:,0,...] = torch.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])
    velocities_root_xyz_no_heading = torch.zeros((velocities_root_xy_no_heading.shape[0], 3)).to(rotations_matrix.dtype)
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = torch.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    root_translation = torch.cumsum(velocities_root_xyz_no_heading, dim=0)
    root_translation[:, 1] = height
    smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
    return smpl_85


def rotations_matrix_to_smpl85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(rotations_matrix).reshape(nfrm, -1)
    smpl_85 = torch.cat([axis_angle, torch.zeros((nfrm, 6)), translation, torch.zeros((nfrm, 10))], dim=-1)
    return smpl_85




def smpl85_2_smpl322(smpl_85_data):
    result = np.concatenate((smpl_85_data[:,:66], np.zeros((smpl_85_data.shape[0], 90)), np.zeros((smpl_85_data.shape[0], 3)), np.zeros((smpl_85_data.shape[0], 50)), np.zeros((smpl_85_data.shape[0], 100)), smpl_85_data[:,72:72+3], smpl_85_data[:,75:]), axis=-1)
    return result

def visualize_smpl_85(data, smpl_model, title=None, output_path='visualize_result', name='', fps=60):
    # data: torch.Size([nframe, 85])
    data = data.to(torch.float32)

    global_orient = data[:, :3]
    body_pose = data[:, 3:72]
    transl = data[:, 72:72+3]
    betas = torch.zeros_like(body_pose[:, :10])

    smpl_output = smpl_model.forward(body_pose=body_pose, global_orient=global_orient, transl=transl)
    verts = smpl_output.vertices


    render_from_verts(verts, smpl_model.faces, outdir=output_path, name=name, fps=fps, png=False, gif=True)
    return


def rgba_to_rgb_white_bg(rgba_image, threshold=5):
    """
    Convert RGBA image to RGB, keeping transparency effect for non-background areas,
    and setting background to pure white.

    Parameters:
        rgba_image: np.array (H, W, 4), float32 (0~1) or uint8 (0~255)
        threshold: int, alpha threshold below which it's considered background

    Returns:
        out_rgb: np.array (H, W, 3), uint8
    """
    if rgba_image.dtype != np.uint8:
        rgba_image = (rgba_image * 255).astype(np.uint8)

    alpha = rgba_image[..., 3:4] / 255.0  # (H,W,1)
    rgb = rgba_image[..., :3].astype(np.float32)

    white = np.ones_like(rgb, dtype=np.float32) * 255

    # Alpha blending with white background
    blended = rgb * 0.8 + white * (1 - 0.8)

    # If alpha is near zero, force pure white
    mask_background = rgba_image[..., 3:4] < threshold
    blended[mask_background.squeeze(-1)] = 255

    return blended.astype(np.uint8)



def render_from_verts(vertices, faces, outdir='test_vis', device_id=0, name=None, png=False, gif=True, fps=60):
    # breakpoint()
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    frames = vertices.shape[0]  # [nb_frames, nb_joints, 3]
    njoints = vertices.shape[1]

    faces = torch.tensor(faces.astype(np.int64), dtype=torch.int64, device=device)

    vertices = vertices.to(device)

    MINS = vertices.view(-1, 3).min(dim=0)[0]
    MAXS = vertices.view(-1, 3).max(dim=0)[0]

    vid = []
    vid_png = []



    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    ground_y = MINS[1] - 1e-3

    # Build a checkerboard ground (0.2 x 0.2 tiles) as a pytorch3d mesh
    tile = 0.5
    minx_f, maxx_f = float(minx.item()), float(maxx.item())
    minz_f, maxz_f = float(minz.item()), float(maxz.item())
    gy_f = float(ground_y.item())

    nx = max(1, int(np.ceil((maxx_f - minx_f) / tile)))
    nz = max(1, int(np.ceil((maxz_f - minz_f) / tile)))

    light = [0.92, 0.92, 0.92]
    dark  = [0.75, 0.75, 0.75]

    g_verts = []
    g_faces = []
    g_colors = []

    def add_tri(v0, v1, v2, color):
        base = len(g_verts)
        g_verts.extend([v0, v1, v2])
        g_faces.append([base + 0, base + 1, base + 2])
        g_colors.extend([color, color, color])

    for ix in range(nx):
        x0 = minx_f + ix * tile
        x1 = min(x0 + tile, maxx_f)
        for iz in range(nz):
            z0 = minz_f + iz * tile
            z1 = min(z0 + tile, maxz_f)
            color = light if ((ix + iz) % 2 == 0) else dark
            # two triangles per cell: (x0,z0)-(x0,z1)-(x1,z1) and (x0,z0)-(x1,z1)-(x1,z0)
            add_tri([x0, gy_f, z0], [x0, gy_f, z1], [x1, gy_f, z1], color)
            add_tri([x0, gy_f, z0], [x1, gy_f, z1], [x1, gy_f, z0], color)

    ground_verts = torch.tensor(g_verts, dtype=vertices.dtype, device=device)
    ground_faces = torch.tensor(g_faces, dtype=torch.int64, device=device)



    # Setup the camera (static for all frames)
    R, T = look_at_view_transform(dist=4.0, elev=30, azim=0)  # optional 튜닝 가능
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=62)

    # Setup rasterizer and shader
    raster_settings = RasterizationSettings(
        image_size=1280,  # 1920
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=64,  # 기본값은 32 또는 0
        max_faces_per_bin=20000,
    )

    lights = PointLights(device=device, location=[[0.0, 2.0, 2.0]]) # 그림자 강하게 생김, 바꿔보기

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    materials = Materials(
	device='cuda',
	specular_color=[[0.0, 0.0, 0.0]],
	shininess=0.0
    )
    # Shineness 0
    # 메쉬 색깔 더 밝게

    for i in tqdm.tqdm(range(frames), leave=False, desc="rendering images"):
        verts_frame = vertices[i].unsqueeze(0)  # (1, J, 3)

        # Combine character mesh with ground mesh into one scene mesh
        num_char_verts = verts_frame.shape[1]
        combined_verts = torch.cat([verts_frame.squeeze(0), ground_verts], dim=0).unsqueeze(0)
        combined_faces = torch.cat([faces, ground_faces + num_char_verts], dim=0).unsqueeze(0)

        # Per-vertex colors: character white, ground checkerboard from g_colors
        char_rgb = torch.ones_like(verts_frame)
        ground_rgb = torch.tensor(g_colors, device=device, dtype=vertices.dtype).unsqueeze(0)
        verts_rgb = torch.cat([char_rgb, ground_rgb], dim=1)
        textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(verts=combined_verts, faces=combined_faces, textures=textures)

        # Render
        with torch.no_grad():
            images = renderer(mesh, materials=materials)
        image = images[0, ..., :3].cpu().numpy()  # RGB

        # Remove white background
        # color = (image * 255).astype(np.uint8)

        color = (image * 255).astype(np.uint8)

        if png:
            img = Image.fromarray(color, mode="RGB")
            img.save(".test1.png")

        # img = Image.fromarray(color_crop, mode="RGB")
        # img.save("./.test_gif.png")

        # img = Image.fromarray(color_crop_png, mode="RGBA")
        # img.save("./.test_png.png")
        # breakpoint()

        vid.append(color)
        if png:
            vid_png.append(color)
    out = np.stack(vid, axis=0)  # (frames, H, W, 4)
    if png:
        out_png = np.stack(vid_png, axis=0)  # (frames, H, W, 4)

    if png:
        safepath = os.path.join(outdir, 'png', name)[:200]
        os.makedirs(safepath, exist_ok=True)
        for i, img in enumerate(out_png):
            img = Image.fromarray(img, mode="RGBA")
            save_path = os.path.join(safepath, f"{i:03d}.png")
            img.save(save_path)
    if (not png) and gif:
        os.makedirs(os.path.join(outdir, 'gif'), exist_ok=True)
        safe_name = (name + '.gif')[:200]  # Simply truncate to 200 characters
        imageio.mimsave(os.path.join(outdir, 'gif', safe_name), out, fps=fps)
    
    return



def visualize_pos_xyz(xyz, title_batch=None, output_path='./', name='', fps=60):
    # xyz: torch.Size([nframe, 22, 3])
    xyz = xyz[:1]   
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_xyz = plot_3d.draw_to_batch(xyz, title_batch, [f'{output_path}/pos_{name}.mp4'], fps=fps)
    return output_path


if __name__ == '__main__':
    njoint = 22
    parser = argparse.ArgumentParser(description='Visualize new representation.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input path')
    parser.add_argument('--mode', type=str, required=True, default='rot', choices=['rot', 'pos'], help='Recover from rotation or position')
    parser.add_argument('--output_dir', type=str, required=True, help='Output path')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    for data_path in tqdm.tqdm(findAllFile(args.input_dir, endswith='.npy')):
        data_272 = np.load(data_path)
        if args.mode == 'rot':
            # recover from rotation
            from visualization.smplx2joints import process_smplx_data
            global_rotation = recover_from_local_rotation(data_272, njoint)  # get the 85-dim smpl data
            visualize_smpl_85(global_rotation, output_path=args.output_dir, name=data_path.split('/')[-1].split('.')[0])
            print(f"Visualized results are saved in {args.output_dir}")
        else:
            # recover from position
            global_position = recover_from_local_position(data_272, njoint)
            global_position = np.expand_dims(global_position, axis=0)
            visualize_pos_xyz(global_position, output_path=args.output_dir, name=data_path.split('/')[-1].split('.')[0])
            print(f"Visualized results are saved in {args.output_dir}")
