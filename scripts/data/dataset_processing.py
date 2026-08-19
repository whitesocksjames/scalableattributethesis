"""
python dataset_processing.py --process='mesh2pc' --input_rootdir='../../dataset/ShapeNet/mesh/' \
    --output_rootdir='../../dataset/ShapeNet/pc_vox8' --input_format='obj' --output_format='h5' --output_length=10000 --num_points=800000 --resolution=255

python dataset_processing.py --process='mapcolor' --input_rootdir='../../dataset/ShapeNet/pc_vox8/'  \
    --image_rootdir='../../dataset/COCO/' --input_format='h5' 
    --output_format='h5' --output_rootdir='../../dataset/ShapeNet/pc_vox8_color/' --output_length=10

apt-get update
apt-get install libglib2.0-dev
pip install opencv-python
"""
import MinkowskiEngine as ME
import open3d as o3d
import numpy as np
import torch
import os
import argparse
import glob
from tqdm import tqdm
from data_utils import read_ply_o3d_geo, write_ply_ascii_geo, write_h5_geo, \
                        read_ply_ascii_geo, read_h5_geo, \
                        read_h5, read_ply_ascii, write_h5, write_ply_ascii, quantize
                        # read_ply_o3d, write_ply_o3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes

###################################### mesh2points #############################################
def mesh2points(mesh, num_points):
    """
    sample points uniformly
    !pip install open3d
    """
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=int(num_points))
    except:
        print("ERROR mesh2points !")
        return np.asarray([[0,0,0]])
    points = np.asarray(pcd.points)

    return points

def random_rotate(points):
    # get_rotate_matrix
    matrix = np.eye(3,dtype='float32')
    matrix[0,0] *= np.random.randint(0,2)*2-1
    matrix = np.dot(matrix, np.linalg.qr(np.random.randn(3,3))[0])
    # random_rotate
    points = np.dot(points, matrix)

    return points

def main_mesh2pc(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points, resolution=255):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    import random
    random.shuffle(input_filedirs)
    input_filedirs=input_filedirs[:input_length]
    print("input length:\t", len(input_filedirs))
    for idx, input_filedir in enumerate(tqdm(input_filedirs)):
        # mesh2points
        mesh = o3d.io.read_triangle_mesh(input_filedir)
        points = mesh2points(mesh, num_points)
        if len(points)==1: continue
        # random rotate
        points = random_rotate(points)
        # quantize
        points = quantize(points, resolution=resolution)
        print("="*20, "nums", len(points))
        # save
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0])
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        if output_format == 'ply': write_ply_ascii_geo(output_filedir+'.ply', points)
        if output_format == 'h5': write_h5_geo(output_filedir+'.h5', points)
        if idx >= output_length: break

    return 


def main_mesh2pc_with_color(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points, resolution=255):
    print(resolution)
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    print(input_filedirs)
    import random
    # random.shuffle(input_filedirs)
    input_filedirs = input_filedirs[:input_length]
    print("input length:\t", len(input_filedirs))
    for idx, input_filedir in enumerate(tqdm(input_filedirs)):

        try:
            print(input_filedir)
            # mesh2points
            mesh = load_objs_as_meshes([input_filedir])
            xyz, rgb = sample_points_from_meshes(meshes=mesh, num_samples=int(num_points), return_textures=True)
            xyz = xyz[0, :, :].numpy()
            rgb = rgb[0, :, :].numpy()

            # quantize
            points = quantize(xyz, resolution=resolution)
            print("="*20, "nums", len(points))

            # unique
            coords, feats = ME.utils.sparse_collate([torch.tensor(points)], [torch.tensor(rgb)])
            pc = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device='cpu')
            print(f'{len(points)} -> {len(pc)}')

            # save
            output_filedir_h5 = output_rootdir + '/h5' + input_filedir[len(input_rootdir):].split('.')[0]
            output_folder_h5, _ = os.path.split(output_filedir_h5)
            os.makedirs(output_folder_h5, exist_ok=True)

            output_filedir_ply = output_rootdir + '/ply' + input_filedir[len(input_rootdir):].split('.')[0]
            output_folder_ply, _ = os.path.split(output_filedir_ply)
            os.makedirs(output_folder_ply, exist_ok=True)

            write_ply_ascii(output_filedir_ply + '.ply', pc.C.numpy()[:, 1:], np.clip((pc.F.numpy() * 255).round(), 0, 255).astype('uint8'))
            write_h5(output_filedir_h5 + '.h5', pc.C.numpy()[:, 1:], np.clip((pc.F.numpy() * 255).round(), 0, 255).astype('uint8'))

            if idx >= output_length: break

        except:
            pass

    return

###################################### partition #############################################
def main_partition(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))[:input_length]
    print("input length:\t", len(input_filedirs))
    from data_utils import kdtree_partition
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        if input_filedir.endswith('h5'): points = read_h5_geo(input_filedir)
        if input_filedir.endswith('ply'): points = read_ply_ascii_geo(input_filedir)
        # partition
        kdtree_parts = kdtree_partition(points, max_num=num_points)
        for idx_part, points_part in enumerate(kdtree_parts):
            points_part = points_part - np.min(points_part, axis=0)
            # save
            output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0]+'_'+str(idx_part))
            output_folder, _ = os.path.split(output_filedir)
            os.makedirs(output_folder, exist_ok=True)
            if output_format == 'ply': write_ply_ascii_geo(output_filedir+'.ply', points_part)
            if output_format == 'h5': write_h5_geo(output_filedir+'.h5', points_part)
            count += 1
        if count >= output_length: break
    
    return


def main_partition_with_color(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, num_points):
    print(input_rootdir)

    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))

    print("input length:\t", len(input_filedirs))
    from data_utils import kdtree_partition
    count = 0
    for _, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        if input_filedir.endswith('h5'): coords, feats = read_h5(input_filedir)
        if input_filedir.endswith('ply'): (coords, feats) = read_ply_ascii(input_filedir)
        points = np.concatenate((coords, feats), axis=1)
        # partition
        num_points = 100000
        kdtree_parts = kdtree_partition(points, max_num=num_points)
        for idx_part, points_part in enumerate(kdtree_parts):
            points_part[:,:3] = points_part[:,:3] - np.min(points_part[:,:3], axis=0)
            # save
            ply_output_filedir = output_rootdir + 'ply/100k/' + input_filedir.split('/')[-2].split('.')[0] + '_P' + str(idx_part)
            h5_output_filedir = output_rootdir + 'h5/100k/' + input_filedir.split('/')[-2].split('.')[0] + '_P' + str(idx_part)

            print(ply_output_filedir)
            print(h5_output_filedir)

            write_ply_ascii(ply_output_filedir + '.ply', points_part[:, :3], points_part[:, 3:])
            write_h5(h5_output_filedir + '.h5', points_part[:, :3], points_part[:, 3:])
            count += 1
        if count >= output_length: break

    return

###################################### map color #############################################
from color_augment import get_color_from_image

def main_map_color(input_rootdir, output_rootdir, input_format, output_format, input_length, output_length, image_rootdir):
    input_filedirs = sorted(glob.glob(os.path.join(input_rootdir, '**', f'*'+input_format), recursive=True))
    image_filedirs = sorted(glob.glob(os.path.join(image_rootdir, '**', f'*'+'jpg'), recursive=True))
    import random
    random.shuffle(input_filedirs)
    input_filedirs=input_filedirs[:input_length]
    print("input length:\t", len(input_filedirs))
    print("image_filedirs length:\t", len(image_filedirs))
    for idx, input_filedir in enumerate(tqdm(input_filedirs)):
        # load
        if input_filedir.endswith('h5'): 
            try: coords = read_h5_geo(input_filedir)
            except OSError: continue
        if input_filedir.endswith('ply'): coords = read_ply_ascii_geo(input_filedir)
        # mesh2points
        imgdir = random.sample(image_filedirs, 1)[0]
        try:rgb = get_color_from_image(coords, src_img_file=imgdir)
        except IndexError: continue
        rgb = np.array(rgb, dtype=np.uint8)
        # save
        img_filename = os.path.split(imgdir)[-1].split('.')[0]
        output_filedir = os.path.join(output_rootdir, input_filedir[len(input_rootdir):].split('.')[0]+'_'+img_filename)
        output_folder, _ = os.path.split(output_filedir)
        os.makedirs(output_folder, exist_ok=True)
        if output_format == 'ply': write_ply_ascii(output_filedir+'.ply', coords, rgb)
        if output_format == 'h5': write_h5(output_filedir+'.h5', coords, rgb)
        if idx >= output_length: break

    return 



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--process", default='partition_color')
    parser.add_argument("--input_rootdir", default='/media/disk1/ZJT/data/RWTT/h5/')
    parser.add_argument("--output_rootdir", default='/media/disk1/ZJT/data/RWTT/train_part/')
    parser.add_argument("--image_rootdir", default='')
    parser.add_argument("--input_format", default='h5')
    parser.add_argument("--output_format", default='h5')
    parser.add_argument("--input_length", type=int, default=int(1e6))
    parser.add_argument("--output_length", type=int, default=int(1e6))
    parser.add_argument("--num_points", type=int, default=1e7)
    parser.add_argument("--resolution", type=int, default=1023)
    parser.add_argument("--precision", type=float, default=0.001)
    parser.add_argument("--voxel_size", type=int, default=2)
    args = parser.parse_args()

    # mesh2pc
    if args.process=='mesh2pc':
        main_mesh2pc(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format, 
                    input_length=args.input_length, output_length=args.output_length, 
                    num_points=args.num_points, resolution=args.resolution)

    if args.process=='mesh2pc_color':
        main_mesh2pc_with_color(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir,
                    input_format=args.input_format, output_format=args.output_format,
                    input_length=args.input_length, output_length=args.output_length,
                    num_points=args.num_points, resolution=args.resolution)

    # partition
    if args.process=='partition':
        main_partition(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                    input_format=args.input_format, output_format=args.output_format, 
                    input_length=args.input_length, output_length=args.output_length, 
                    num_points=args.num_points)

    if args.process=='partition_color':
        main_partition_with_color(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir,
                    input_format=args.input_format, output_format=args.output_format,
                    input_length=args.input_length, output_length=args.output_length,
                    num_points=args.num_points)

    # map color
    if args.process=='mapcolor':
        main_map_color(input_rootdir=args.input_rootdir, output_rootdir=args.output_rootdir, 
                input_format=args.input_format, output_format=args.output_format, 
                input_length=args.input_length, output_length=args.output_length, 
                image_rootdir=args.image_rootdir)

