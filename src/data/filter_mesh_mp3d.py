import open3d as o3d
import numpy as np
import trimesh
import multiprocessing as mp
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def compute_close_points(cloud1, cloud2, th):
    """Computes the inlier indices for cloud2.

    Parameters:
        cloud1: input point cloud #1 with N points.
        cloud2: input point cloud #2 with M points.
        th: threshold value s.t. point dists <= th are considered inliers.
    Returns:
        Inlier indices of size V <= M.
    """

    # for ever point in cloud2, compute distance to points in cloud 1
    dists = np.asarray(cloud2.compute_point_cloud_distance(cloud1))
    inliers = np.where(dists <= th)[0]
    return inliers


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q


if __name__ == "__main__":

    dataset = 'MP3D'
    scenes = ["GdvgFV5R1Z5","gZ6f7yhEvPG","HxpKQynjfin","pLe4wQe7qrG","YmJkqBEsHnH"]
    sample_ref_mesh = True
    threshold = 0.1
    for scene in scenes:
        source_mesh_file = f'./data/{dataset}/v1/scans/{scene}/mesh.obj'
        semantic_ply_file = f'./data/{dataset}/v1/tasks/mp3d/{scene}/{scene}_semantic.ply'

        gt_mesh = o3d.io.read_triangle_mesh(source_mesh_file)
        gt_pc = gt_mesh.sample_points_uniformly(number_of_points=50000)
        gt_kdtree = o3d.geometry.KDTreeFlann(gt_pc)

        ply = PlyData.read(semantic_ply_file)
        vertices = ply['vertex'].data
        faces = ply['face'].data

        v_xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)

        keep_mask = np.zeros(len(v_xyz), dtype=bool)
        for i, pt in enumerate(tqdm(v_xyz, desc="Checking vertex proximity")):
            _, _, dist = gt_kdtree.search_knn_vector_3d(pt, 1)
            if np.sqrt(dist[0]) < threshold:
                keep_mask[i] = True

        old_to_new = -np.ones(len(v_xyz), dtype=int)
        old_to_new[keep_mask] = np.arange(np.sum(keep_mask))

        filtered_vertices = vertices[keep_mask]

        valid_faces = []
        for face in faces:
            inds = face[0]  # face["vertex_indices"]
            if all(old_to_new[i] >= 0 for i in inds):
                new_inds = [old_to_new[i] for i in inds]
                new_face = ([tuple(new_inds)],)  # vertex_indices is always first
                # Preserve other face fields if present
                for name in face.dtype.names[1:]:
                    new_face += (face[name],)
                valid_faces.append(new_face)

        face_dtype = faces.dtype
        filtered_faces = np.array(valid_faces, dtype=face_dtype)

        vertex_el = PlyElement.describe(filtered_vertices, 'vertex')
        face_el = PlyElement.describe(filtered_faces, 'face')

        output_ply_file = f'./data/{dataset}/v1/tasks/mp3d/{scene}/semantic_clean.ply'
        PlyData([vertex_el, face_el], text=False).write(output_ply_file)
        print(
            f"Cleaned semantic mesh saved to {output_ply_file} with {len(filtered_vertices)} vertices and {len(filtered_faces)} faces.")
