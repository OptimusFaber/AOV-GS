import torch
import heapq
import numpy as np

def generate_start_end_points(occ_grid: torch.Tensor, vxl_size: float = 0.05):
    """
    Generates a start point and end point in free space based on the occupancy grid.

    Args:
        occ_grid: [D, H, W], with voxel size V
            - 1: occupied
            - 0: unexplored
            - -1: free region type 1
            - -2: free region type 2
        vxl_size: voxel size (in meters)

    Returns:
        start_point: [3], coordinate in physical space
        end_point: [3], coordinate in physical space
    """
    D, H, W = occ_grid.shape
    # Pad the occupancy grid to handle edge cases
    # padded_occ_grid = torch.zeros(D + 2, H + 2, W + 2, dtype=occ_grid.dtype, device=occ_grid.device)
    # padded_occ_grid[:-2, :-2, :-2] = occ_grid

    # Define free space where voxels are -1 or -2
    # free_mask = (padded_occ_grid == -1) | (padded_occ_grid == -2)
    free_mask = (occ_grid == -1) | (occ_grid == -2)
    free_mask = free_mask.long()
    # A cube is free if all its eight corners are free
    free_voxels = (
        free_mask[:-1, :-1, :-1] +
        free_mask[1:, :-1, :-1] +
        free_mask[:-1, 1:, :-1] +
        free_mask[:-1, :-1, 1:] +
        free_mask[1:, 1:, :-1] +
        free_mask[1:, :-1, 1:] +
        free_mask[:-1, 1:, 1:] +
        free_mask[1:, 1:, 1:]
    )

    free_voxels = (free_voxels>0)

    # Find indices of free voxels
    free_indices = torch.nonzero(free_voxels)

    if free_indices.shape[0] < 2:
        raise ValueError("Not enough free space to generate start and end points.")
    while True:
        # Randomly select two different indices
        idx = torch.randperm(free_indices.shape[0])[:2]
        start_idx = free_indices[idx[0]]
        end_idx = free_indices[idx[1]]

        # Convert voxel indices to physical coordinates (center of the voxel)
        start_point = (start_idx.float() + torch.rand(3)) * vxl_size
        end_point = (end_idx.float() + torch.rand(3)) * vxl_size

        if not is_line_free(start_point, end_point, free_voxels, vxl_size, vxl_size/10):    
            return start_point, end_point

def save_3d_map_with_path(occ_grid, start_loc, end_loc, path, filename='map_with_path.ply', vxl_size=0.05):
    """
    Save a 3D map with the path as a .ply file.

    Args:
        occ_grid: torch.Tensor of shape [D, H, W], occupancy grid.
            - 1: occupied
            - 0: unexplored
            - -1: free region type 1
            - -2: free region type 2
        start_loc: torch.Tensor of shape [3], starting location in voxel indices.
        end_loc: torch.Tensor of shape [3], ending location in voxel indices.
        path: torch.Tensor of shape [N, 3], path through the grid in voxel indices.
        filename: str, output filename for the .ply file.
        vxl_size: float, size of each voxel in meters.

    Returns:
        None. Saves the 3D map with the path as a .ply file.
    """
    import open3d as o3d

    free_mask = (occ_grid == -1) | (occ_grid == -2)
    free_mask = free_mask.long()
    # A cube is free if all its eight corners are free
    free_voxels = (
        free_mask[:-1, :-1, :-1] +
        free_mask[1:, :-1, :-1] +
        free_mask[:-1, 1:, :-1] +
        free_mask[:-1, :-1, 1:] +
        free_mask[1:, 1:, :-1] +
        free_mask[1:, :-1, 1:] +
        free_mask[:-1, 1:, 1:] +
        free_mask[1:, 1:, 1:]
    )
    
    D, H, W = free_voxels.shape
    occupied_mask = (free_voxels<1).flatten().numpy()
    # Ensure input is on CPU and in numpy format
    start_loc_np = start_loc.cpu().numpy()
    end_loc_np = end_loc.cpu().numpy()
    path_np = path#.cpu().numpy()

    # Create a meshgrid of voxel indices
    zz, yy, xx = np.meshgrid(
        np.arange(D),
        np.arange(H),
        np.arange(W),
        indexing='ij'
    )
    voxel_indices = np.vstack((zz.flatten(), yy.flatten(), xx.flatten())).T
    occupied_voxels = voxel_indices[occupied_mask]

    # Convert voxel indices to coordinates (multiply by voxel size)
    occupied_points = occupied_voxels * vxl_size

    # Process the path, start, and end locations
    path_points = path_np #* vxl_size
    start_point = start_loc_np #* vxl_size
    end_point = end_loc_np #* vxl_size

    # Combine all points
    all_points = np.vstack((occupied_points, path_points, start_point, end_point))

    # Assign colors
    num_occupied = occupied_points.shape[0]
    num_path = path_points.shape[0]

    colors = np.zeros((all_points.shape[0], 3), dtype=np.float32)

    # Occupied voxels: gray
    colors[:num_occupied] = [0.5, 0.5, 0.5]
    # Path: red
    colors[num_occupied:num_occupied + num_path] = [1.0, 0.0, 0.0]
    # Start point: green
    colors[num_occupied + num_path] = [0.0, 1.0, 0.0]
    # End point: blue
    colors[num_occupied + num_path + 1] = [0.0, 0.0, 1.0]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save to .ply file
    o3d.io.write_point_cloud(filename, pcd)

def interpolate_path(A: torch.Tensor, B: torch.Tensor, step: float):
    """
    Generates a list of intermediate points between two locations, A and B, with a given step size,
    excluding A and including B in the resulting list.

    Args:
        A (torch.Tensor): The starting location (1D tensor representing a point in n-dimensional space).
        B (torch.Tensor): The ending location (1D tensor representing a point in n-dimensional space).
        step (float): The step size between consecutive points.

    Returns:
        List[torch.Tensor]: A list of tensors representing intermediate points from A to B, excluding A and including B.
    
    Raises:
        ValueError: If A and B have different shapes or step size is non-positive.
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")
    if step <= 0:
        raise ValueError("Step size must be a positive value.")

    ### Calculate the total distance between A and B ###
    distance = torch.norm(B - A)

    ### Calculate the number of intermediate steps needed ###
    num_steps = int(distance / step)

    ### Generate the intermediate points, excluding A and including B ###
    points = [A + (B - A) * ((i + 1) * step / distance) for i in range(num_steps)]

    ### Append B explicitly to ensure it is included ###
    points.append(B)
    # points = torch.stack(points)
    return points

def interpolate_path_seq(path_positions, step_times):
    breakpoint()
    # Ensure step_times is a tensor and matches the device
    # if isinstance(step_times, int):
    step_times = torch.tensor([step_times] * (len(path_positions) - 1), device=path_positions.device)
    
    # Total number of interpolation points (sum of step_times) plus original points
    total_steps = step_times.sum() + len(path_positions)
    
    # Calculate cumulative indices for placing original points
    cum_steps = torch.cat([torch.tensor([0], device=path_positions.device), step_times.cumsum(0)])
    
    # Initialize tensor to hold interpolated path
    interpolated_positions = torch.zeros(total_steps, 3, device=path_positions.device)
    
    # Place original points in the interpolated tensor at cumulative indices
    interpolated_positions[cum_steps] = path_positions
    
    # Calculate weights for each segment in a fully vectorized way
    segments = path_positions[1:] - path_positions[:-1]
    weights = torch.cat([torch.linspace(0, 1, steps.item() + 2, device=path_positions.device)[1:-1]
                         for steps in step_times])
    
    # Broadcast and compute interpolated positions
    weights = weights.unsqueeze(1)  # Shape [total_interpolations, 1] for broadcasting
    interpolated_points = (path_positions[:-1].repeat_interleave(step_times) +
                           weights * segments.repeat_interleave(step_times))
    
    # Place interpolated points in their positions in the interpolated path tensor
    non_original_indices = torch.arange(total_steps, device=path_positions.device).isin(cum_steps).logical_not()
    interpolated_positions[non_original_indices] = interpolated_points
    
    return interpolated_positions

def is_line_free(start_loc, end_loc, free_space, voxel_size, step_size):
    # Compute the distance between start and end
    dist = torch.norm(end_loc - start_loc)
    num_steps = int(dist / step_size) + 1
    # Generate points along the line
    t_values = torch.linspace(0, 1, steps=num_steps)
    points = start_loc[None, :] + t_values[:, None] * (end_loc - start_loc)[None, :]
    # Map points to voxel indices
    indices = (points / voxel_size).long()
    # Ensure indices are within the bounds of the free_space grid
    D, H, W = free_space.shape

    indices = torch.clamp(indices, min=torch.tensor([0, 0, 0]), max=torch.tensor([D - 1, H - 1, W - 1]))
    # Remove duplicate indices
    indices = indices.unique(dim=0)
    # Check if all the indices are in free space
    for idx in indices:
        idx_tuple = tuple(idx.tolist())
        if not free_space[idx_tuple]:
            return False
    return True

def path_planning(occ_grid: torch.Tensor, start_loc: torch.Tensor, end_loc: torch.Tensor, max_trans: float = 0.1,step_times = 1.0, vxl_size: float = 0.05) -> torch.Tensor:
    """ path planning using occupancy grid, only plan based on free regions (including both type1 and type2)
 
    Args:
        occ_grid: [D,H,W], with voxel size V
            - 1: occupied
            - 0: unexplored
            - -1: free region type 1
            - -2: free region type 2
        start_loc: [3] assumed to be in free region
        end_loc: [3] assumed to be in free region
        max_trans: maximum translation step (in meter)
        vxl_size: voxel size (in meter)
        
    Returns:
        path: [N,3], planned path in voxel space
    """
    # Convert start_loc and end_loc to voxel indices
    start_idx = (start_loc / vxl_size).long()
    end_idx = (end_loc / vxl_size).long()

    # Pad the occupancy grid to handle edge cases
    # D, H, W = occ_grid.shape
    # padded_occ_grid = torch.zeros(D + 1, H + 1, W + 1, dtype=occ_grid.dtype, device=occ_grid.device)
    # padded_occ_grid[:-1, :-1, :-1] = occ_grid

    # Define free space where voxels are -1 or -2
    free_mask = (occ_grid == -1) | (occ_grid == -2)
    free_mask = free_mask.long()
    # Compute free_space
    free_space = (
        free_mask[:-1, :-1, :-1] +
        free_mask[1:, :-1, :-1] +
        free_mask[:-1, 1:, :-1] +
        free_mask[:-1, :-1, 1:] +
        free_mask[1:, 1:, :-1] +
        free_mask[1:, :-1, 1:] +
        free_mask[:-1, 1:, 1:] +
        free_mask[1:, 1:, 1:]
    )
    free_space = (free_space>0)
    # Check if start_idx and end_idx are in free_space
    if not free_space[start_idx[0], start_idx[1], start_idx[2]]:
        raise ValueError('Start location is not in free space')
    if not free_space[end_idx[0], end_idx[1], end_idx[2]]:
        raise ValueError('End location is not in free space')
    
    #proceed with A* algorithm
    # path_positions = astar(occ_grid, free_space, start_loc, end_loc, max_trans, vxl_size)
    path_positions = astar(free_space, start_loc, end_loc, max_trans*step_times, vxl_size)

    # path_positions = interpolate_path_seq(path_positions, step_times)
    return path_positions

def astar(free_space, start_loc, end_loc, max_trans, voxel_size):
    """
    A* path planning in continuous space without quantization.

    Args:
        free_space: 3D occupancy grid (torch.Tensor of bools), True if free
        start_loc: torch.Tensor of shape [3], starting position in meters
        end_loc: torch.Tensor of shape [3], goal position in meters
        max_trans: float, maximum allowed movement per step (meters)
        voxel_size: float, size of each voxel (meters)

    Returns:
        path: torch.Tensor of shape [N, 3], planned path in continuous space
    """
    import heapq
    # Initialize open list and closed set for start
    open_list = []
    start_key = tuple(start_loc.tolist())
    heapq.heappush(open_list, (0.0, start_key))
    closed_set = set()
    # Store g values and parents
    g_score = {}
    g_score[start_key] = 0.0
    parents = {}

    # Initialize open list and closed set for end
    open_list_end = []
    end_key = tuple(end_loc.tolist())
    heapq.heappush(open_list_end, (0.0, end_key))
    closed_set_end = set()
    # Store g values and parents
    g_score_end = {}
    g_score_end[end_key] = 0.0
    parents_end = {}

    # Define heuristic function (Euclidean distance)
    def heuristic(pos_key, pos_key_end):
        pos_tensor = torch.tensor(pos_key)
        pos_end_tensor = torch.tensor(pos_key_end)
        return torch.norm(pos_end_tensor - pos_tensor)

    # Generate possible movement directions
    directions = []
    for dx in [-1, -0.5, 0, 0.5, 1]:
        for dy in [-1, -0.5, 0, 0.5, 1]:
            for dz in [-1, -0.5, 0, 0.5, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                direction = torch.tensor([dx, dy, dz], dtype=torch.float32)
                direction_norm = torch.norm(direction)
                if direction_norm == 0:
                    continue
                unit_direction = direction / direction_norm
                directions.append(unit_direction)

    # # While open list is not empty
    num_step = 0
    while open_list or open_list_end:
        _, current_key = heapq.heappop(open_list)
        _, current_key_end = heapq.heappop(open_list_end)
        # print(current_key)
        current_pos = torch.tensor(current_key)
        current_pos_end = torch.tensor(current_key_end)

        # Check if we have reached the goal
        if is_line_free(current_pos, current_pos_end, free_space, voxel_size, voxel_size/10):#or num_step > 1000
            # print("number of steps: ", str(num_step))
            # interplate path
            # if num_step > 1000:
            #     path = [end_loc]
            # else:
            path = interpolate_path(current_pos, current_pos_end, max_trans)
            path#.reverse()

            pos_key = current_key
            pos_key_end = current_key_end
            while True:
                pos_end = torch.tensor(pos_key_end)
                path.append(pos_end)
                if pos_key_end in parents_end:
                    pos_key_end = parents_end[pos_key_end]
                else:
                    break
            path.reverse()
            while True:
                pos = torch.tensor(pos_key)
                path.append(pos)
                if pos_key in parents:
                    pos_key = parents[pos_key]
                else:
                    break
            path.reverse()

            path = torch.stack(path, dim=0)
            return path
        
        if current_key not in closed_set:
            closed_set.add(current_key)
            # For each direction, generate a neighbor position
            for direction in directions:
                neighbor_pos = current_pos + max_trans * direction
                neighbor_key = tuple(neighbor_pos.tolist())

                if neighbor_key in closed_set:
                    continue

                # Check if path from current_pos to neighbor_pos is free
                if not is_line_free(current_pos, neighbor_pos, free_space, voxel_size, voxel_size/10):
                    continue

                # Compute tentative g_score
                tentative_g = g_score[current_key] + max_trans

                if neighbor_key not in g_score or tentative_g < g_score[neighbor_key]:
                    g_score[neighbor_key] = tentative_g
                    parents[neighbor_key] = current_key
                    f_score = tentative_g + heuristic(neighbor_key, current_key_end)
                    heapq.heappush(open_list, (f_score.item(), neighbor_key))

        if current_key_end not in closed_set_end:
            closed_set_end.add(current_key_end)
            # For each direction, generate a neighbor position
            for direction in directions:
                neighbor_pos_end = current_pos_end + max_trans * direction
                neighbor_key_end = tuple(neighbor_pos_end.tolist())

                if neighbor_key_end in closed_set_end:
                    continue

                # Check if path from current_pos to neighbor_pos is free
                if not is_line_free(current_pos_end, neighbor_pos_end, free_space, voxel_size, voxel_size/10):
                    continue

                # Compute tentative g_score
                tentative_g_end = g_score_end[current_key_end] + max_trans

                if neighbor_key_end not in g_score_end or tentative_g_end < g_score_end[neighbor_key_end]:
                    g_score_end[neighbor_key_end] = tentative_g_end
                    parents_end[neighbor_key_end] = current_key_end
                    f_score_end = tentative_g_end + heuristic(neighbor_key_end, current_key)
                    heapq.heappush(open_list_end, (f_score_end.item(), neighbor_key_end))

        num_step = num_step + 1
        
    raise ValueError('No path found')