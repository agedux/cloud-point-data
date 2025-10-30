import open3d as o3d
import numpy as np
import os
from datetime import datetime

def add_defects(pcd, noise_level_percentage):
    """
    Adds simulated defects to a point cloud and returns the noisy cloud 
    along with the indices of the added noise points.
    """
    noisy_pcd = o3d.geometry.PointCloud(pcd)
    points = np.asarray(noisy_pcd.points)
    num_points_clean = len(points)

    # --- Add Outliers/Confounding Points ---
    noise_magnitude = (noise_level_percentage / 100.0) * np.mean(pcd.get_max_bound() - pcd.get_min_bound()) / 5
    num_noisy_points = int(num_points_clean * (noise_level_percentage / 100.0))
    
    noise_indices_gaussian = np.array([], dtype=int)
    if num_noisy_points > 0:
        noisy_indices_gaussian = np.random.choice(num_points_clean, num_noisy_points, replace=False)
        noise = np.random.normal(0, noise_magnitude, (num_noisy_points, 3))
        points[noisy_indices_gaussian] += noise

    num_random_outliers = int(num_points_clean * (noise_level_percentage / 200.0))
    if num_random_outliers > 0:
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        random_outliers = np.random.uniform(min_bound, max_bound, (num_random_outliers, 3))
        noise_indices_random = np.arange(len(points), len(points) + len(random_outliers))
        points = np.vstack((points, random_outliers))
    else:
        noise_indices_random = np.array([], dtype=int)

    ground_truth_noise_indices = np.union1d(noise_indices_gaussian, noise_indices_random)
    
    noisy_pcd.points = o3d.utility.Vector3dVector(points)

    # --- Create Holes ---
    # if (noise_level_percentage / 10) > 0:
    #     for _ in range(int(noise_level_percentage / 10)):
    #         hole_center_index = np.random.randint(0, len(np.asarray(noisy_pcd.points)))
    #         hole_center = np.asarray(noisy_pcd.points)[hole_center_index]
    #         hole_radius = noise_magnitude * (2 + (noise_level_percentage/10.0))
    #         pcd_tree = o3d.geometry.KDTreeFlann(noisy_pcd)
    #         [_, points_to_remove_indices, _] = pcd_tree.search_radius_vector_3d(hole_center, hole_radius)
            
    #         points_to_keep_mask = np.ones(len(noisy_pcd.points), dtype=bool)
    #         points_to_keep_mask[list(points_to_remove_indices)] = False
    #         noisy_pcd = noisy_pcd.select_by_index(np.where(points_to_keep_mask)[0])

    return noisy_pcd, ground_truth_noise_indices

def calculate_chamfer_distance(pcd1, pcd2):
    """
    Calculates and prints the symmetric Chamfer Distance between two point clouds.
    """
    dist_pcd1_to_pcd2 = pcd1.compute_point_cloud_distance(pcd2)
    dist_pcd2_to_pcd1 = pcd2.compute_point_cloud_distance(pcd1)
    
    chamfer_dist = np.mean(dist_pcd1_to_pcd2) + np.mean(dist_pcd2_to_pcd1)
    
    print("--- Quality Metrics (Similarity) ---")
    print(f"    Chamfer Distance to Clean Cloud: {chamfer_dist:.4f}")
    print("------------------------------------")

def calculate_denoising_metrics(total_points_in_noisy_pcd, ground_truth_noise_indices, denoised_inlier_indices):
    """
    Calculates and prints denoising metrics (Precision, Recall, F1-score)
    based on the provided ground truth and denoising results, as per the PDF.
    """
    all_indices = set(range(total_points_in_noisy_pcd))
    ground_truth_noise_set = set(ground_truth_noise_indices)
    # Clean points are all points that are not noise
    ground_truth_clean_set = all_indices - ground_truth_noise_set

    removed_indices = all_indices - set(denoised_inlier_indices)

    # True Positives: Actual noise points that were correctly removed.
    tp = len(removed_indices.intersection(ground_truth_noise_set))
    
    # False Positives: Clean points that were incorrectly removed.
    fp = len(removed_indices.intersection(ground_truth_clean_set))

    # False Negatives: Noise points that were not removed.
    fn = len(ground_truth_noise_set) - tp

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("--- Denoising Metrics (based on PDF) ---")
    print(f"    Precision (Pd): {precision:.4f}")
    print(f"    Recall (Rd):    {recall:.4f}")
    print(f"    F1-score:       {f1_score:.4f}")
    print("----------------------------------------")

def _get_octree_density_map(pcd, octree_max_depth):
    octree = o3d.geometry.Octree(max_depth=octree_max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    points = np.asarray(pcd.points)
    # Dictionary to store point indices for each leaf node, identified by a hashable key
    leaf_node_key_to_point_indices = {}

    for i, point in enumerate(points):
        # locate_leaf_node returns the leaf node and its info
        leaf_node, info = octree.locate_leaf_node(point)
        if leaf_node:
            # Use a tuple of (origin_x, origin_y, origin_z, extent) as a hashable key for the leaf node
            node_key = (info.origin[0], info.origin[1], info.origin[2], info.size)
            if node_key not in leaf_node_key_to_point_indices:
                leaf_node_key_to_point_indices[node_key] = []
            leaf_node_key_to_point_indices[node_key].append(i)

    # Calculate densities for each point
    point_densities = np.zeros(len(points))
    for node_key, indices in leaf_node_key_to_point_indices.items():
        density = len(indices)
        for idx in indices:
            point_densities[idx] = density
    
    # Handle points that might not have been assigned to any leaf node (e.g., if extent was too small initially)
    # Assign a default density of 1 for such points, or consider them as outliers if appropriate.
    # For now, we'll assign 1 to avoid division by zero.
    point_densities[point_densities == 0] = 1

    return point_densities

def remove_outliers(pcd, octree_max_depth=8, k=10, threshold_multiplier=1.0):
    if len(pcd.points) < k + 1:
        return pcd, np.arange(len(pcd.points))
    
    points = np.asarray(pcd.points)
    
    # Get octree-based densities for each point
    point_densities = _get_octree_density_map(pcd, octree_max_depth)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    deltas = np.zeros(len(points))
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k + 1)
        neighbors = points[idx[1:]]
        avg_distance = np.mean(np.linalg.norm(points[i] - neighbors, axis=1))
        
        density = point_densities[i]
        
        if density > 0:
            deltas[i] = avg_distance / density
        else:
            deltas[i] = float('inf')
    if len(deltas) == 0: return pcd, np.arange(len(pcd.points))
    delta_threshold = np.percentile(deltas, 90) * threshold_multiplier
    inlier_indices = np.where(deltas < delta_threshold)[0]
    return pcd.select_by_index(inlier_indices), inlier_indices

def remove_confounding_points(pcd, k=20):
    if len(pcd.points) < k:
        return pcd
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_points = np.copy(points)
    for i in range(len(points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
        neighbors = points[idx]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov(neighbors.T)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        projection = points[i] - np.dot(points[i] - centroid, normal) * normal
        new_points[i] = projection
    pcd_projected = o3d.geometry.PointCloud()
    pcd_projected.points = o3d.utility.Vector3dVector(new_points)
    return pcd_projected

# def fill_holes(pcd, alpha_multiplier=4.0, max_area_threshold_factor=3.0, points_per_iteration=50):
#     pcd_filled = o3d.geometry.PointCloud(pcd)
#     avg_dist = np.mean(pcd_filled.compute_nearest_neighbor_distance())
#     if avg_dist == 0:
#         print("    Average distance is zero. Cannot fill holes.")
#         return pcd_filled
#     alpha = avg_dist * alpha_multiplier
#     max_iterations = 200
#     for i in range(max_iterations):
#         print(f"  Hole filling iteration {i+1}...")
#         try:
#             mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_filled, alpha)
#             mesh.compute_vertex_normals()
#         except Exception as e:
#             print(f"    Could not create mesh: {e}. Stopping.")
#             break
#         triangles = np.asarray(mesh.triangles)
#         vertices = np.asarray(mesh.vertices)
#         if len(triangles) == 0:
#             print("    No triangles in mesh. Stopping.")
#             break
#         triangle_areas = np.zeros(len(triangles))
#         for j, tri_indices in enumerate(triangles):
#             v0, v1, v2 = vertices[tri_indices[0]], vertices[tri_indices[1]], vertices[tri_indices[2]]
#             triangle_areas[j] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
#         avg_area = np.mean(triangle_areas)
#         max_area_threshold = avg_area * max_area_threshold_factor
#         sorted_triangle_indices = np.argsort(triangle_areas)[::-1]
#         new_centroids = []
#         num_added = 0
#         for tri_index in sorted_triangle_indices[:points_per_iteration]:
#             if triangle_areas[tri_index] < max_area_threshold:
#                 break
#             verts_of_largest_triangle = vertices[triangles[tri_index]]
#             centroid = np.mean(verts_of_largest_triangle, axis=0)
#             new_centroids.append(centroid)
#             num_added += 1
#         if num_added == 0:
#             print("    No significant holes found to fill in this iteration. Stopping.")
#             break
#         current_points = np.asarray(pcd_filled.points)
#         new_points_start_index = len(current_points)
#         pcd_filled.points = o3d.utility.Vector3dVector(np.vstack((current_points, np.array(new_centroids))))
#         print(f"    Inserted {num_added} new points. Refining their positions...")
        
#         # --- Refinement Step ---
#         pcd_filled_tree = o3d.geometry.KDTreeFlann(pcd_filled)
#         points_to_refine_indices = range(new_points_start_index, len(pcd_filled.points))
#         updated_points = np.asarray(pcd_filled.points)

#         for pt_idx in points_to_refine_indices:
#             point_to_refine = updated_points[pt_idx]
#             [_, idx, _] = pcd_filled_tree.search_knn_vector_3d(point_to_refine, 20)
#             neighbors = updated_points[idx]
            
#             centroid = np.mean(neighbors, axis=0)
#             cov_matrix = np.cov(neighbors.T)
#             if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
#                 continue
#             eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#             normal = eigenvectors[:, np.argmin(eigenvalues)]
            
#             projection = point_to_refine - np.dot(point_to_refine - centroid, normal) * normal
#             updated_points[pt_idx] = projection
#         pcd_filled.points = o3d.utility.Vector3dVector(updated_points)
#         # --- End of Refinement Step ---

#     else:
#         print(f"    Reached max iterations ({max_iterations}). Stopping.")
#     return pcd_filled

def main():
    # --- Setup ---
    output_dir = "output"
    # The user created the directory manually, but this check makes the script more robust.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # data_name = "bunny" # Used for filenames

    # # --- Load clean bunny point cloud ---
    # try:
    #     bunny_mesh = o3d.data.BunnyMesh()
    #     pcd_clean = o3d.io.read_point_cloud(bunny_mesh.path)
    # except Exception as e:
    #     print(f"Could not load BunnyMesh: {e}")
    #     mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    #     pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)

    data_name = "armadillo" # Used for filenames
    
    # --- Load clean armadillo point cloud ---
    try:
        armadillo_mesh = o3d.data.ArmadilloMesh()
        pcd_clean = o3d.io.read_point_cloud(armadillo_mesh.path)
    except Exception as e:
        print(f"Could not load ArmadilloMesh: {e}")
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)

    pcd_clean.paint_uniform_color([0.5, 0.5, 0.5])
    print("Visualizing original clean point cloud...")
        
    o3d.visualization.draw_geometries([pcd_clean], window_name="Original Clean Point Cloud")

    noise_levels = [30] # Use one level for focused interactive tuning

    for noise_level in noise_levels:
        print(f"\n========== Processing with {noise_level}% noise ========== ")
        
        pcd_clean_copy = o3d.geometry.PointCloud(pcd_clean)

        # --- Add Defects ---
        pcd_noisy, noise_indices = add_defects(pcd_clean_copy, noise_level)
        pcd_noisy.paint_uniform_color([1, 0, 0])
        print(f"  Points after adding defects: {len(pcd_noisy.points)} (contains {len(noise_indices)} noise points)")
        
        # Save noisy point cloud
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_noisy_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_noisy)
        print(f"  Saved: {filename}")
        
        o3d.visualization.draw_geometries([pcd_noisy], window_name=f"{noise_level}% Noisy Point Cloud")
        
        # --- Remove Outliers ---
        print("1. Removing outliers...")
        pcd_outliers_removed, inlier_indices_denoised = remove_outliers(pcd_noisy, octree_max_depth=10, k=15, threshold_multiplier=0.5)
        
        # --- Calculate Denoising Metrics ---
        calculate_denoising_metrics(len(pcd_noisy.points), noise_indices, inlier_indices_denoised)
        pcd_outliers_removed.paint_uniform_color([0, 1, 0])
        
        # Save outliers removed point cloud
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_outliers_removed_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_outliers_removed)
        print(f"  Saved: {filename}")

        o3d.visualization.draw_geometries([pcd_outliers_removed], window_name="1. Outliers Removed")

        # --- Remove Confounding Points ---
        print("2. Removing confounding points (Plane Projection)...")
        pcd_smoothed = pcd_outliers_removed
        smoothing_iterations = 3
        smoothing_k = 30
        for i in range(smoothing_iterations):
            print(f"  Smoothing iteration {i+1}/{smoothing_iterations}...")
            pcd_confounding_removed = remove_confounding_points(pcd_smoothed, k=smoothing_k)


        # --- [新增步驟] 使用 Taubin 濾波器進行密度均化與最終平滑 ---
        # print("2.5 Applying Taubin filter for final smoothing and resampling...")
        # pcd_confounding_removed = pcd_smoothed.filter_smooth_taubin(number_of_iterations=10)
        pcd_confounding_removed.paint_uniform_color([0, 0, 1])
        print(f"  Points after all smoothing steps: {len(pcd_confounding_removed.points)}")

        # Save confounding points removed point cloud
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_confounding_removed_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_confounding_removed)
        print(f"  Saved: {filename}")

        o3d.visualization.draw_geometries([pcd_confounding_removed], window_name="2. Confounding Points Removed")

        # --- Fill Holes ---
        # print("3. Filling holes...")
        # pcd_holes_filled = fill_holes(pcd_confounding_removed, alpha_multiplier=4.0, max_area_threshold_factor=3.0, points_per_iteration=50)
        # pcd_holes_filled.paint_uniform_color([1, 1, 0])
        # print(f"  Points after hole filling: {len(pcd_holes_filled.points)}")

        # # Save holes filled point cloud
        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # filename = os.path.join(output_dir, f"{data_name}_holes_filled_{timestamp}.ply")
        # o3d.io.write_point_cloud(filename, pcd_holes_filled)
        # print(f"  Saved: {filename}")

        # o3d.visualization.draw_geometries([pcd_holes_filled], window_name="3. Holes Filled")

        # --- Final Evaluation ---
        print(f"\n--- Final Evaluation for {noise_level}% noise ---")
        calculate_chamfer_distance(pcd_confounding_removed, pcd_clean_copy)

        print(f"Displaying final comparison for {noise_level}% noise:")
        pcd_clean_copy.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_noisy.paint_uniform_color([1, 0, 0])
        pcd_confounding_removed.paint_uniform_color([0, 1, 0]) # Final result in Green
        
        pcd_noisy.translate((0.2, 0, 0))
        pcd_confounding_removed.translate((0.4, 0, 0))

        o3d.visualization.draw_geometries([pcd_clean_copy, pcd_noisy, pcd_confounding_removed],
                                          window_name=f"Final Comparison (L-R): Clean, Noisy, Processed")

if __name__ == "__main__":
    main()
