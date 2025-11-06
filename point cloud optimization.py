import open3d as o3d
import numpy as np
import os
from datetime import datetime
from scipy.spatial import cKDTree

def add_defects(pcd, noise_level_percentage, num_holes=5):
    """
    在點雲上模擬並添加瑕疵，包括噪點、離群點和孔洞。

    Args:
        pcd (open3d.geometry.PointCloud): 原始的乾淨點雲。
        noise_level_percentage (float): 噪點等級百分比，用於控制噪點的數量與強度。
        num_holes (int): 要在點雲上製造的孔洞數量。

    Returns:
        tuple:
            - open3d.geometry.PointCloud: 含有瑕疵的新點雲。
            - numpy.ndarray: 所有被視為「真實噪點」的點的索引，用於後續評估去噪演算法的性能。
    """
    noisy_pcd = o3d.geometry.PointCloud(pcd)
    points = np.asarray(noisy_pcd.points)
    num_points_clean = len(points)

    # --- 步驟 1: 添加噪點 (高斯噪點與隨機離群點) ---
    # 根據點雲的邊界框大小計算噪點的強度
    noise_magnitude = (noise_level_percentage / 100.0) * np.mean(pcd.get_max_bound() - pcd.get_min_bound()) / 5
    # 根據噪點比例計算要添加高斯噪點的點數
    num_noisy_points = int(num_points_clean * (noise_level_percentage / 100.0))
    
    noise_indices_gaussian = np.array([], dtype=int)
    if num_noisy_points > 0:
        # 隨機選取點，並加上符合高斯分佈的位移，模擬緊貼表面的「混淆點」
        noisy_indices_gaussian = np.random.choice(num_points_clean, num_noisy_points, replace=False)
        noise = np.random.normal(0, noise_magnitude, (num_noisy_points, 3))
        points[noisy_indices_gaussian] += noise

    # 添加完全隨機的「離群點」
    num_random_outliers = int(num_points_clean * (noise_level_percentage / 200.0))
    if num_random_outliers > 0:
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        # 在點雲的邊界框內生成均勻分佈的隨機點
        random_outliers = np.random.uniform(min_bound, max_bound, (num_random_outliers, 3))
        noise_indices_random = np.arange(len(points), len(points) + len(random_outliers))
        points = np.vstack((points, random_outliers))
    else:
        noise_indices_random = np.array([], dtype=int)

    # 合併所有噪點的索引，作為後續評估的「真實答案」
    ground_truth_noise_indices = np.union1d(noise_indices_gaussian, noise_indices_random)
    
    noisy_pcd.points = o3d.utility.Vector3dVector(points)

    # --- 步驟 2: 製造孔洞 ---
    if num_holes > 0:
        for _ in range(num_holes):
            # 隨機選擇一個點作為孔洞中心
            hole_center_index = np.random.randint(0, len(np.asarray(noisy_pcd.points)))
            hole_center = np.asarray(noisy_pcd.points)[hole_center_index]
            # 根據噪點強度定義孔洞半徑
            hole_radius = noise_magnitude * 3.0 
            pcd_tree = o3d.geometry.KDTreeFlann(noisy_pcd)
            # 搜尋孔洞半徑內的所有點
            [_, points_to_remove_indices, _] = pcd_tree.search_radius_vector_3d(hole_center, hole_radius)
            
            # 創建一個遮罩，標記要被移除的點
            points_to_keep_mask = np.ones(len(noisy_pcd.points), dtype=bool)
            points_to_keep_mask[list(points_to_remove_indices)] = False
            # 根據遮罩選取要保留的點，從而移除孔洞區域的點
            noisy_pcd = noisy_pcd.select_by_index(np.where(points_to_keep_mask)[0])

    return noisy_pcd, ground_truth_noise_indices

def calculate_chamfer_distance(pcd1, pcd2):
    """
    計算並印出兩個點雲之間的對稱倒角距離 (Symmetric Chamfer Distance)。
    此指標用於評估兩個點雲的相似度，值越小代表越相似。
    """
    # 計算 pcd1 中每個點到 pcd2 的最近距離
    dist_pcd1_to_pcd2 = pcd1.compute_point_cloud_distance(pcd2)
    # 計算 pcd2 中每個點到 pcd1 的最近距離
    dist_pcd2_to_pcd1 = pcd2.compute_point_cloud_distance(pcd1)
    
    # 將兩個方向的平均距離相加，得到對稱倒角距離
    chamfer_dist = np.mean(dist_pcd1_to_pcd2) + np.mean(dist_pcd2_to_pcd1)
    
    print("--- 品質評估 (相似度) ---")
    print(f"    與乾淨點雲的倒角距離: {chamfer_dist:.4f}")
    print("------------------------------------")

def calculate_denoising_metrics(total_points_in_noisy_pcd, ground_truth_noise_indices, denoised_inlier_indices):
    """
    根據參考文件的定義，計算並印出去噪演算法的性能指標 (Precision, Recall, F1-score)。

    Args:
        total_points_in_noisy_pcd (int): 帶噪點雲的總點數。
        ground_truth_noise_indices (set): 真實噪點的索引集合。
        denoised_inlier_indices (set): 經過演算法處理後，被判定為「內點」(Inlier) 的索引集合。
    """
    all_indices = set(range(total_points_in_noisy_pcd))
    ground_truth_noise_set = set(ground_truth_noise_indices)
    # 真實的乾淨點 = 所有點 - 真實的噪點
    ground_truth_clean_set = all_indices - ground_truth_noise_set

    # 被演算法移除的點 = 所有點 - 演算法判定的內點
    removed_indices = all_indices - set(denoised_inlier_indices)

    # 真陽性 (TP): 被正確移除的噪點 (演算法移除的點 與 真實噪點 的交集)
    tp = len(removed_indices.intersection(ground_truth_noise_set))
    
    # 偽陽性 (FP): 被錯誤移除的乾淨點 (演算法移除的點 與 真實乾淨點 的交集)
    fp = len(removed_indices.intersection(ground_truth_clean_set))

    # 偽陰性 (FN): 未被移除的噪點 (真實噪點的總數 - 被正確移除的噪點)
    fn = len(ground_truth_noise_set) - tp

    # 根據公式計算指標
    # 精確率 (Precision, Pd): 在所有被移除的點中，有多少比例是真正的噪點
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # 召回率 (Recall, Rd): 在所有真正的噪點中，有多少比例被成功移除了
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1-Score: 精確率和召回率的調和平均數，是綜合評價指標
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("--- 去噪性能指標 (根據參考文件) ---")
    print(f"    精確率 (Precision, Pd): {precision:.4f}")
    print(f"    召回率 (Recall, Rd):    {recall:.4f}")
    print(f"    F1-score:               {f1_score:.4f}")
    print("----------------------------------------")

def _get_octree_density_map(pcd, octree_max_depth):
    """
    [輔助函式] 使用八叉樹 (Octree) 結構計算點雲中每個點的局部密度。
    密度被定義為該點所在的最小八叉樹立方體內的點的數量。
    """
    # 建立八叉樹並從點雲進行轉換
    octree = o3d.geometry.Octree(max_depth=octree_max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    points = np.asarray(pcd.points)
    # 建立一個字典，用於儲存每個八叉樹葉節點中的點的索引
    leaf_node_key_to_point_indices = {}

    # 遍歷所有點，將它們分配到對應的葉節點中
    for i, point in enumerate(points):
        # 找到點所在的葉節點
        leaf_node, info = octree.locate_leaf_node(point)
        if leaf_node:
            # 使用葉節點的 (原點, 大小) 作為獨一無二的鍵
            node_key = (info.origin[0], info.origin[1], info.origin[2], info.size)
            if node_key not in leaf_node_key_to_point_indices:
                leaf_node_key_to_point_indices[node_key] = []
            leaf_node_key_to_point_indices[node_key].append(i)

    # 根據每個葉節點的點數計算密度，並賦值給對應的點
    point_densities = np.zeros(len(points))
    for node_key, indices in leaf_node_key_to_point_indices.items():
        density = len(indices)
        for idx in indices:
            point_densities[idx] = density
    
    # 處理可能未被分配到任何葉節點的點，給予預設密度 1，以避免除以零的錯誤
    point_densities[point_densities == 0] = 1

    return point_densities

def remove_outliers(pcd, octree_max_depth=8, k=10, threshold_multiplier=1.0):
    """
    移除離群點。此函式實現了參考文件中結合「八叉樹密度」與「統計分析」的演算法。
    """
    if len(pcd.points) < k + 1:
        return pcd, np.arange(len(pcd.points))
    
    points = np.asarray(pcd.points)
    
    # 步驟 1: 取得每個點的八叉樹局部密度 (pn)
    point_densities = _get_octree_density_map(pcd, octree_max_depth)

    # 步驟 2: 計算每個點與其 k 個鄰居的平均距離
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    deltas = np.zeros(len(points))
    for i in range(len(points)):
        # 搜尋 k+1 個鄰居 (包含點本身)
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k + 1)
        neighbors = points[idx[1:]] # 排除點本身
        avg_distance = np.mean(np.linalg.norm(points[i] - neighbors, axis=1))
        
        density = point_densities[i]
        
        # 步驟 3: 根據參考文件公式計算離群機率 delta
        # delta = 平均鄰居距離 / 局部密度
        if density > 0:
            deltas[i] = avg_distance / density
        else:
            # 如果密度為0，視為無限大的離群機率
            deltas[i] = float('inf')
            
    if len(deltas) == 0: return pcd, np.arange(len(pcd.points))

    # 步驟 4: 建立閾值並移除離群點
    # 使用百分位數來設定一個動態閾值，使其對不同點雲更具適應性
    delta_threshold = np.percentile(deltas, 90) * threshold_multiplier
    # delta 值小於閾值的點被視為內點 (inlier)
    inlier_indices = np.where(deltas < delta_threshold)[0]
    
    return pcd.select_by_index(inlier_indices), inlier_indices

def remove_confounding_points(pcd, k=20):
    """
    移除混淆點 (平滑化)。此函式實現了參考文件中「局部平面擬合與投影」的演算法。
    """
    if len(pcd.points) < k:
        return pcd
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_points = np.copy(points)

    # 遍歷每個點
    for i in range(len(points)):
        # 步驟 1: 搜尋 k 個最近的鄰居
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
        neighbors = points[idx]

        # 步驟 2: 使用鄰居點進行局部平面擬合
        # 計算鄰居點的質心
        centroid = np.mean(neighbors, axis=0)
        # 計算協方差矩陣
        cov_matrix = np.cov(neighbors.T)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            continue
        # 透過計算協方差矩陣的特徵向量，找到最小特徵值對應的特徵向量，即為平面的法向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]

        # 步驟 3: 將當前點投影到擬合出的局部平面上
        projection = points[i] - np.dot(points[i] - centroid, normal) * normal
        new_points[i] = projection
        
    pcd_projected = o3d.geometry.PointCloud()
    pcd_projected.points = o3d.utility.Vector3dVector(new_points)
    return pcd_projected

def fill_holes_local_adaptive_alpha(pcd, base_scale=2.0, local_factor=6.0):
    """[孔洞填補方法A] 使用自適應 Alpha-shape 演算法進行重建。"""
    points = np.asarray(pcd.points)
    if len(points) < 3:
        return pcd

    # 根據局部點密度計算自適應的 alpha 值
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=8)
    local_mean = np.mean(dists[:, 1:], axis=1)
    global_avg = np.mean(local_mean)

    alpha = np.clip(np.max(local_mean) * base_scale,
                    global_avg * 0.5,
                    global_avg * local_factor)
    print(f"[Alpha-fill] 使用自適應 alpha = {alpha:.5f}")

    # 從點雲創建 Alpha-shape 三角網格
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    # 從生成的網格上均勻採樣，得到填補孔洞後的點雲
    return mesh.sample_points_uniformly(len(points))

def fill_holes_poisson_enhanced(pcd, depth=8):
    """[孔洞填補方法B] 使用增強的泊松表面重建 (Poisson Surface Reconstruction)。"""
    # 估計並對齊法向量，這是泊松重建的關鍵前置步驟
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(20)
    
    # 從點雲創建泊松網格
    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # 對生成的網格進行裁剪，以去除邊界外的多餘部分
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.01, bbox.get_center())
    mesh_crop = mesh_poisson.crop(bbox)
    mesh_crop.remove_degenerate_triangles()
    mesh_crop.remove_duplicated_vertices()
    mesh_crop.compute_vertex_normals()
    # 從最終的網格上均勻採樣，得到點雲
    return mesh_crop.sample_points_uniformly(len(np.asarray(pcd.points)))

def fill_holes_combined(pcd):
    """
    [當前使用的孔洞填補方法] 結合 Alpha-shape 和泊松重建的優點。
    """
    print("執行 Alpha-shape 重建...")
    pcd_alpha = fill_holes_local_adaptive_alpha(pcd)
    print("執行泊松表面重建...")
    pcd_poisson = fill_holes_poisson_enhanced(pcd)

    # 裁剪泊松重建的結果，使其範圍與 Alpha-shape 的結果大致相同
    bbox_alpha = pcd_alpha.get_axis_aligned_bounding_box()
    pcd_poisson_cropped = pcd_poisson.crop(bbox_alpha)

    print("結合 Alpha-shape 和泊松重建的結果...")
    combined_points = np.vstack((np.asarray(pcd_alpha.points),
                                 np.asarray(pcd_poisson_cropped.points)))
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(combined_points)
    # 使用體素下採樣來合併重複的點並均化密度
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.001)
    return pcd_combined


def fill_holes(pcd, alpha_multiplier=4.0, max_area_threshold_factor=3.0, points_per_iteration=50):
    """
    [未被使用的孔洞填補方法] 實現參考文件中描述的「基於優先級的迭代填補」演算法。
    """
    pcd_filled = o3d.geometry.PointCloud(pcd)
    # 計算平均點距，用於設定 alpha 值
    avg_dist = np.mean(pcd_filled.compute_nearest_neighbor_distance())
    if avg_dist == 0:
        print("    平均點距為零，無法填補孔洞。")
        return pcd_filled
    alpha = avg_dist * alpha_multiplier
    max_iterations = 10 # 設定最大迭代次數以防止無限循環

    # 迭代填補孔洞
    for i in range(max_iterations):
        print(f"  孔洞填補迭代 {i+1}...")
        # 步驟 1: 創建 Alpha-shape 三角網格
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_filled, alpha)
            mesh.compute_vertex_normals()
        except Exception as e:
            print(f"    無法創建網格: {e}。停止填補。")
            break
        
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        if len(triangles) == 0:
            print("    網格中沒有三角形。停止填補。")
            break

        # 步驟 2: 計算每個三角形的面積作為優先級
        triangle_areas = np.zeros(len(triangles))
        for j, tri_indices in enumerate(triangles):
            v0, v1, v2 = vertices[tri_indices[0]], vertices[tri_indices[1]], vertices[tri_indices[2]]
            # 使用向量叉積計算面積
            triangle_areas[j] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        
        avg_area = np.mean(triangle_areas)
        max_area_threshold = avg_area * max_area_threshold_factor
        # 按面積從大到小排序，面積大的優先處理
        sorted_triangle_indices = np.argsort(triangle_areas)[::-1]
        
        new_centroids = []
        num_added = 0
        # 步驟 3: 在面積最大的三角形的重心處插入新點
        for tri_index in sorted_triangle_indices[:points_per_iteration]:
            # 如果三角形面積小於閾值，則認為它不是一個需要填補的孔洞
            if triangle_areas[tri_index] < max_area_threshold:
                break
            verts_of_largest_triangle = vertices[triangles[tri_index]]
            centroid = np.mean(verts_of_largest_triangle, axis=0)
            new_centroids.append(centroid)
            num_added += 1
            
        if num_added == 0:
            print("    在此次迭代中未找到需要填補的大孔洞。停止填補。")
            break
            
        current_points = np.asarray(pcd_filled.points)
        new_points_start_index = len(current_points)
        pcd_filled.points = o3d.utility.Vector3dVector(np.vstack((current_points, np.array(new_centroids))))
        print(f"    插入了 {num_added} 個新點。正在精煉其位置...")
        
        # 步驟 4: [額外步驟] 對新插入的點進行精煉，使其與周圍曲面更平滑
        pcd_filled_tree = o3d.geometry.KDTreeFlann(pcd_filled)
        points_to_refine_indices = range(new_points_start_index, len(pcd_filled.points))
        updated_points = np.asarray(pcd_filled.points)

        for pt_idx in points_to_refine_indices:
            point_to_refine = updated_points[pt_idx]
            # 類似 remove_confounding_points 的方法，將新點投影到局部平面上
            [_, idx, _] = pcd_filled_tree.search_knn_vector_3d(point_to_refine, 20)
            neighbors = updated_points[idx]
            
            centroid = np.mean(neighbors, axis=0)
            cov_matrix = np.cov(neighbors.T)
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                continue
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, np.argmin(eigenvalues)]
            
            projection = point_to_refine - np.dot(point_to_refine - centroid, normal) * normal
            updated_points[pt_idx] = projection
        pcd_filled.points = o3d.utility.Vector3dVector(updated_points)

    else:
        print(f"    已達到最大迭代次數 ({max_iterations})。停止填補。")
    return pcd_filled

def main():
    # --- 1. 設置與載入資料 ---
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data_name = "bunny" # 用於檔案命名

    # 載入乾淨的 Bunny 點雲
    try:
        bunny_mesh = o3d.data.BunnyMesh()
        pcd_clean = o3d.io.read_point_cloud(bunny_mesh.path)
    except Exception as e:
        print(f"無法載入 BunnyMesh: {e}")
        # 如果載入失敗，則創建一個球體作為替代
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)

    # data_name = "armadillo" # 用於檔案命名
    # # 載入乾淨的 Armadillo 點雲
    # try:
    #     armadillo_mesh = o3d.data.ArmadilloMesh()
    #     pcd_clean = o3d.io.read_point_cloud(armadillo_mesh.path)
    # except Exception as e:
    #     print(f"無法載入 ArmadilloMesh: {e}")
    #     mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    #     pcd_clean = mesh.sample_points_uniformly(number_of_points=20000)

    pcd_clean.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色
    print("顯示原始乾淨點雲...")
    # o3d.visualization.draw_geometries([pcd_clean], window_name="Original Clean Point Cloud")

    # --- 2. 添加瑕疵 (噪點與孔洞) ---
    noise_levels = [20] # 專注於一個噪點等級進行互動式調參

    for noise_level in noise_levels:
        print(f"\n========== 使用 {noise_level}% 噪點等級進行處理 ========== ")
        
        pcd_clean_copy = o3d.geometry.PointCloud(pcd_clean)

        # 呼叫 add_defects 函式生成帶有瑕疵的點雲
        pcd_noisy, noise_indices = add_defects(pcd_clean_copy, noise_level, num_holes=5)
        pcd_noisy.paint_uniform_color([1, 0, 0]) # 紅色
        print(f"  添加瑕疵後的點數: {len(pcd_noisy.points)} (包含 {len(noise_indices)} 個噪點)")
        
        # 儲存帶噪點的點雲
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_noisy_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_noisy)
        print(f"  已儲存: {filename}")
        
        # o3d.visualization.draw_geometries([pcd_noisy], window_name=f"{noise_level}% Noisy Point Cloud")
        
        # --- 3. 執行點雲優化流程 ---
        
        # --- 步驟 3.1: 移除離群點 ---
        print("1. 正在移除離群點...")
        # 可調整參數: octree_max_depth, k, threshold_multiplier
        pcd_outliers_removed, inlier_indices_denoised = remove_outliers(pcd_noisy, octree_max_depth=10, k=15, threshold_multiplier=0.5)
        
        # 立刻計算去噪性能指標
        calculate_denoising_metrics(len(pcd_noisy.points), noise_indices, inlier_indices_denoised)
        pcd_outliers_removed.paint_uniform_color([0, 1, 0]) # 綠色
        
        # 儲存移除離群點後的點雲
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_outliers_removed_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_outliers_removed)
        print(f"  已儲存: {filename}")

        # o3d.visualization.draw_geometries([pcd_outliers_removed], window_name="1. Outliers Removed")

        # --- 步驟 3.2: 移除混淆點 (表面平滑化) ---
        print("2. 正在移除混淆點 (局部平面投影)...")
        pcd_smoothed = pcd_outliers_removed
        smoothing_iterations = 3 # 執行多輪平滑以獲得更好效果
        smoothing_k = 30 # 平滑時考慮的鄰居數量
        for i in range(smoothing_iterations):
            print(f"  平滑化迭代 {i+1}/{smoothing_iterations}...")
            pcd_confounding_removed = remove_confounding_points(pcd_smoothed, k=smoothing_k)

        pcd_confounding_removed.paint_uniform_color([0, 0, 1]) # 藍色
        print(f"  所有平滑步驟後的點數: {len(pcd_confounding_removed.points)}")

        # 儲存移除混淆點後的點雲
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_confounding_removed_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_confounding_removed)
        print(f"  已儲存: {filename}")

        o3d.visualization.draw_geometries([pcd_confounding_removed], window_name="2. Confounding Points Removed")

        # --- 步驟 3.3: 填補孔洞 ---
        print("3. 正在填補孔洞 (使用 Alpha-shape + 泊松重建)...")
        pcd_holes_filled = fill_holes_combined(pcd_confounding_removed)
        pcd_holes_filled.paint_uniform_color([1, 1, 0]) # 黃色
        print(f"  填補孔洞後的點數: {len(pcd_holes_filled.points)}")

        # 儲存填補孔洞後的點雲
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{data_name}_holes_filled_{timestamp}.ply")
        o3d.io.write_point_cloud(filename, pcd_holes_filled)
        print(f"  已儲存: {filename}")

        o3d.visualization.draw_geometries([pcd_holes_filled], window_name="3. Holes Filled")

        # --- 4. 最終評估與視覺化比較 ---
        print(f"\n--- {noise_level}% 噪點等級的最終評估 ---")
        # 注意：此處評估的是填補孔洞前的點雲與乾淨點雲的相似度
        calculate_chamfer_distance(pcd_confounding_removed, pcd_clean_copy)

        print(f"顯示 {noise_level}% 噪點等級的最終比較:")
        pcd_clean_copy.paint_uniform_color([0.5, 0.5, 0.5]) # 原始: 灰色
        pcd_noisy.paint_uniform_color([1, 0, 0]) # 帶瑕疵: 紅色
        pcd_holes_filled.paint_uniform_color([0, 1, 0]) # 最終結果: 綠色
        
        # 為了方便並排比較，將點雲在 x 軸上平移
        pcd_noisy.translate((0.2, 0, 0))
        pcd_holes_filled.translate((0.4, 0, 0))

        o3d.visualization.draw_geometries([pcd_clean_copy, pcd_noisy, pcd_holes_filled],
                                          window_name=f"最終比較 (由左至右): 乾淨, 帶瑕疵, 處理後")

if __name__ == "__main__":
    main()
