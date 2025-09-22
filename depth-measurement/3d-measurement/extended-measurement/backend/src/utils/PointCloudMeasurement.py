import open3d as o3d
import numpy as np
from typing import List, Tuple, Optional

class PointCloudMeasurement:
    '''
    A class to measure volume and dimenisons of object pointclouds
    '''

    def __init__(self):
        
        # Note: parameters in millimeters 
        self.distance_thr = 10
        self.sample_points = 3
        self.max_iterations = 500

        self.voxel_size = 5
        self.points_buffer: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.points_buffer_plane: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()

        # For ground + HG
        self.point_cloud_tf: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.point_cloud_plane: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.cell = 5.0
        self.fill_iters = 1
        self.z_base = 10.0
        self.plane_eq = None
        self.plane_points = None
        self.center = None

        self.volume = None
        self.dimensions = None
        self.obb = None
        
        self.corners3d = None
        self.corners3d_table = None
        self.R_w2t = None
        self.t_w2t = None

        self.obb_pts = None
        self.obb_edges = None


    def update_point_cloud(self, points: np.ndarray):
        """
        Updates the Open3D point cloud data using an internal buffer, which makes execution faster.

        Args:
            points (np.ndarray): A NumPy array of 3D points.
        """
        if points.shape[0] > self.points_buffer.shape[0]:
            self.points_buffer = np.empty((points.shape[0], 3), dtype=np.float64)

        np.copyto(self.points_buffer[: points.shape[0]], points)
        self.point_cloud.points = o3d.utility.Vector3dVector(
            self.points_buffer[: points.shape[0]]
        )

    
    def update_point_cloud_plane(self, points: np.ndarray):
        """
        Updates the Open3D point cloud data using an internal buffer, which makes execution faster.

        Args:
            points (np.ndarray): A NumPy array of 3D points.
        """
        if points.shape[0] > self.points_buffer_plane.shape[0]:
            self.points_buffer_plane = np.empty((points.shape[0], 3), dtype=np.float64)

        np.copyto(self.points_buffer_plane[: points.shape[0]], points)
        self.point_cloud_plane.points = o3d.utility.Vector3dVector(
            self.points_buffer_plane[: points.shape[0]]
        )

    def set_point_cloud(
        self, pcl_points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Sets the point cloud for the fitter, applying filtering and downsampling.

        Args:
            pcl_points (np.ndarray): The 3D points of the point cloud.
            colors (np.ndarray, optional): The colors corresponding to the points. Defaults to None.
        """
        #print('shape object: ', pcl_points.shape)
        self.update_point_cloud(pcl_points)
        if colors is not None and colors.size > 0:
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        self.filter_point_cloud()

        filtered_points, filtered_colors = self.MAD_filtering(self.point_cloud)
        if filtered_points.size == 0:
            return
        self.point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
        self.point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        #print('shape object after filtering: ', filtered_points.shape)

    def set_point_cloud_plane(self, pcl_points: np.ndarray):

        #print('shape plane: ', pcl_points.shape)
        self.update_point_cloud_plane(pcl_points)
        self.filter_point_cloud_plane()

    def filter_point_cloud(self):

        self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=self.voxel_size)
        self.point_cloud, _ = self.point_cloud.remove_statistical_outlier(20, 0.1)

        points = np.asarray(self.point_cloud.points)

        # removes points with near z = 0 values 
        eps = 1e-6     
        mask = points[:, 2] > eps
        self.point_cloud = self.point_cloud.select_by_index(np.where(mask)[0])
    
    def filter_point_cloud_plane(self):

        self.point_cloud_plane = self.point_cloud_plane.voxel_down_sample(voxel_size=10)
        self.point_cloud_plane, _ = self.point_cloud_plane.remove_statistical_outlier(20, 0.1)

        points = np.asarray(self.point_cloud_plane.points)

        # removes points with near z = 0 values 
        eps = 1e-6     
        mask = points[:, 2] > eps
        self.point_cloud_plane = self.point_cloud_plane.select_by_index(np.where(mask)[0])

        #print('shape plane after filtering: ', np.array(self.point_cloud_plane.points).shape)
    
    
    def MAD_filtering(self, pcl: o3d.geometry.PointCloud, k: int = 3):
        """
        Filters the point cloud using the Median Absolute Deviation (MAD) method.

        This method is robust to outliers and filters points that are far from the median.

        Args:
            pcl (o3d.geometry.PointCloud): The input point cloud.
            k (int, optional): The number of MADs to use as a threshold. Defaults to 3.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the filtered points and colors.
        """
        colors = np.asarray(pcl.colors)
        points = np.asarray(pcl.points)

        if points.shape[0] == 0:
            return np.array([]), np.array([])

        median_point = np.median(points, axis=0)
        distances = np.linalg.norm(points - median_point, axis=1)
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))

        if mad == 0:
            mask = np.ones_like(distances, dtype=bool)
        else:
            mask = np.abs(distances - median_distance) < k * mad

        return points[mask], colors[mask]
    
    def fit_plane(self) -> Tuple[Optional[np.ndarray], Optional[List[int]], bool]:
        """
        Fits a plane to the point cloud using RANSAC.

        Returns:
            tuple: A tuple containing the plane equation, the inlier indices, and a boolean indicating success.
        """
        if len(self.point_cloud_plane.points) < self.sample_points:
            return None, None, False

        try:
            plane_eq, plane_inliers = self.point_cloud_plane.segment_plane(
                self.distance_thr, self.sample_points, self.max_iterations
            )
        except RuntimeError:
            return None, None, False
        
        inlier_ratio = len(plane_inliers) / len(self.point_cloud_plane.points)

        if inlier_ratio >= 0.2:
            return np.array(plane_eq), plane_inliers, True

        return None, None, False
    
    def clear_plane(self):
        self.plane_eq = None
        self.R_w2t = None
        self.t_w2t = None
    
    def to_table_frame(self):
        # ---- 0) validate plane ----
        peq = np.asarray(self.plane_eq, dtype=np.float64).ravel()
        if peq.size != 4 or not np.all(np.isfinite(peq)):
            raise ValueError("Bad plane_eq (must be 4 finite coeffs)")

        a, b, c, d = peq
        #print('plane equation: ', peq)
        n = np.array([a, b, c], dtype=np.float64)
        g = np.linalg.norm(n)
        if not np.isfinite(g) or g == 0.0:
            raise ValueError("Bad plane normal")

        n /= g
        d /= g

        # point on plane
        p0 = -d * n

        # orient normal so object lies at +z
        if self.center is not None:
            dotc = float(np.dot(self.center - p0, n))
            if np.isfinite(dotc) and dotc < 0.0:
                n = -n
                d = -d
                p0 = -d * n  # keep consistent

        # ---- 1) build orthonormal basis ----
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(n, ref))) > 0.99:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        x_axis = np.cross(ref, n)
        nx = np.linalg.norm(x_axis)
        if nx == 0.0 or not np.isfinite(nx):
            # fallback if cross failed numerically
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = np.cross(ref, n)
            nx = np.linalg.norm(x_axis)
            if nx == 0.0:
                raise ValueError("Could not build orthonormal basis")
        x_axis /= nx
        y_axis = np.cross(n, x_axis)

        E = np.column_stack([x_axis, y_axis, n])  # world basis as columns
        R = E.T                                   # world->table rotation (3x3)
        t = -R @ p0                               # translation so plane maps to z=0

        # ---- 2) transform points (ensure contiguous float64) ----
        P = np.asarray(self.point_cloud.points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3 or P.size == 0:
            raise ValueError("Empty or invalid object point cloud")

        # Use matmul in (N,3) form; avoid transpose juggling
        P_tf = P @ R.T + t  # (N,3) @ (3,3)ᵀ + (3,) → (N,3)

        # Ensure contiguous memory for Open3D
        P_tf = np.ascontiguousarray(P_tf, dtype=np.float64)

        # ---- 3) write into a FRESH geometry to avoid aliasing ----
        pc_tf = o3d.geometry.PointCloud()
        pc_tf.points = o3d.utility.Vector3dVector(P_tf)

        if self.point_cloud.has_colors():
            C = np.asarray(self.point_cloud.colors, dtype=np.float64)
            if C.shape[0] == P.shape[0] and C.shape[1] == 3 and np.all(np.isfinite(C)):
                pc_tf.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(C))
            # else: silently skip colors if mismatched

        self.point_cloud_tf = pc_tf
        self.R_w2t = R
        self.t_w2t = t
        return R, t
    
    def size_from_table_frame(self):
        pts = np.asarray(self.point_cloud_tf.points)
        H = float(np.percentile(pts[:,2], 98)) - 10.0  # robust height
        #L, W, rect_xy = self.obb2d_from_xy(pts[:, :2])
        L, W, rect_xy = self.min_area_rect_from_points(pts[:, :2])
        return L, W, H, rect_xy
    
    def obb2d_from_xy(self, xy: np.ndarray):
        mu = xy.mean(axis=0)
        X = xy - mu
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        R2 = Vt.T
        Y = X @ R2
        mn, mx = Y.min(axis=0), Y.max(axis=0)
        ext = mx - mn
        order = np.argsort(ext)[::-1]
        ext, R2, mn, mx = ext[order], R2[:, order], mn[order], mx[order]
        rect_p = np.array([[mn[0], mn[1]],[mx[0], mn[1]],[mx[0], mx[1]],[mn[0], mx[1]]])
        corners_xy = rect_p @ R2.T + mu
        return float(ext[0]), float(ext[1]), corners_xy
    
    def convex_hull_2d(self, xy: np.ndarray):
        # Andrew’s monotone chain (NumPy only)
        pts = np.unique(xy, axis=0)
        pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
        def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
        lower=[]
        for p in pts:
            while len(lower)>=2 and cross(lower[-2],lower[-1],p) <= 0: lower.pop()
            lower.append(p)
        upper=[]
        for p in pts[::-1]:
            while len(upper)>=2 and cross(upper[-2],upper[-1],p) <= 0: upper.pop()
            upper.append(p)
        return np.array(lower[:-1]+upper[:-1])   # CCW polygon

    def min_area_rect_from_points(self, xy: np.ndarray):
        hull = self.convex_hull_2d(xy)
        edges = np.diff(np.vstack([hull, hull[0]]), axis=0)
        angs  = np.unique(np.mod(np.arctan2(edges[:,1], edges[:,0]), np.pi/2))
        best = (np.inf, None)
        for th in angs:
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c,-s],[s,c]])      # rotate by +θ
            q = hull @ R.T                    # rotate points by −θ
            mn, mx = q.min(0), q.max(0)
            area = (mx - mn).prod()
            if area < best[0]: best = (area, (mn, mx, R))
        mn, mx, R = best[1]
        rect = np.array([[mn[0], mn[1]],
                        [mx[0], mn[1]],
                        [mx[0], mx[1]],
                        [mn[0], mx[1]]])
        rect_xy = rect @ R                   # <-- FIX: use R, not R.T
        L, W = (mx - mn)
        if L < W:
            L, W = W, L
            rect_xy = rect_xy[[3,0,1,2]]
        return float(L), float(W), rect_xy
    
    def points_in_poly(self, pts_xy: np.ndarray, poly_xy: np.ndarray):
        x, y = pts_xy[:,0], pts_xy[:,1]
        px, py = poly_xy[:,0], poly_xy[:,1]
        n = len(poly_xy)
        inside = np.zeros(len(pts_xy), dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = px[i], py[i]; xj, yj = px[j], py[j]
            cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            inside ^= cond
            j = i
        return inside
    
    def _grid_and_masks(self, polygon_xy, cell):
        xmin, ymin = polygon_xy.min(axis=0); xmax, ymax = polygon_xy.max(axis=0)
        nx = int(np.ceil((xmax - xmin)/cell)); ny = int(np.ceil((ymax - ymin)/cell))
        cx = xmin + (np.arange(nx) + 0.5) * cell
        cy = ymin + (np.arange(ny) + 0.5) * cell
        CX, CY = np.meshgrid(cx, cy)
        inside_cells = self.points_in_poly(np.c_[CX.ravel(), CY.ravel()], polygon_xy).reshape(ny, nx)
        return xmin, ymin, nx, ny, inside_cells

    def _assign_vertex_heights(self, pts, xmin, ymin, nx, ny, cell, inside_cells, z_base, fill_iters):
        x, y, z = pts[:,0], pts[:,1], np.clip(pts[:,2] - z_base, 0, None)
        Hv = np.full((ny+1, nx+1), -np.inf, float)
        i0 = np.floor((x - xmin)/cell).astype(int)
        j0 = np.floor((y - ymin)/cell).astype(int)
        for ii, jj, h in zip(i0, j0, z):
            if ii < 0 or jj < 0 or ii >= nx or jj >= ny: continue
            if not inside_cells[jj, ii]: continue
            Hv[jj  , ii  ] = max(Hv[jj  , ii  ], h)
            Hv[jj  , ii+1] = max(Hv[jj  , ii+1], h)
            Hv[jj+1, ii  ] = max(Hv[jj+1, ii  ], h)
            Hv[jj+1, ii+1] = max(Hv[jj+1, ii+1], h)
        Hv[~np.isfinite(Hv)] = 0.0
        mask_v = np.zeros_like(Hv, dtype=bool)
        mask_v[:-1,:-1] |= inside_cells
        mask_v[1: ,:-1] |= inside_cells
        mask_v[:-1,1: ] |= inside_cells
        mask_v[1: ,1: ] |= inside_cells
        for _ in range(fill_iters):
            Hpad = np.pad(Hv, 1, mode='edge')
            max8 = np.maximum.reduce([
                Hpad[0:-2,0:-2], Hpad[0:-2,1:-1], Hpad[0:-2,2:  ],
                Hpad[1:-1,0:-2],                    Hpad[1:-1,2:  ],
                Hpad[2:  ,0:-2], Hpad[2:  ,1:-1], Hpad[2:  ,2:  ],
            ])
            Hv = np.where((Hv==0) & mask_v, max8, Hv)
        return Hv, mask_v
    
    def make_height_surface_grid(self, polygon_xy):
        pts = np.asarray(self.point_cloud_tf.points)
        xmin, ymin, nx, ny, inside = self._grid_and_masks(polygon_xy, self.cell)
        Hv, mask_v = self._assign_vertex_heights(pts, xmin, ymin, nx, ny, self.cell, inside, self.z_base, self.fill_iters)
        gx = xmin + np.arange(nx+1) * self.cell
        gy = ymin + np.arange(ny+1) * self.cell
        return dict(xmin=xmin, ymin=ymin, nx=nx, ny=ny, cell=self.cell,
                    inside=inside, Hv=Hv, mask_v=mask_v, z_base=self.z_base, gx=gx, gy=gy)
    
    def volume_from_height_grid(self, G):
        """Integrate the same top: two triangles per inside cell, vertex heights in Hv."""
        nx, ny, cell, inside, Hv = G["nx"], G["ny"], G["cell"], G["inside"], G["Hv"]
        Atri = 0.5 * cell * cell
        V = 0.0
        for j in range(ny):
            for i in range(nx):
                if not inside[j,i]: continue
                h00 = Hv[j,i]; h10 = Hv[j,i+1]; h11 = Hv[j+1,i+1]; h01 = Hv[j+1,i]
                V += Atri * (h00 + h10 + h11) / 3.0
                V += Atri * (h00 + h11 + h01) / 3.0
        return float(V)  # mm^3
    
    def get_3d_corners_groundHG(self, rect_xy: np.ndarray, z_base: float, z_top: float):
        """
        rect_xy: (4,2) rectangle corners (counterclockwise) in the table plane
        z_base, z_top: heights in the table frame
        Returns an Open3D LineSet you can visualize.
        """
        bottom = np.c_[rect_xy, np.full(4, z_base)]
        top    = np.c_[rect_xy, np.full(4, z_top)]
        self.corners3d_table   = np.vstack([bottom, top])  # indices 0..3 bottom, 4..7 top

        self.corners3d = (self.corners3d_table - self.t_w2t) @ self.R_w2t

    def get_obb_wire(self):
        """Return (N=8 points, M=12 edges) in camera/world coords."""
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(self.obb)
        self.obb_pts = np.asarray(ls.points, dtype=float)      # (8,3)
        self.obb_edges = np.asarray(ls.lines,  dtype=int)      # (12,2)

    def get_measurement_OBB(self):

        self.obb = self.point_cloud.get_minimal_oriented_bounding_box()

        self.dimensions = self.obb.extent/10.0
        self.volume = float(self.obb.volume())/1000.0

        self.get_obb_wire()

        print('Dimensions [cm]: ', self.dimensions)
        print('Volume [cm3]: ', self.volume)

    def get_measurement_groundHG(self):

        if self.plane_eq is not None:

            #print('Plane eq: ', self.plane_eq)
            
            #print(np.array(self.point_cloud.points).shape)
            self.center = np.asarray(self.point_cloud.points).mean(axis=0)
            #print(self.center)
            R, t = self.to_table_frame()
            #print('Calculated transform')
            L, W, H, rect_xy = self.size_from_table_frame()
            #print('Calculated dimensions')
            
            G = self.make_height_surface_grid(rect_xy)
            #print('Calculated height grid')

            self.dimensions = np.array([L, W, H], dtype=float)/10.0
            self.volume = self.volume_from_height_grid(G)/1000.0

            self.get_3d_corners_groundHG(rect_xy, self.z_base, H)


            print('Dimensions [cm]: ', self.dimensions)
            print('Volume [cm3]: ', self.volume)
        else:
            print('Ground plane not set!')
    




        
        
