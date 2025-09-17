from typing import Tuple, List, Optional

import depthai as dai
import numpy as np
from depthai_nodes.message import SegmentationMask


class DINOPatchMatcher(dai.node.ThreadedHostNode):
	def __init__(self) -> None:
		super().__init__()
		self._nn_size: Tuple[int, int] = (480, 352)
		self._selection_norm_xy: Optional[Tuple[float, float]] = None
		self._save_after_frames: int = 15
		self._frames_seen: int = 0
		self._force_save: bool = False
		self._template_vec: Optional[np.ndarray] = None
		self._similarity_thresh: float = 0.6
		self._save_path: Optional[str] = None

		self.input = self.createInput(
			types=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.NNData, True)],
			blocking=False,
			queueSize=4,
		)
		self.output = self.createOutput()

	def build(self, nn: dai.Node.Output) -> "DINOPatchMatcher":
		nn.link(self.input)
		return self

	def set_nn_size(self, nn_size: Tuple[int, int]) -> None:
		self._nn_size = nn_size

	def set_selection(self, x_norm: float, y_norm: float, save_after_frames: int) -> None:
		self._selection_norm_xy = (float(x_norm), float(y_norm))
		self._save_after_frames = int(save_after_frames)

	def set_similarity_threshold(self, thresh: float) -> None:
		self._similarity_thresh = float(thresh)

	def force_save_selection(self) -> None:
		self._force_save = True

	def set_save_path(self, path: Optional[str]) -> None:
		self._save_path = path

	def _extract_grid_features(self, nn_data: dai.NNData) -> Optional[np.ndarray]:
		name = "embeddings"
		arr = nn_data.getTensor(
			name, dequantize=True, storageOrder=dai.TensorInfo.StorageOrder.NCHW
		)
		feats = arr.transpose(0, 3, 1, 2)
		feats = feats.reshape(-1, feats.shape[3])
		feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
		return feats.reshape(arr.shape[3], arr.shape[1], arr.shape[2])

	def _grid_cell_from_norm_xy(self, grid_h: int, grid_w: int) -> Optional[Tuple[int, int]]:
		if self._selection_norm_xy is None:
			return None
		x_norm, y_norm = self._selection_norm_xy
		x_norm = min(max(x_norm, 0.0), 1.0)
		y_norm = min(max(y_norm, 0.0), 1.0)
		j = int(x_norm * grid_w)
		i = int(y_norm * grid_h)
		j = min(max(j, 0), grid_w - 1)
		i = min(max(i, 0), grid_h - 1)
		return (i, j)

	def _maybe_update_template(self, grid_feats: np.ndarray) -> None:
		if self._template_vec is not None and not self._force_save:
			return
		if self._selection_norm_xy is None:
			return
		if self._frames_seen < self._save_after_frames and not self._force_save:
			return
		self._force_save = False
		i_j = self._grid_cell_from_norm_xy(grid_feats.shape[0], grid_feats.shape[1])
		if i_j is None:
			return
		i, j = i_j
		vec = grid_feats[i, j]
		self._template_vec = vec.copy()
		if self._save_path:
			try:
				np.save(self._save_path, self._template_vec)
			except Exception:
				pass

	def _emit_mask(self, grid_feats: np.ndarray, nn_data: dai.NNData) -> None:
		h, w, c = grid_feats.shape
		mask_np = np.zeros((h, w), dtype=np.int16)
		if self._template_vec is not None:
			sims = np.tensordot(grid_feats, self._template_vec, axes=([2], [0]))
			mask_np[sims >= self._similarity_thresh] = 1

		seg = SegmentationMask()
		seg.mask = mask_np
		seg.setTimestamp(nn_data.getTimestamp())
		seg.setSequenceNum(nn_data.getSequenceNum())
		self.output.send(seg)

	def run(self) -> None:
		while self.isRunning():
			nn_data = self.input.tryGet()
			if nn_data is None:
				continue
			self._frames_seen += 1
			grid_feats = self._extract_grid_features(nn_data)
			if grid_feats is None:
				continue
			self._maybe_update_template(grid_feats)
			self._emit_mask(grid_feats, nn_data)

