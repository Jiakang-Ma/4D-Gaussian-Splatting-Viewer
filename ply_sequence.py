"""PLY sequence loader for frame-by-frame playback."""

import os
import re
import glob
from typing import Optional, Tuple, List
import util_gau


class PLYSequenceLoader:
    """Scan a directory of frame_XXXXXX.ply files and load frames on demand."""

    def __init__(self):
        self.frame_paths: List[str] = []
        self.total_frames: int = 0
        self.current_frame: int = -1

    def scan_directory(self, dir_path: str) -> Tuple[bool, str]:
        """Scan for frame_*.ply files, sorted by frame number.
        Returns (success, error_message)."""
        pattern = os.path.join(dir_path, "frame_*.ply")
        files = glob.glob(pattern)

        if not files:
            self.frame_paths = []
            self.total_frames = 0
            return False, "所选目录中未找到 PLY 序列文件"

        # Extract frame numbers and sort
        def frame_num(path):
            m = re.search(r'frame_(\d+)\.ply$', os.path.basename(path))
            return int(m.group(1)) if m else 0

        files.sort(key=frame_num)
        self.frame_paths = files
        self.total_frames = len(files)
        self.current_frame = -1
        return True, ""

    def load_frame(self, frame_index: int) -> Optional[util_gau.GaussianData]:
        """Load a specific frame's PLY data."""
        if frame_index < 0 or frame_index >= self.total_frames:
            return None
        try:
            return util_gau.load_ply(self.frame_paths[frame_index])
        except Exception:
            return None
