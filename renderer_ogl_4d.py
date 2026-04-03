"""Extended OpenGL renderer with 4D Gaussian support.

Adds SSBO binding=2 for 4D temporal parameters, binding=3 for turbo LUT,
and uniforms for time, motion order, heatmap mode, and selection.
"""

from OpenGL import GL as gl
import numpy as np
import util
import util_gau
from renderer_ogl import GaussianRenderBase, _sort_gaussian
from loader_4d import Checkpoint4DData

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except Exception:
    wglSwapIntervalEXT = None


class OpenGLRenderer4D(GaussianRenderBase):
    """OpenGL renderer with 4D checkpoint support."""

    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders(
            'shaders/gau_vert_4d.glsl', 'shaders/gau_frag_4d.glsl')

        # Quad geometry
        self.quad_v = np.array([
            -1, 1, 1, 1, 1, -1, -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2, 0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)

        vao, _ = util.set_attributes(
            self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao

        self.gau_bufferid = None
        self.index_bufferid = None
        self._4d_bufferid = None
        self._lut_bufferid = None

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # 4D state
        self.is_4d = False
        self.motion_order = 1
        self._num_gaussians = 0

        self._set_4d_defaults()
        self.update_vsync()

    def _set_4d_defaults(self):
        util.set_uniform_1f(self.program, 0.0, "current_time")
        util.set_uniform_1int(self.program, 1, "motion_order")
        util.set_uniform_1int(self.program, 0, "heatmap_mode")
        util.set_uniform_1int(self.program, -1, "selected_gaussian")
        util.set_uniform_1f(self.program, 0.01, "temporal_threshold")
        util.set_uniform_1int(self.program, 0, "is_4d_mode")
        util.set_uniform_1f(self.program, 0.0, "heatmap_min")
        util.set_uniform_1f(self.program, 1.0, "heatmap_max")

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)

    # --- Standard renderer interface ---

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(
            self.program, "gaussian_data", gaussian_data,
            bind_idx=0, buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(
            self.program, "gi", index,
            bind_idx=1, buffer_id=self.index_bufferid)

    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(),
                            "hfovxy_focal")

    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            len(self.quad_f.reshape(-1)),
            gl.GL_UNSIGNED_INT, None, num_gau)

    # --- 4D-specific methods ---

    def upload_4d_data(self, data: Checkpoint4DData):
        """Pack and upload 4D temporal parameters to SSBO binding=2."""
        N = data.n_gaussians
        self.is_4d = True
        self.motion_order = data.motion_order
        self._num_gaussians = N

        dim_4d = 5 + max(0, data.motion_order - 1) * 3
        buf = np.zeros((N, dim_4d), dtype=np.float32)
        buf[:, 0] = data.times[:, 0]
        buf[:, 1] = data.durations[:, 0]
        buf[:, 2:5] = data.velocities

        offset = 5
        if data.motion_order >= 2 and data.accels is not None:
            buf[:, offset:offset+3] = data.accels
            offset += 3
        if data.motion_order >= 3 and data.jerks is not None:
            buf[:, offset:offset+3] = data.jerks
            offset += 3
        if data.motion_order >= 4 and data.snaps is not None:
            buf[:, offset:offset+3] = data.snaps

        flat = np.ascontiguousarray(buf.reshape(-1))
        self._4d_bufferid = util.set_storage_buffer_data(
            self.program, "gaussian_4d_data", flat,
            bind_idx=2, buffer_id=self._4d_bufferid)

        util.set_uniform_1int(self.program, 1, "is_4d_mode")
        util.set_uniform_1int(self.program, data.motion_order, "motion_order")

    def upload_turbo_lut(self, lut: np.ndarray):
        """Upload turbo colormap LUT [256, 3] to SSBO binding=3."""
        flat = np.ascontiguousarray(lut.reshape(-1).astype(np.float32))
        self._lut_bufferid = util.set_storage_buffer_data(
            self.program, "turbo_lut", flat,
            bind_idx=3, buffer_id=self._lut_bufferid)

    def update_time_uniform(self, t: float):
        util.set_uniform_1f(self.program, t, "current_time")

    def set_motion_order(self, order: int):
        self.motion_order = order
        util.set_uniform_1int(self.program, order, "motion_order")

    def set_heatmap_mode(self, mode: int):
        util.set_uniform_1int(self.program, mode, "heatmap_mode")

    def set_heatmap_range(self, vmin: float, vmax: float):
        util.set_uniform_1f(self.program, vmin, "heatmap_min")
        util.set_uniform_1f(self.program, vmax, "heatmap_max")

    def set_selected_gaussian(self, index: int):
        util.set_uniform_1int(self.program, index, "selected_gaussian")

    def set_temporal_threshold(self, threshold: float):
        util.set_uniform_1f(self.program, threshold, "temporal_threshold")
