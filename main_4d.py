"""4D Gaussian Splatting Viewer — main entry point.

Extends the original GaussianSplattingViewer with:
- 4D checkpoint (.pt) loading and time-axis playback
- Gaussian attribute inspection (ray-cast selection)
- Heatmap visualization modes
- PLY sequence playback
- Training bridge for real-time gradient/ADC monitoring
"""

import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import time as pytime
import os
import sys
import argparse
import imageio
import tkinter as tk
from tkinter import filedialog

# Add script dir to path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
os.chdir(dir_path)

import util
import util_gau
from renderer_ogl_4d import OpenGLRenderer4D
from loader_4d import Checkpoint4DData, load_checkpoint
from time_controller import TimeController
from heatmap import HeatmapManager, HeatmapMode
from inspector import GaussianInspector
from ply_sequence import PLYSequenceLoader
from training_bridge import TrainingBridge

# --- Global state ---
g_camera = util.Camera(720, 1280)
g_renderer: OpenGLRenderer4D = None
g_scale_modifier = 1.0
g_auto_sort = True
g_render_mode = 7
g_render_mode_tables = [
    "Gaussian Ball", "Flat Ball", "Billboard", "Depth",
    "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"
]

# 4D state
g_checkpoint_data: Checkpoint4DData = None
g_time_ctrl = TimeController(total_frames=120, fps=30.0)
g_heatmap = HeatmapManager()
g_inspector = GaussianInspector()
g_ply_seq = PLYSequenceLoader()
g_bridge = TrainingBridge()

# UI state
g_show_control = True
g_show_help = True
g_show_camera = False
g_show_properties = False
g_show_4d_panel = True
g_temporal_threshold = 0.01
g_error_msg = ""
g_error_timer = 0.0

# Mode: "static" | "4d_checkpoint" | "ply_sequence" | "training"
g_mode = "static"
g_gaussians = None
g_last_frame_time = 0.0


def impl_glfw_init():
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(g_camera.w, g_camera.h,
                                "4D Gaussian Splatting Viewer", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    if not window:
        glfw.terminate()
        exit(1)
    return window


def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)


def mouse_button_callback(window, button, action, mod):
    global g_show_properties
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

    # Middle click for Gaussian picking
    if button == glfw.MOUSE_BUTTON_MIDDLE and pressed and g_mode == "4d_checkpoint":
        _pick_gaussian(window)


def _pick_gaussian(window):
    """Ray-cast pick a Gaussian under the cursor."""
    global g_show_properties
    if g_checkpoint_data is None:
        return
    xpos, ypos = glfw.get_cursor_pos(window)
    w, h = glfw.get_framebuffer_size(window)

    # NDC coords
    ndc_x = (2.0 * xpos / w) - 1.0
    ndc_y = 1.0 - (2.0 * ypos / h)

    # Unproject ray
    fovx, fovy, focal = g_camera.get_htanfovxy_focal()
    ray_dir = np.array([ndc_x * fovx, ndc_y * fovy, 1.0], dtype=np.float32)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # Transform to world space
    view_mat = g_camera.get_view_matrix()
    rot_inv = view_mat[:3, :3].T
    ray_dir_world = rot_inv @ ray_dir
    ray_origin = g_camera.position.copy()

    # Compute current positions
    t = g_time_ctrl.current_time
    dt = t - g_checkpoint_data.times[:, 0]
    positions = g_checkpoint_data.means + g_checkpoint_data.velocities * dt[:, np.newaxis]
    if g_checkpoint_data.motion_order >= 2 and g_checkpoint_data.accels is not None:
        positions += 0.5 * g_checkpoint_data.accels * (dt ** 2)[:, np.newaxis]

    # Compute opacities
    s = np.maximum(np.exp(g_checkpoint_data.durations[:, 0]), 0.02)
    temporal_opa = np.exp(-0.5 * (dt / s) ** 2)
    base_opa = 1.0 / (1.0 + np.exp(-g_checkpoint_data.opacities[:, 0]))
    final_opa = base_opa * temporal_opa

    idx = g_inspector.pick(ray_origin, ray_dir_world, positions,
                           g_checkpoint_data.scales, final_opa)
    if idx is not None:
        g_show_properties = True
        g_renderer.set_selected_gaussian(idx)
        pos = positions[idx]
        print(f"[Pick] Selected Gaussian #{idx} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    else:
        g_show_properties = False
        g_renderer.set_selected_gaussian(-1)


def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)


def key_callback(window, key, scancode, action, mods):
    if action in (glfw.REPEAT, glfw.PRESS):
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)
        elif key == glfw.KEY_SPACE:
            g_time_ctrl.toggle_play()
        elif key == glfw.KEY_G:
            _toggle_gradient_heatmap()


def _toggle_gradient_heatmap():
    """Toggle gradient heatmap on/off with G key.
    
    Gradient mode works CPU-side: compute colors from grad_accum,
    replace SH data, and re-upload to GPU.
    """
    global g_gaussians
    if g_checkpoint_data is None or g_mode != "4d_checkpoint":
        return
    has_grad = (hasattr(g_checkpoint_data, 'grad_accum')
                and g_checkpoint_data.grad_accum is not None)
    if not has_grad:
        _show_error("This checkpoint has no grad_accum data")
        return

    if g_heatmap.mode == HeatmapMode.GRADIENT:
        # Turn off — restore original SH colors
        g_heatmap.mode = HeatmapMode.NONE
        g_renderer.set_heatmap_mode(0)
        g_gaussians = util_gau.GaussianData(
            xyz=g_checkpoint_data.means,
            rot=g_checkpoint_data.quats,
            scale=g_checkpoint_data.scales,
            opacity=1.0 / (1.0 + np.exp(-g_checkpoint_data.opacities)),
            sh=g_checkpoint_data.sh,
        )
        g_renderer.update_gaussian_data(g_gaussians)
        g_renderer.sort_and_update(g_camera)
        print("[Heatmap] Gradient OFF — restored original colors")
    else:
        # Turn on — compute gradient colors CPU-side
        g_heatmap.mode = HeatmapMode.GRADIENT
        # Don't set shader heatmap_mode — we handle this CPU-side
        g_renderer.set_heatmap_mode(0)

        scalars = g_checkpoint_data.grad_accum
        # Log scale for better visualization of skewed distribution
        log_scalars = np.log1p(scalars)  # log(1 + x) to handle zeros
        colors_rgb = g_heatmap.normalize_and_map(log_scalars)  # [N, 3]

        # Convert to SH DC format: (color - 0.5) / SH_C0
        SH_C0 = 0.28209479
        sh_colors = ((colors_rgb - 0.5) / SH_C0).astype(np.float32)

        # Pad to match original sh_dim
        sh_dim = g_checkpoint_data.sh.shape[1]
        if sh_colors.shape[1] < sh_dim:
            padding = np.zeros((sh_colors.shape[0], sh_dim - 3), dtype=np.float32)
            sh_colors = np.concatenate([sh_colors, padding], axis=1)

        g_gaussians = util_gau.GaussianData(
            xyz=g_checkpoint_data.means,
            rot=g_checkpoint_data.quats,
            scale=g_checkpoint_data.scales,
            opacity=1.0 / (1.0 + np.exp(-g_checkpoint_data.opacities)),
            sh=sh_colors,
        )
        g_renderer.update_gaussian_data(g_gaussians)
        g_renderer.sort_and_update(g_camera)

        vmin, vmax = float(log_scalars.min()), float(log_scalars.max())
        print(f"[Heatmap] Gradient ON — log1p range [{vmin:.3f}, {vmax:.3f}]")


def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)


def _show_error(msg: str):
    global g_error_msg, g_error_timer
    g_error_msg = msg
    g_error_timer = pytime.time() + 5.0


def _load_pt_file(path: str):
    """Load a .pt checkpoint file."""
    global g_checkpoint_data, g_mode, g_gaussians
    try:
        data = load_checkpoint(path)
    except (ValueError, Exception) as e:
        _show_error(f"加载失败: {e}")
        return

    g_checkpoint_data = data
    g_mode = "4d_checkpoint"

    # Convert to GaussianData for the base renderer pipeline
    g_gaussians = util_gau.GaussianData(
        xyz=data.means,
        rot=data.quats,
        scale=data.scales,
        opacity=1.0 / (1.0 + np.exp(-data.opacities)),  # sigmoid
        sh=data.sh,
    )
    g_renderer.update_gaussian_data(g_gaussians)
    g_renderer.sort_and_update(g_camera)

    # Upload 4D data
    g_renderer.upload_4d_data(data)
    g_renderer.upload_turbo_lut(g_heatmap.turbo_lut)

    # Set total frames from checkpoint metadata or CLI arg
    # The checkpoint stores normalized times [0, 1], so we need the actual frame count.
    # Try to read from checkpoint, otherwise use --total-frames arg.
    total = args.total_frames
    g_time_ctrl.total_frames = total
    g_time_ctrl.current_time = 0.0
    g_time_ctrl.is_playing = False
    g_renderer.update_time_uniform(0.0)

    print(f"Loaded 4D checkpoint: {data.n_gaussians} gaussians, "
          f"motion_order={data.motion_order}, step={data.train_step}, "
          f"total_frames={total}")


def _load_ply_file(path: str):
    """Load a single .ply file (static mode)."""
    global g_gaussians, g_mode, g_checkpoint_data
    try:
        g_gaussians = util_gau.load_ply(path)
        g_mode = "static"
        g_checkpoint_data = None
        g_renderer.update_gaussian_data(g_gaussians)
        g_renderer.sort_and_update(g_camera)
        # Disable 4D mode in shader
        util.set_uniform_1int(g_renderer.program, 0, "is_4d_mode")
    except Exception as e:
        _show_error(f"PLY 加载失败: {e}")


def _load_ply_sequence(dir_path: str):
    """Load a PLY sequence directory."""
    global g_mode, g_checkpoint_data
    ok, err = g_ply_seq.scan_directory(dir_path)
    if not ok:
        _show_error(err)
        return
    g_mode = "ply_sequence"
    g_checkpoint_data = None
    g_time_ctrl.total_frames = g_ply_seq.total_frames
    g_time_ctrl.current_time = 0.0
    g_time_ctrl.is_playing = False
    # Load first frame
    _load_ply_sequence_frame(0)
    print(f"Loaded PLY sequence: {g_ply_seq.total_frames} frames")


def _load_ply_sequence_frame(frame_idx: int):
    """Load and display a specific PLY sequence frame."""
    global g_gaussians
    gaus = g_ply_seq.load_frame(frame_idx)
    if gaus is not None:
        g_gaussians = gaus
        g_renderer.update_gaussian_data(gaus)
        g_renderer.sort_and_update(g_camera)
        util.set_uniform_1int(g_renderer.program, 0, "is_4d_mode")
        g_ply_seq.current_frame = frame_idx


def _draw_4d_panel():
    """Draw the 4D control panel in imgui."""
    global g_show_properties, g_temporal_threshold

    if not g_show_4d_panel:
        return

    imgui.begin("4D Controls", True)

    # --- Mode info ---
    if g_mode == "4d_checkpoint" and g_checkpoint_data:
        d = g_checkpoint_data
        imgui.text(f"Gaussians: {d.n_gaussians:,}")
        imgui.text(f"Motion order: {d.motion_order}")
        imgui.text(f"Time range: [{d.time_min:.3f}, {d.time_max:.3f}]")
        imgui.text(f"Train step: {d.train_step}")
        imgui.separator()

        # Time slider
        changed, t = imgui.slider_float("Time", g_time_ctrl.current_time,
                                        0.0, 1.0, "t = %.3f")
        if changed:
            g_time_ctrl.set_time(t)
            g_renderer.update_time_uniform(t)

        # Play controls
        play_label = "Pause" if g_time_ctrl.is_playing else "Play"
        if imgui.button(play_label):
            g_time_ctrl.toggle_play()
        imgui.same_line()
        if imgui.button("Stop"):
            g_time_ctrl.stop()
            g_renderer.update_time_uniform(0.0)

        imgui.same_line()
        changed, g_time_ctrl.fps = imgui.slider_float(
            "FPS", g_time_ctrl.fps, 1.0, 120.0, "%.0f")

        frame_idx = g_time_ctrl.get_frame_index()
        imgui.text(f"Frame: {frame_idx}/{g_time_ctrl.total_frames - 1}  "
                   f"(t={g_time_ctrl.current_time:.3f})")

        imgui.separator()

        # Temporal opacity threshold
        changed, g_temporal_threshold = imgui.slider_float(
            "Temporal Threshold", g_temporal_threshold,
            0.0, 1.0, "%.3f")
        if changed:
            g_renderer.set_temporal_threshold(g_temporal_threshold)
        imgui.same_line()
        if imgui.button("reset##thresh"):
            g_temporal_threshold = 0.01
            g_renderer.set_temporal_threshold(0.01)

        imgui.separator()

        # Heatmap mode
        heatmap_names = ["Normal", "Velocity", "Temporal Opacity",
                         "Duration", "Base Opacity"]
        has_grad = (hasattr(g_checkpoint_data, 'grad_accum')
                    and g_checkpoint_data.grad_accum is not None)
        if has_grad or g_bridge.connected:
            heatmap_names.append("Gradient")
        if has_grad:
            imgui.text("Grad data: available (press G to toggle)")
        else:
            imgui.text("Grad data: not in this checkpoint")
        changed, mode_idx = imgui.combo("Heatmap", g_heatmap.mode.value,
                                        heatmap_names)
        if changed:
            g_heatmap.mode = HeatmapMode(mode_idx)
            g_renderer.set_heatmap_mode(mode_idx)
            # Compute scalar range for shader normalization
            if g_heatmap.mode != HeatmapMode.NONE:
                scalars = g_heatmap.compute_scalar_field(
                    g_checkpoint_data, g_time_ctrl.current_time)
                if scalars is not None:
                    vmin, vmax = g_heatmap.get_range(scalars)
                    g_renderer.set_heatmap_range(vmin, vmax)

    elif g_mode == "ply_sequence":
        imgui.text(f"PLY Sequence: {g_ply_seq.total_frames} frames")
        changed, frame = imgui.slider_int("Frame", g_ply_seq.current_frame,
                                          0, g_ply_seq.total_frames - 1)
        if changed:
            _load_ply_sequence_frame(frame)

        play_label = "Pause" if g_time_ctrl.is_playing else "Play"
        if imgui.button(play_label):
            g_time_ctrl.toggle_play()

    elif g_mode == "training":
        imgui.text(f"Training Bridge: {'Connected' if g_bridge.connected else 'Disconnected'}")
        imgui.text(f"Step: {g_bridge.current_step}")

    else:
        imgui.text("Load a .pt checkpoint or PLY to begin")

    imgui.separator()

    # Training bridge controls
    if not g_bridge.connected:
        if imgui.button("Start Training Bridge"):
            g_bridge.start()
    else:
        if imgui.button("Stop Bridge"):
            g_bridge.stop()
        imgui.text(f"Bridge step: {g_bridge.current_step}")

    imgui.end()


def _draw_properties_panel():
    """Draw the Gaussian properties panel."""
    global g_show_properties
    if not g_show_properties or g_inspector.selected_index is None:
        return
    if g_checkpoint_data is None:
        return

    imgui.begin("Gaussian Properties", True)
    props = g_inspector.get_properties(
        g_inspector.selected_index, g_checkpoint_data,
        g_time_ctrl.current_time)

    imgui.text(f"Index: {props['index']}")
    imgui.separator()

    cx, cy, cz = props['canonical_pos']
    imgui.text(f"Canonical pos: ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    px, py, pz = props['current_pos']
    imgui.text(f"Current pos:   ({px:.4f}, {py:.4f}, {pz:.4f})")

    vx, vy, vz = props['velocity']
    imgui.text(f"Velocity: ({vx:.4f}, {vy:.4f}, {vz:.4f})")
    imgui.text(f"  |v| = {np.sqrt(vx**2+vy**2+vz**2):.6f}")

    if 'acceleration' in props:
        ax, ay, az = props['acceleration']
        imgui.text(f"Accel: ({ax:.4f}, {ay:.4f}, {az:.4f})")
    if 'jerk' in props:
        jx, jy, jz = props['jerk']
        imgui.text(f"Jerk:  ({jx:.4f}, {jy:.4f}, {jz:.4f})")
    if 'snap' in props:
        sx, sy, sz = props['snap']
        imgui.text(f"Snap:  ({sx:.4f}, {sy:.4f}, {sz:.4f})")

    imgui.separator()
    imgui.text(f"Canonical time: {props['canonical_time']:.4f}")
    imgui.text(f"Duration: {props['duration']:.4f} (log: {props['duration_log']:.4f})")
    imgui.text(f"Base opacity: {props['base_opacity']:.4f}")
    imgui.text(f"Temporal opacity: {props['temporal_opacity']:.4f}")
    imgui.text(f"Final opacity: {props['final_opacity']:.4f}")

    if 'grad_accum' in props:
        imgui.text(f"Grad accum: {props['grad_accum']:.6f}")

    if imgui.button("Deselect"):
        g_inspector.deselect()
        g_renderer.set_selected_gaussian(-1)
        g_show_properties = False

    imgui.end()


def _draw_error_overlay():
    """Show error message if any."""
    if g_error_msg and pytime.time() < g_error_timer:
        imgui.begin("Error", True)
        imgui.text_colored(g_error_msg, 1.0, 0.3, 0.3)
        imgui.end()


def main():
    global g_camera, g_renderer, g_scale_modifier, g_auto_sort
    global g_show_control, g_show_help, g_show_camera, g_show_4d_panel
    global g_render_mode, g_gaussians, g_last_frame_time, g_mode

    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()
    root.withdraw()

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_window_size_callback(window, window_resize_callback)

    # Init 4D renderer
    g_renderer = OpenGLRenderer4D(g_camera.w, g_camera.h)

    # Default gaussians
    g_gaussians = util_gau.naive_gaussian()
    g_renderer.update_gaussian_data(g_gaussians)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 4)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)

    g_last_frame_time = pytime.time()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        now = pytime.time()
        dt = now - g_last_frame_time
        g_last_frame_time = now

        # Update time controller
        if g_mode == "4d_checkpoint" and g_time_ctrl.tick(dt):
            g_renderer.update_time_uniform(g_time_ctrl.current_time)
            # Update heatmap range if in heatmap mode
            if g_heatmap.mode != HeatmapMode.NONE and g_checkpoint_data:
                scalars = g_heatmap.compute_scalar_field(
                    g_checkpoint_data, g_time_ctrl.current_time)
                if scalars is not None:
                    vmin, vmax = g_heatmap.get_range(scalars)
                    g_renderer.set_heatmap_range(vmin, vmax)

        elif g_mode == "ply_sequence" and g_time_ctrl.is_playing:
            g_time_ctrl.tick(dt)
            new_frame = g_time_ctrl.get_frame_index()
            if new_frame != g_ply_seq.current_frame:
                _load_ply_sequence_frame(
                    min(new_frame, g_ply_seq.total_frames - 1))

        # Poll training bridge
        if g_bridge.connected:
            frame = g_bridge.poll()
            if frame is not None:
                # Update display with training data
                if frame.means is not None and g_checkpoint_data:
                    g_checkpoint_data.means = frame.means
                if frame.grad_accum is not None and g_checkpoint_data:
                    g_checkpoint_data.grad_accum = frame.grad_accum
                g_bridge.current_step = frame.step

        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Update camera
        if g_camera.is_pose_dirty:
            g_renderer.update_camera_pose(g_camera)
            g_camera.is_pose_dirty = False
        if g_camera.is_intrin_dirty:
            g_renderer.update_camera_intrin(g_camera)
            g_camera.is_intrin_dirty = False

        if g_auto_sort:
            g_renderer.sort_and_update(g_camera)

        g_renderer.draw()

        # === imgui UI ===
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item("Open .pt Checkpoint")[0]:
                    path = filedialog.askopenfilename(
                        title="Open 4D Checkpoint",
                        filetypes=[('PyTorch Checkpoint', '*.pt')])
                    if path:
                        _load_pt_file(path)
                if imgui.menu_item("Open .ply File")[0]:
                    path = filedialog.askopenfilename(
                        title="Open PLY",
                        filetypes=[('PLY file', '*.ply')])
                    if path:
                        _load_ply_file(path)
                if imgui.menu_item("Open PLY Sequence Dir")[0]:
                    dir_path = filedialog.askdirectory(
                        title="Open PLY Sequence Directory")
                    if dir_path:
                        _load_ply_sequence(dir_path)
                imgui.end_menu()
            if imgui.begin_menu("Window", True):
                _, g_show_control = imgui.menu_item("Control", None, g_show_control)
                _, g_show_4d_panel = imgui.menu_item("4D Controls", None, g_show_4d_panel)
                _, g_show_camera = imgui.menu_item("Camera", None, g_show_camera)
                _, g_show_help = imgui.menu_item("Help", None, g_show_help)
                imgui.end_menu()
            imgui.end_main_menu_bar()

        # Control panel
        if g_show_control:
            imgui.begin("Control", True)
            imgui.text(f"fps = {imgui.get_io().framerate:.1f}")
            if g_gaussians:
                imgui.text(f"# Gaussians = {len(g_gaussians)}")

            changed, g_camera.fovy = imgui.slider_float(
                "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f")
            g_camera.is_intrin_dirty = changed

            changed, g_scale_modifier = imgui.slider_float(
                "scale", g_scale_modifier, 0.1, 10, "%.3f")
            if changed:
                g_renderer.set_scale_modifier(g_scale_modifier)

            changed, g_render_mode = imgui.combo(
                "shading", g_render_mode, g_render_mode_tables)
            if changed:
                g_renderer.set_render_mod(g_render_mode - 4)

            changed, g_auto_sort = imgui.checkbox("auto sort", g_auto_sort)
            if imgui.button("sort now"):
                g_renderer.sort_and_update(g_camera)

            if imgui.button("save image"):
                w, h = glfw.get_framebuffer_size(window)
                gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                gl.glReadBuffer(gl.GL_FRONT)
                buf = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8, -1).reshape(h, w, 3)
                imageio.imwrite("save.png", img[::-1])

            imgui.end()

        # 4D panel
        _draw_4d_panel()
        _draw_properties_panel()
        _draw_error_overlay()

        # Camera panel
        if g_show_camera:
            imgui.begin("Camera", True)
            if imgui.button("rot 180"):
                g_camera.flip_ground()
            changed, g_camera.target_dist = imgui.slider_float(
                "target dist", g_camera.target_dist, 1., 8., "%.3f")
            if changed:
                g_camera.update_target_distance()
            changed, g_camera.rot_sensitivity = imgui.slider_float(
                "rotate", g_camera.rot_sensitivity, 0.002, 0.1, "%.3f")
            changed, g_camera.trans_sensitivity = imgui.slider_float(
                "move", g_camera.trans_sensitivity, 0.001, 0.03, "%.3f")
            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                "zoom", g_camera.zoom_sensitivity, 0.001, 0.05, "%.3f")
            imgui.end()

        if g_show_help:
            imgui.begin("Help", True)
            imgui.text("File > Open .pt Checkpoint  — load 4D data")
            imgui.text("File > Open .ply File       — load static PLY")
            imgui.text("File > Open PLY Sequence Dir — load frame sequence")
            imgui.text("")
            imgui.text("Left click + drag  — rotate camera")
            imgui.text("Right click + drag — translate camera")
            imgui.text("Middle click       — select Gaussian (4D mode)")
            imgui.text("Q/E                — roll camera")
            imgui.text("Space              — play/pause")
            imgui.text("G                  — toggle gradient heatmap")
            imgui.text("Scroll             — zoom")
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # Cleanup
    g_bridge.stop()
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(
        description="4D Gaussian Splatting Viewer")
    parser.add_argument("--hidpi", action="store_true",
                        help="Enable HiDPI scaling")
    parser.add_argument("--port", type=int, default=6009,
                        help="Training bridge port")
    parser.add_argument("--total-frames", type=int, default=120,
                        help="Total number of frames in the sequence (e.g., 120 for 0-119)")
    args = parser.parse_args()
    g_bridge.port = args.port
    main()
