"""Time axis controller for 4D Gaussian playback."""


class TimeController:
    """Manages normalized time state and playback logic."""

    def __init__(self, total_frames: int = 300, fps: float = 30.0):
        self.current_time: float = 0.0   # normalized [0, 1]
        self.total_frames: int = max(total_frames, 1)
        self.fps: float = max(fps, 0.001)
        self.is_playing: bool = False
        self.loop: bool = True

    def set_time(self, t: float) -> None:
        """Set current time and pause playback."""
        self.current_time = max(0.0, min(1.0, t))
        self.is_playing = False

    def tick(self, delta_seconds: float) -> bool:
        """Advance time if playing. Returns True if time changed."""
        if not self.is_playing or delta_seconds <= 0:
            return False

        dt = delta_seconds * self.fps / max(self.total_frames - 1, 1)
        old = self.current_time
        self.current_time += dt

        if self.current_time >= 1.0:
            if self.loop:
                self.current_time = self.current_time % 1.0
            else:
                self.current_time = 1.0
                self.is_playing = False

        return self.current_time != old

    def get_frame_index(self) -> int:
        """Current time mapped to frame index."""
        return int(self.current_time * (self.total_frames - 1))

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing

    def stop(self) -> None:
        self.is_playing = False
        self.current_time = 0.0
