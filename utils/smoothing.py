import time
import numpy as np
import utils.data as data_utils
from utils.json_manager import load_json
import utils.globals as g


class VectorKalmanFilter:
    """Multi‑dimensional (vector) Kalman filter used for the high‑priority actions
    that need per‑frame smoothing. A separate instance is created per *action*,
    not per index, so that highly‑correlated channels (e.g. XYZ position) can be
    treated together.
    """

    def __init__(self, q_process: float, r_measure: float, dim: int):
        self.dim = dim
        self.Q = np.eye(dim) * q_process  # Process‑noise covariance
        self.R = np.eye(dim) * r_measure  # Measurement‑noise covariance
        self.x = None                     # State estimate (dim,)
        self.P = np.eye(dim)              # Estimate covariance

    def predict(self, dt: float = 1.0):
        """Time‑update (prediction) step."""
        if self.x is None:
            return  # filter not initialised yet
        # Simple random‑walk model ⇒ F = I, so only P changes
        self.P += self.Q * dt

    def update(self, z, is_rotation: bool = False):
        """Measurement‑update (correction) step.

        Args:
            z            : iterable/np.ndarray of new observations (dim,)
            is_rotation  : treat the measurements as angles in degrees and wrap
                            differences across ±180°.
        Returns:
            np.ndarray   : the updated state estimate (copy).
        """
        z = np.asarray(z, dtype=np.float64)

        # First measurement initialises the filter
        if self.x is None:
            self.x = z.copy()
            return self.x.copy()

        # Innovation (measurement residual)
        if is_rotation:
            y = np.vectorize(angle_diff)(z, self.x)
        else:
            y = z - self.x

        # Innovation covariance & Kalman gain
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K) @ self.P
        return self.x.copy()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def angle_diff(current: float, target: float) -> float:
    """Minimum signed difference from *target* to *current* (deg)."""
    diff = (current - target) % 360.0
    return diff - 360.0 if diff > 180.0 else diff


def update_target_value(target: dict, smoothed_delta: float, is_rotation: bool):
    """Write the smoothed delta back into the target slot (in‑place)."""
    if is_rotation:
        target["v"] = (target["v"] + smoothed_delta) % 360.0
    else:
        target["v"] += smoothed_delta


# ---------------------------------------------------------------------------
# Public API used by the main application
# ---------------------------------------------------------------------------

def setup_smoothing():
    """Load settings/smoothing.json and prepare global structures.

    Returns the parsed configuration so that callers can inspect it if they
    need to.
    """
    smoothing_config = load_json("./settings/smoothing.json")

    g.kalman_filters = {}
    g.indices_map = {}
    g.smoothing_config = smoothing_config

    # Build one VectorKalmanFilter per *action* that declares kalman_params.
    for action, params in smoothing_config["Parameters"].items():
        indices = params.get("indices", [])
        g.indices_map[action] = indices  # may be an empty list

        if action == "OtherBlendShapes":
            # Explicitly skip Kalman filter creation for the fallback bucket
            continue

        if "kalman_params" in params and indices:
            q = params["kalman_params"]["q_process"]
            r = params["kalman_params"]["r_measure"]
            dim = len(indices)
            g.kalman_filters[action] = VectorKalmanFilter(q, r, dim)

    return smoothing_config


# ---------------------------------------------------------------------------
# Main worker: applies smoothing every frame in its own thread
# ---------------------------------------------------------------------------

def apply_smoothing():
    """Continuous loop that smooths g.latest_data into g.data until the
    stop_event is set.  It follows the new rules:

    *  For all actions that declare explicit indices, we keep the existing
       (vector) Kalman filter pipeline.
    *  Any index **not** covered by those lists is treated as an
       "OtherBlendShapes" channel.
    *  OtherBlendShapes NEVER uses the Kalman filter ‑ values are smoothed via
       the same first‑order exponential scheme as before.
    """

    last_time = time.perf_counter()
    frame_duration = 1.0 / 1000.0  # 1 kHz worker loop

    while not g.stop_event.is_set() and g.config["Smoothing"]["enable"]:
        now = time.perf_counter()
        dt_base = now - last_time  # seconds since previous iteration

        # Keep track of which indices have already been handled so that we can
        # drop the remainder into OtherBlendShapes afterwards.
        handled_indices: set[int] = set()

        # ------------------------------------------------------------------
        # Pass 1 – process all *named* actions
        # ------------------------------------------------------------------
        for action, params in g.smoothing_config["Parameters"].items():
            indices = g.indices_map.get(action, [])
            if not indices:  # skip empty index lists (we'll handle leftovers later)
                continue

            handled_indices.update(indices)

            target_key = params["key"]
            shifting = params.get("shifting", 0)
            is_rotation = params.get("is_rotation", False)
            dt_mul = params.get("dt_multiplier", 20)
            dt = dt_base * dt_mul

            # Gather the observation vector for this action
            try:
                obs_vec = [g.latest_data[idx] for idx in indices]
            except IndexError:
                # Source array shorter than expected – just skip this action
                continue

            # Kalman filter unless this is the catch‑all OtherBlendShapes bucket
            if action in g.kalman_filters:
                kf = g.kalman_filters[action]
                kf.predict(dt_base)
                filt_vec = kf.update(obs_vec, is_rotation)
            else:
                filt_vec = obs_vec  # raw values (no KF)

            # Write the smoothed deltas back to the destination buffer
            for local_i, idx in enumerate(indices):
                target_idx = idx - shifting
                data_array = g.data[target_key]
                if not (0 <= target_idx < len(data_array)):
                    continue  # out‑of‑range – ignore gracefully

                target = data_array[target_idx]
                raw = filt_vec[local_i]
                delta = angle_diff(raw, target["v"]) if is_rotation else raw - target["v"]
                sm_delta = delta * dt
                update_target_value(target, sm_delta, is_rotation)

        # ------------------------------------------------------------------
        # Pass 2 – fall back to "OtherBlendShapes" for *everything else*
        # ------------------------------------------------------------------
        other_cfg = g.smoothing_config["Parameters"].get("OtherBlendShapes", {})
        if other_cfg:
            target_key = other_cfg.get("key", "blendShapes")
            shifting = other_cfg.get("shifting", 0)
            is_rotation = other_cfg.get("is_rotation", False)
            dt_mul = other_cfg.get("dt_multiplier", 20)
            dt = dt_base * dt_mul

            for idx, raw in enumerate(g.latest_data):
                if idx in handled_indices:
                    continue  # already processed above

                target_idx = idx - shifting
                data_array = g.data[target_key]
                if not (0 <= target_idx < len(data_array)):
                    continue

                target = data_array[target_idx]
                delta = angle_diff(raw, target["v"]) if is_rotation else raw - target["v"]
                sm_delta = delta * dt
                update_target_value(target, sm_delta, is_rotation)

        # ------------------------------------------------------------------
        last_time = now
        time.sleep(frame_duration)
