"""Numerical compatibility tests for ExVR -> upstream oneup03/VRto3D OpenTrack.

The upstream reference code transcribed here is:

  vrto3d/src/hmd_device_driver.cpp  OpenTrackThread():
      struct TOpenTrack { double X, Y, Z, Yaw, Pitch, Roll; };
      open_track_att_ = HmdQuaternion_FromEulerAngles(
          DEG_TO_RAD(open_track.Roll), DEG_TO_RAD(open_track.Pitch),
          DEG_TO_RAD(-open_track.Yaw));
      open_track_pos_ = { -(open_track.X / 100.0), -(open_track.Y / 100.0),
                           open_track.Z / 100.0 };

  utils/vrmath/vrmath.h  HmdQuaternion_FromEulerAngles()

Run from the repository root:  python3 test_vrto3d_compat.py
"""

import math
import random
import struct

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.sender import pack_hmd_data

DEG_TO_RAD = math.pi / 180.0


def hmd_quaternion_from_euler_angles(roll, pitch, yaw):
    """1:1 transcription of HmdQuaternion_FromEulerAngles in vrmath.h."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    # returns w, x, y, z
    return (
        cr * cp * cy + sr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        sr * cp * cy - cr * sp * sy,
    )


def upstream_rebuild(x, y, z, yaw_deg, pitch_deg, roll_deg):
    """Rebuild the SteamVR pose exactly like upstream OpenTrackThread does."""
    q = hmd_quaternion_from_euler_angles(
        roll_deg * DEG_TO_RAD, pitch_deg * DEG_TO_RAD, -yaw_deg * DEG_TO_RAD
    )
    rotation = R.from_quat([q[1], q[2], q[3], q[0]])
    position = np.array([-x / 100.0, -y / 100.0, z / 100.0])
    return rotation, position


def old_fork_rotation(rot):
    """The DriverPose_t orientation the old VirtualHMD fork produced."""
    return R.from_euler("xzy", [rot[1], rot[2], -rot[0]], degrees=True)


def old_fork_position(pos):
    """The SteamVR position the old VirtualHMD fork produced."""
    return np.array([pos[0] / 100.0, pos[2] / 100.0, pos[1] / 100.0])


def make_pose_data(rot, pos):
    entry = lambda v: {"e": True, "v": v, "s": 0.0}
    data = {
        "Rotation": [entry(v) for v in rot],
        "Position": [entry(v) for v in pos],
    }
    return data, data


def angular_error_deg(a, b):
    return math.degrees((a * b.inv()).magnitude())


def test_upstream_euler_convention_matches_scipy_zxy():
    rng = random.Random(1)
    worst = 0.0
    for _ in range(2000):
        roll, pitch, yaw = (rng.uniform(-math.pi, math.pi) for _ in range(3))
        w, x, y, z = hmd_quaternion_from_euler_angles(roll, pitch, yaw)
        expected = R.from_quat([x, y, z, w])
        actual = R.from_euler("zxy", [roll, pitch, yaw])
        worst = max(worst, angular_error_deg(expected, actual))
    assert worst < 1e-6, f"vrmath.h != scipy zxy, worst {worst} deg"


def test_orientation_round_trip():
    rng = random.Random(2)
    worst = 0.0
    for _ in range(500):
        rot = [rng.uniform(-180, 180) for _ in range(3)]
        pos = [rng.uniform(-30, 30) for _ in range(3)]

        data, default = make_pose_data(rot, pos)
        packet = pack_hmd_data(data, default)
        assert len(packet) == 48, f"packet must be 48 bytes, got {len(packet)}"
        x, y, z, yaw, pitch, roll = struct.unpack("<6d", packet)

        rebuilt_rotation, rebuilt_position = upstream_rebuild(x, y, z, yaw, pitch, roll)
        worst = max(
            worst,
            angular_error_deg(old_fork_rotation(rot), rebuilt_rotation),
            float(np.abs(old_fork_position(pos) - rebuilt_position).max()),
        )
    assert worst < 1e-4, f"round trip mismatch, worst {worst}"


def test_position_mapping():
    pos = [10.0, -20.0, 30.0]
    data, default = make_pose_data([0.0, 0.0, 0.0], pos)
    packet = pack_hmd_data(data, default)
    x, y, z, yaw, pitch, roll = struct.unpack("<6d", packet)
    _, rebuilt_position = upstream_rebuild(x, y, z, yaw, pitch, roll)
    assert np.allclose(old_fork_position(pos), rebuilt_position, atol=1e-9), (
        f"position mismatch: old fork {old_fork_position(pos)} vs upstream {rebuilt_position}"
    )


if __name__ == "__main__":
    test_upstream_euler_convention_matches_scipy_zxy()
    print("PASS: upstream HmdQuaternion_FromEulerAngles == scipy 'zxy' convention")
    test_orientation_round_trip()
    print("PASS: orientation + position round trip through upstream VRto3D math (48-byte <6d packet)")
    test_position_mapping()
    print("PASS: position mapping preserves the old fork's SteamVR pose")
