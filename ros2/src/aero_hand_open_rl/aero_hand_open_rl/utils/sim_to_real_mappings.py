# Copyright 2025 TetherIA, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# utils/sim_to_real_mappings.py
"""
Actuator mapping utilities:
- Convert actuator range of aero hand open <-> MuJoCo tendon length range
- Array-based API (fixed order), avoids dict overhead
"""
import math

from aero_open_sdk.joints_to_actuations import JointsToActuationsModel
from aero_open_sdk.aero_hand_constants import AeroHandConstants

# Index: [index, middle, ring, pinky, thumb_abd, th1, th2]
# Finger SIM_RANGE matches MuJoCo actuator ctrlrange (right_*_A_tendon).
SIM_RANGE = [
    (0.05852, 0.110387),  # right_index_A_tendon
    (0.05852, 0.110387),  # right_middle_A_tendon
    (0.05852, 0.110387),  # right_ring_A_tendon
    (0.05852, 0.110387),  # right_pinky_A_tendon
    (-0.0254462, 1.77858),  # right_thumb_A_cmc_abd in sim
    (0.026941, 0.0382787),  # right_th1_A_pip in sim
    (0.0839985, 0.110133),  # right_th2_A_pip in sim
]

ACTUATIONS_LOWER_LIMITS = AeroHandConstants.actuation_lower_limits
ACTUATIONS_UPPER_LIMITS = AeroHandConstants.actuation_upper_limits

ACTUATION_RANGE = [
    (
        ACTUATIONS_LOWER_LIMITS[0],
        ACTUATIONS_UPPER_LIMITS[0],
    ),  # right_thumb_A_abd in servo
    (
        ACTUATIONS_LOWER_LIMITS[1],
        ACTUATIONS_UPPER_LIMITS[1],
    ),  # right_thumb_A_flex in servo
    (
        ACTUATIONS_LOWER_LIMITS[2],
        ACTUATIONS_UPPER_LIMITS[2],
    ),  # right_thumb_A_mcp in servo
    (ACTUATIONS_LOWER_LIMITS[3], ACTUATIONS_UPPER_LIMITS[3]),  # right_index
    (ACTUATIONS_LOWER_LIMITS[4], ACTUATIONS_UPPER_LIMITS[4]),  # right_middle
    (ACTUATIONS_LOWER_LIMITS[5], ACTUATIONS_UPPER_LIMITS[5]),  # right_ring
    (ACTUATIONS_LOWER_LIMITS[6], ACTUATIONS_UPPER_LIMITS[6]),  # right_pinky
]

THUMB_ABD_ACTUATION = 0
THUMB_FLEX_ACTUATION = 1
THUMB_MCP_ACTUATION = 2
FINGER_IDX_ACTUATION = 3
FINGER_MIDDLE_ACTUATION = 4
FINGER_RING_ACTUATION = 5
FINGER_PINKY_ACTUATION = 6

THUMB_ABD_SIM = 4
THUMB_FLEX_SIM = 5
THUMB_MCP_SIM = 6
FINGER_IDX_SIM = 0
FINGER_MIDDLE_SIM = 1
FINGER_RING_SIM = 2
FINGER_PINKY_SIM = 3

PI = 3.141592653589793
MOTOR_PULLEY_RADIUS = 9.000  # mm

# Sim settle anchors: ctrl → (MCP, PIP, DIP) degrees. Measured 2026-07-31 on
# AeroCubeRotateZAxis.
_FINGER_CTRL_JOINT_ANCHORS = {
    # ctrl: (mcp, pip, dip) deg — shared shape, pinky slightly more flexed at home
    FINGER_IDX_SIM: (
        (0.110387, (0.0, 0.0, 0.0)),
        (0.1040, (17.0, 0.0, 0.0)),
        (0.0900, (73.6, 0.0, 0.0)),
        (0.0700, (90.1, 57.8, 57.9)),
        (0.05852, (90.3, 90.1, 90.1)),
    ),
    FINGER_MIDDLE_SIM: (
        (0.110387, (0.0, 0.0, 0.0)),
        (0.1040, (19.7, 0.0, 0.0)),
        (0.0900, (75.6, 0.0, 0.0)),
        (0.0700, (90.1, 59.9, 60.0)),
        (0.05852, (90.3, 90.1, 90.2)),
    ),
    FINGER_RING_SIM: (
        (0.110387, (0.0, 0.0, 0.0)),
        (0.1040, (16.9, 0.0, 0.0)),
        (0.0900, (73.6, 0.0, 0.0)),
        (0.0700, (90.1, 57.7, 57.9)),
        (0.05852, (90.3, 90.1, 90.1)),
    ),
    FINGER_PINKY_SIM: (
        (0.110387, (0.0, 0.0, 0.0)),
        (0.1040, (26.2, 0.0, 0.0)),
        (0.0900, (82.4, 0.0, 0.0)),
        (0.0700, (90.1, 65.5, 65.6)),
        (0.05852, (90.5, 90.2, 90.3)),
    ),
}

_JTA = JointsToActuationsModel()

# Real index calibration, measured visually on 2026-07-31 after homing:
#
#   commanded motor deg: 48.02, 72.03, 96.04, 111.20, 189.20
#   observed MCP deg:      0.0,  15.0,  30.0,  40.0,  90.0
#
# Least-squares fit (RMSE ≈ 0.35 motor deg):
#   motor_deg = 48.486 + 1.5661 * MCP_deg
#
# The 48.5° intercept is real cable take-up. The old ideal cable model omitted
# it and used only the MCP cable term (≈1.39 motor-deg/MCP-deg), so this
# checkpoint's simulated 17–39° MCP motion became only 23–53° at the motor,
# almost entirely inside the real dead zone.
_INDEX_REAL_MOTOR_OFFSET_DEG = 48.486
_INDEX_REAL_MOTOR_PER_MCP_DEG = 1.5661
_INDEX_DISTAL_FLEX_START_CTRL = 0.0700


def _interp_finger_joints(
    sim_ctrl: float, finger_sim_idx: int
) -> tuple[float, float, float]:
    """Piecewise-linear sim ctrl → (mcp, pip, dip) radians from settle anchors."""
    anchors = _FINGER_CTRL_JOINT_ANCHORS[finger_sim_idx]
    # anchors sorted open→curl (descending ctrl)
    if sim_ctrl >= anchors[0][0]:
        mcp, pip, dip = anchors[0][1]
    elif sim_ctrl <= anchors[-1][0]:
        mcp, pip, dip = anchors[-1][1]
    else:
        mcp = pip = dip = 0.0
        for i in range(len(anchors) - 1):
            c0, j0 = anchors[i]
            c1, j1 = anchors[i + 1]
            if c1 <= sim_ctrl <= c0:
                t = (c0 - sim_ctrl) / (c0 - c1) if c0 != c1 else 0.0
                mcp = j0[0] + t * (j1[0] - j0[0])
                pip = j0[1] + t * (j1[1] - j0[1])
                dip = j0[2] + t * (j1[2] - j0[2])
                break
    return (
        math.radians(mcp),
        math.radians(pip),
        math.radians(dip),
    )


def _index_real_actuation(sim_ctrl: float) -> float:
    """Map sim index ctrl to measured real motor degrees."""
    mcp, _, _ = _interp_finger_joints(sim_ctrl, FINGER_IDX_SIM)
    mcp_deg = math.degrees(mcp)
    motor_deg = (
        _INDEX_REAL_MOTOR_OFFSET_DEG
        + _INDEX_REAL_MOTOR_PER_MCP_DEG * mcp_deg
    )

    # Below ctrl=0.07 the sim MCP is already saturated near 90° while PIP/DIP
    # continue closing. Preserve that distal phase by interpolating to the
    # calibrated full-grasp endpoint.
    if sim_ctrl < _INDEX_DISTAL_FLEX_START_CTRL:
        sim_lo = SIM_RANGE[FINGER_IDX_SIM][0]
        t = (
            (_INDEX_DISTAL_FLEX_START_CTRL - sim_ctrl)
            / (_INDEX_DISTAL_FLEX_START_CTRL - sim_lo)
        )
        t = min(max(t, 0.0), 1.0)
        start_mcp, _, _ = _interp_finger_joints(
            _INDEX_DISTAL_FLEX_START_CTRL, FINGER_IDX_SIM
        )
        start_motor = (
            _INDEX_REAL_MOTOR_OFFSET_DEG
            + _INDEX_REAL_MOTOR_PER_MCP_DEG * math.degrees(start_mcp)
        )
        motor_deg = start_motor + t * (
            ACTUATION_RANGE[FINGER_IDX_ACTUATION][1] - start_motor
        )

    lo, hi = ACTUATION_RANGE[FINGER_IDX_ACTUATION]
    return float(min(max(motor_deg, lo), hi))


def _index_real_actuation_to_sim(motor_deg: float) -> float:
    """Inverse of ``_index_real_actuation`` for actuator-derived proprio."""
    act_lo, act_hi = ACTUATION_RANGE[FINGER_IDX_ACTUATION]
    motor_deg = min(max(float(motor_deg), act_lo), act_hi)
    sim_lo, sim_hi = SIM_RANGE[FINGER_IDX_SIM]

    start_mcp, _, _ = _interp_finger_joints(
        _INDEX_DISTAL_FLEX_START_CTRL, FINGER_IDX_SIM
    )
    distal_start_motor = (
        _INDEX_REAL_MOTOR_OFFSET_DEG
        + _INDEX_REAL_MOTOR_PER_MCP_DEG * math.degrees(start_mcp)
    )
    if motor_deg >= distal_start_motor:
        t = (motor_deg - distal_start_motor) / (act_hi - distal_start_motor)
        return _INDEX_DISTAL_FLEX_START_CTRL + t * (
            sim_lo - _INDEX_DISTAL_FLEX_START_CTRL
        )

    if motor_deg <= _INDEX_REAL_MOTOR_OFFSET_DEG:
        return sim_hi

    target_mcp_deg = (
        (motor_deg - _INDEX_REAL_MOTOR_OFFSET_DEG)
        / _INDEX_REAL_MOTOR_PER_MCP_DEG
    )
    anchors = _FINGER_CTRL_JOINT_ANCHORS[FINGER_IDX_SIM]
    for i in range(len(anchors) - 1):
        ctrl_open, joints_open = anchors[i]
        ctrl_closed, joints_closed = anchors[i + 1]
        mcp_open = joints_open[0]
        mcp_closed = joints_closed[0]
        if mcp_open <= target_mcp_deg <= mcp_closed:
            span = mcp_closed - mcp_open
            t = (target_mcp_deg - mcp_open) / span if span else 0.0
            return ctrl_open + t * (ctrl_closed - ctrl_open)
    return sim_lo


def sim_to_actuation_finger_kinematic(sim_ctrl: float, finger_sim_idx: int) -> float:
    """Sim finger tendon ctrl → motor degrees via settle joints + cable model."""
    if finger_sim_idx == FINGER_IDX_SIM:
        return _index_real_actuation(sim_ctrl)

    mcp, pip, dip = _interp_finger_joints(sim_ctrl, finger_sim_idx)
    mot_rad = _JTA.finger_actuations(mcp, pip, dip)
    mot_deg = mot_rad / PI * 180.0
    ai = {
        FINGER_IDX_SIM: FINGER_IDX_ACTUATION,
        FINGER_MIDDLE_SIM: FINGER_MIDDLE_ACTUATION,
        FINGER_RING_SIM: FINGER_RING_ACTUATION,
        FINGER_PINKY_SIM: FINGER_PINKY_ACTUATION,
    }[finger_sim_idx]
    lo, hi = ACTUATION_RANGE[ai]
    return float(min(max(mot_deg, lo), hi))


#### sim to actuation ####


def sim_to_actuation_forward(
    x: float,
    lo: float,
    hi: float,
    min_u: int = ACTUATIONS_LOWER_LIMITS[0],
    max_u: int = ACTUATIONS_UPPER_LIMITS[0],
) -> float:
    """
    ctrl forward mapping: lo -> min_u, hi -> max_u
    """
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    t = (x - lo) / (hi - lo)
    return min_u + t * (max_u - min_u)


def sim_to_actuation_reverse(
    x: float,
    lo: float,
    hi: float,
    min_u: int = ACTUATIONS_LOWER_LIMITS[0],
    max_u: int = ACTUATIONS_UPPER_LIMITS[0],
) -> float:
    """
    ctrl reverse mapping: lo -> max_u, hi -> min_u
    """
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    t = (hi - x) / (hi - lo)
    return min_u + t * (max_u - min_u)


def sim_to_actuation_thumb_mcp(
    sim_abd_joint: float, sim_flex_tendon: float, sim_mcp_tendon: float
) -> float:

    joint_abd = sim_abd_joint

    joint_flex = (
        0.000344 * sim_abd_joint
        - 78.088995 * sim_flex_tendon
        + 0.188440 * sim_mcp_tendon
        + 2.977490
    )
    joint_mcp = (
        0.004162 * sim_abd_joint
        - 11.373921 * sim_flex_tendon
        - 56.722756 * sim_mcp_tendon
        + 6.666491
    )

    joint_ip = (
        0.004528469071365329 * sim_abd_joint
        - 11.422035184164583 * sim_flex_tendon
        - 56.887542891723974 * sim_mcp_tendon
        + 6.687096101625219
    )

    JTA = JointsToActuationsModel()
    JTA_abd, JTA_flex, JTA_mcp = JTA.thumb_actuations(
        joint_abd, joint_flex, joint_mcp, joint_ip
    )

    return JTA_abd / PI * 180, JTA_flex / PI * 180, JTA_mcp / PI * 180


def sim_array_to_actuation_array(sim_arr):

    actuation_arr = [0.0] * len(sim_arr)

    actuation_arr[THUMB_ABD_ACTUATION] = sim_to_actuation_forward(
        sim_arr[THUMB_ABD_SIM],
        SIM_RANGE[THUMB_ABD_SIM][0],
        SIM_RANGE[THUMB_ABD_SIM][1],
        ACTUATION_RANGE[THUMB_ABD_ACTUATION][0],
        ACTUATION_RANGE[THUMB_ABD_ACTUATION][1],
    )  # in degrees

    (
        actuation_arr[THUMB_ABD_ACTUATION],
        actuation_arr[THUMB_FLEX_ACTUATION],
        actuation_arr[THUMB_MCP_ACTUATION],
    ) = sim_to_actuation_thumb_mcp(
        sim_arr[THUMB_ABD_SIM],
        sim_arr[THUMB_FLEX_SIM],
        sim_arr[THUMB_MCP_SIM],
    )

    actuation_arr[FINGER_IDX_ACTUATION] = sim_to_actuation_finger_kinematic(
        sim_arr[FINGER_IDX_SIM], FINGER_IDX_SIM
    )
    actuation_arr[FINGER_MIDDLE_ACTUATION] = sim_to_actuation_finger_kinematic(
        sim_arr[FINGER_MIDDLE_SIM], FINGER_MIDDLE_SIM
    )

    actuation_arr[FINGER_RING_ACTUATION] = sim_to_actuation_finger_kinematic(
        sim_arr[FINGER_RING_SIM], FINGER_RING_SIM
    )
    actuation_arr[FINGER_PINKY_ACTUATION] = sim_to_actuation_finger_kinematic(
        sim_arr[FINGER_PINKY_SIM], FINGER_PINKY_SIM
    )
    return actuation_arr


#### actuation to sim ####


def actuation_to_sim_forward(
    u: int,
    lo: float,
    hi: float,
    min_u: int = ACTUATIONS_LOWER_LIMITS[0],
    max_u: int = ACTUATIONS_UPPER_LIMITS[0],
) -> float:
    """
    forward mapping:
    - min_u -> lo
    - max_u -> hi
    """
    u = max(min_u, min(max_u, int(u)))
    return lo + (u - min_u) / (max_u - min_u) * (hi - lo)


def actuation_to_sim_reverse(
    u: int,
    lo: float,
    hi: float,
    min_u: int = ACTUATIONS_LOWER_LIMITS[0],
    max_u: int = ACTUATIONS_UPPER_LIMITS[0],
) -> float:
    """
    reverse mapping:
    - min_u -> hi
    - max_u -> lo
    """
    u = max(min_u, min(max_u, int(u)))
    return hi - (u - min_u) / (max_u - min_u) * (hi - lo)


# fitting result: Actuation (cable length) ≈ -977.220399 * flex + 37.517992 + 2.5000 * abd
# fitting result: Actuation (cable length) ≈ -1241.571958 * flex + 136.590025 + 2.5000 * abd


def actuation_to_sim_thumb_cmc_flex(
    actuation_cmc_flex: float, actuation_abd: float
) -> float:

    cable = actuation_cmc_flex / 180 * PI * MOTOR_PULLEY_RADIUS

    res = ((cable - 2.5000 * actuation_abd) - 37.517992) / (-977.220399)

    return res


def actuation_to_sim_thumb_tendon(
    actuation_thumb_tendon: float, actuation_abd: float
) -> float:

    cable = actuation_thumb_tendon / 180 * PI * MOTOR_PULLEY_RADIUS

    res = ((cable - 2.5000 * actuation_abd) - 136.590025) / (-1241.571958)

    return res


def actuation_array_to_sim_array(actuation_arr):
    """
    input: [u0..u6] uint16
    output: [ctrl0..ctrl6] float
    """
    sim_arr = [0.0] * len(actuation_arr)

    sim_arr[THUMB_ABD_SIM] = actuation_to_sim_forward(
        actuation_arr[THUMB_ABD_ACTUATION],
        SIM_RANGE[THUMB_ABD_SIM][0],
        SIM_RANGE[THUMB_ABD_SIM][1],
        ACTUATION_RANGE[THUMB_ABD_ACTUATION][0],
        ACTUATION_RANGE[THUMB_ABD_ACTUATION][1],
    )

    sim_arr[THUMB_FLEX_SIM] = actuation_to_sim_thumb_cmc_flex(
        actuation_arr[THUMB_FLEX_ACTUATION],
        sim_arr[THUMB_ABD_SIM],
    )

    sim_arr[THUMB_MCP_SIM] = actuation_to_sim_thumb_tendon(
        actuation_arr[THUMB_MCP_ACTUATION],
        sim_arr[THUMB_ABD_SIM],
    )

    sim_arr[FINGER_IDX_SIM] = _index_real_actuation_to_sim(
        actuation_arr[FINGER_IDX_ACTUATION]
    )
    sim_arr[FINGER_MIDDLE_SIM] = actuation_to_sim_reverse(
        actuation_arr[FINGER_MIDDLE_ACTUATION],
        SIM_RANGE[FINGER_MIDDLE_SIM][0],
        SIM_RANGE[FINGER_MIDDLE_SIM][1],
        ACTUATION_RANGE[FINGER_MIDDLE_ACTUATION][0],
        ACTUATION_RANGE[FINGER_MIDDLE_ACTUATION][1],
    )
    sim_arr[FINGER_RING_SIM] = actuation_to_sim_reverse(
        actuation_arr[FINGER_RING_ACTUATION],
        SIM_RANGE[FINGER_RING_SIM][0],
        SIM_RANGE[FINGER_RING_SIM][1],
        ACTUATION_RANGE[FINGER_RING_ACTUATION][0],
        ACTUATION_RANGE[FINGER_RING_ACTUATION][1],
    )
    sim_arr[FINGER_PINKY_SIM] = actuation_to_sim_reverse(
        actuation_arr[FINGER_PINKY_ACTUATION],
        SIM_RANGE[FINGER_PINKY_SIM][0],
        SIM_RANGE[FINGER_PINKY_SIM][1],
        ACTUATION_RANGE[FINGER_PINKY_ACTUATION][0],
        ACTUATION_RANGE[FINGER_PINKY_ACTUATION][1],
    )
    return sim_arr
