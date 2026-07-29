"""Run a SAC policy checkpoint on the real Aero Hand (experimental).

This script loads a Brax SAC checkpoint, builds the policy, reads actuator
positions from the hand, constructs a compatible observation, and sends
actuation commands back to the hand.

IMPORTANT:
- This is a best-effort bridge from sim to real hardware.
- Start with --dry_run and very small gains.
- Run homing before execution if needed.
"""

import json
import math
import os
import pathlib
import sys
import time
import threading
from collections.abc import Mapping

# Allow running this file as a script without `pip install -e .`.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(_PROJECT_ROOT))

from absl import app
from absl import flags
from absl import logging
from etils import epath
from ml_collections import config_dict
import numpy as np
import cv2
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R

logging.set_verbosity(logging.INFO)

_CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Path to the root checkpoints directory (logs/<exp>/checkpoints).",
)
_CHECKPOINT_STEP = flags.DEFINE_integer(
    "checkpoint_step",
    -1,
    "Checkpoint step directory to load (e.g. 123456). -1 uses latest.",
)
_PORT = flags.DEFINE_string(
    "port",
    None,
    "Serial port (e.g. /dev/serial/by-id/...). If None, auto-detect.",
)
_JAX_PLATFORM = flags.DEFINE_enum(
    "jax_platform",
    "cpu",
    ["cpu", "gpu"],
    "Where to run JAX computations.",
)
_RATE_HZ = flags.DEFINE_float(
    "rate_hz",
    -1.0,
    "Control loop rate. If <= 0, uses env ctrl_dt.",
)
_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    10_000,
    "Maximum control steps to run.",
)
_DETERMINISTIC = flags.DEFINE_boolean(
    "deterministic",
    True,
    "Use deterministic actions.",
)
_DRY_RUN = flags.DEFINE_boolean(
    "dry_run",
    True,
    "Do not send commands to the hand; just print them.",
)
_DO_HOMING = flags.DEFINE_boolean(
    "do_homing",
    False,
    "Send homing command before running.",
)
_HOMING_TIMEOUT_S = flags.DEFINE_float(
  "homing_timeout_s",
  175.0,
  "Timeout for homing ACK wait (seconds). Homing can take a while.",
)
_MOVE_TO_NEUTRAL = flags.DEFINE_boolean(
  "move_to_neutral",
  True,
  "Move to neutral pose before running the policy.",
)
_NEUTRAL_DURATION_S = flags.DEFINE_float(
  "neutral_duration_s",
  1.0,
  "Seconds to ramp to neutral pose.",
)
_NEUTRAL_FINGER_DEG = flags.DEFINE_float(
    "neutral_finger_deg",
    0.0,
    "Finger flex target (degrees) for neutral pose. Applied to index/middle/ring/pinky tendons.",
)
_NEUTRAL_THUMB_DEG = flags.DEFINE_float(
    "neutral_thumb_deg",
    0.0,
    "Thumb target (degrees) for neutral pose. Applied to "
    "thumb_cmc_abd/thumb_cmc_flex/thumb_tendon. Previously the thumb was never "
    "reset to a known value by --move_to_neutral at all.",
)
# _ACT_GAIN_DEG: основной коэффициент переноса (как сильно действия в симе
# превращаются в градусы актуаторов на реальной руке).
_ACT_GAIN_DEG = flags.DEFINE_float(
    "actuation_gain_deg",
    1.0,
    "Scale sim motor targets into degrees for the SDK.",
)
_THUMB_GAIN = flags.DEFINE_float(
  "thumb_gain",
  1.0,
  "Extra gain for thumb actuators (first 3 channels).",
)
_FINGER_GAIN = flags.DEFINE_float(
  "finger_gain",
  1.0,
  "Extra gain for finger tendons (last 4 channels).",
)
# _ACT_BIAS_DEG: сдвиг (offset) для подгонки нейтрального положения.
_ACT_BIAS_DEG = flags.DEFINE_float(
    "actuation_bias_deg",
    0.0,
    "Bias added after scaling into degrees.",
)
# _MAX_ACT_STEP_DEG: ограничение на резкость команд (защита железа).
_MAX_ACT_STEP_DEG = flags.DEFINE_float(
    "max_actuation_step_deg",
    5.0,
    "Max per-step change in degrees (safety clamp).",
)
_USE_HAND_STATE = flags.DEFINE_boolean(
    "use_hand_state",
    True,
    "If true, build observation from hand actuations; else use zeros.",
)
_DEBUG_EVERY = flags.DEFINE_integer(
  "debug_every",
  50,
  "Print debug info every N control steps (0 disables).",
)
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    None,
    "Environment name override (optional).",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")

_MATCH_HOMING_DYNAMICS = flags.DEFINE_boolean(
  "match_homing_dynamics",
  False,
  "If true, configure servo speed/torque to homing-like values before running.",
)
_SERVO_SPEED = flags.DEFINE_integer(
  "servo_speed",
  None,
  "If set, calls hand.set_speed(id, servo_speed) for id=0..6 (0..32766).",
)
_SERVO_TORQUE = flags.DEFINE_integer(
  "servo_torque",
  None,
  "If set, calls hand.set_torque(id, servo_torque) for id=0..6 (0..1000).",
)


def _find_network_config_path(step_dir: epath.Path) -> epath.Path:
  p = step_dir / "config.json"
  if p.exists():
    return p
  p = step_dir / "ppo_network_config.json"
  if p.exists():
    return p
  p = step_dir / "sac_network_config.json"
  if p.exists():
    return p
  for cand in step_dir.glob("*.json"):
    try:
      data = json.loads(cand.read_text())
    except Exception:  # pylint: disable=broad-except
      continue
    if isinstance(data, dict) and "observation_size" in data and "action_size" in data:
      return cand
  raise ValueError(
      "No network config json found in checkpoint step dir: "
      f"{step_dir.as_posix()}"
  )


def _load_ppo_config_safe(config_path):
  """Workaround for a brax checkpoint.load_config bug: it does
  networks.KERNEL_INITIALIZER[None] for any *_kernel_init_fn field that was
  saved as null (e.g. mean_kernel_init_fn), raising KeyError. Only resolve
  fields that are actually set."""
  from brax.training import networks as base_networks  # pylint: disable=import-error
  from ml_collections import config_dict  # pylint: disable=import-error

  loaded_dict = json.loads(epath.Path(config_path).read_text())
  nfk = loaded_dict["network_factory_kwargs"]
  if "activation" in nfk:
    nfk["activation"] = base_networks.ACTIVATION[nfk["activation"]]
  for k in list(nfk.keys()):
    if k.endswith("kernel_init_fn") and nfk[k] is not None:
      nfk[k] = base_networks.KERNEL_INITIALIZER[nfk[k]]
  return config_dict.create(**loaded_dict)


def _latest_checkpoint_step_dir(ckpt_root: epath.Path) -> epath.Path | None:
  if not ckpt_root.exists() or not ckpt_root.is_dir():
    return None
  step_dirs: list[epath.Path] = []
  for p in ckpt_root.glob("*"):
    if p.is_dir() and p.name.isdigit():
      step_dirs.append(p)
  if not step_dirs:
    return None
  step_dirs.sort(key=lambda x: int(x.name))
  return step_dirs[-1]


def _load_env_config(ckpt_root: epath.Path, *, registry):
  env_name = _ENV_NAME.value
  env_cfg_path = ckpt_root / "config.json"
  if env_cfg_path.exists():
    env_cfg_dict = json.loads(env_cfg_path.read_text())
    env_cfg = config_dict.ConfigDict(env_cfg_dict)
    if env_name is None:
      env_name = env_cfg_dict.get("env_name", None)
    if env_name is None:
      env_name = "AeroCubeRotateZAxis"
    env_cfg["impl"] = _IMPL.value
    return env_name, env_cfg
  if env_name is None:
    env_name = "AeroCubeRotateZAxis"
  env_cfg = registry.get_default_config(env_name)
  env_cfg["impl"] = _IMPL.value
  return env_name, env_cfg


class ZedCubeTracker:
  """Фоновый трекер кубика, который считает позицию, кватернион, и линейную/угловую скорости."""
  def __init__(self):
    self.zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_fps = 30
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.sdk_gpu_id = 0
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Не удалось открыть камеру ZED")

    marker_size = 38.0
    half = marker_size / 2.0
    self.board_corners = [
        np.array([[half,  half,  half], [-half,  half,  half], [-half, half,  -half], [half, half,  -half]], dtype=np.float32),
        np.array([[half,  -half,  half], [half,  half,  half], [half, half,  -half], [half, -half,  -half]], dtype=np.float32),
        np.array([[-half,  -half,  half], [half,  -half,  half], [half, -half,  -half], [-half, -half,  -half]], dtype=np.float32),
        np.array([[-half,  half,  half], [-half,  -half,  half], [-half, -half,  -half], [-half, half,  -half]], dtype=np.float32),
        np.array([[half,  half,  half], [half,  -half,  half], [-half, half,  half], [-half, -half,  half]], dtype=np.float32),
        np.array([[-half,  half,  -half], [-half,  -half,  -half], [half, -half,  -half], [half, half,  -half]], dtype=np.float32),
    ]
    self.board_ids = np.array([17, 29, 13, 21, 25, 9], dtype=np.int32)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    self.board = cv2.aruco.Board(self.board_corners, aruco_dict, self.board_ids)

    calib_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
    self.camera_matrix = np.array([[calib_params.right_cam.fx, 0, calib_params.right_cam.cx],
                                   [0, calib_params.right_cam.fy, calib_params.right_cam.cy],
                                   [0, 0, 1]], dtype=np.float32)
    self.dist_coeffs = np.array(calib_params.right_cam.disto, dtype=np.float32)

    self.T_cam_to_base = None

    # State variables for background tracking
    self.lock = threading.Lock()
    self.running = True

    self.pos = np.zeros(3, dtype=np.float32)
    # Инициализация кватерниона [w, x, y, z] (без вращения это [1, 0, 0, 0])
    self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    self.linvel = np.zeros(3, dtype=np.float32)
    self.angvel = np.zeros(3, dtype=np.float32)

    self.last_time = time.time()

    # solvePnP pose-ambiguity mitigation: seed each frame with the previous
    # frame's rvec/tvec (useExtrinsicGuess) to bias the solver towards
    # temporal consistency, plus a jump-rejection filter as a safety net.
    # Without this, partial marker occlusion (e.g. fingers covering some
    # faces) causes solvePnP to flip between two ~equally-valid poses that
    # can be >1m apart frame-to-frame.
    self.last_rvec = None
    self.last_tvec = None
    self._JUMP_REJECT_M = 0.03  # reject a new pose if it jumps >3cm in one frame
    self._MAX_CONSECUTIVE_REJECTS = 5  # force-accept after this many rejects in a row
    self._consecutive_rejects = 0

    self._calibrate()
    self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
    self.thread.start()

  def _get_transform_matrix(self, rvec, tvec):
    Rot, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rot
    T[:3, 3] = tvec.flatten()
    return T

  def _calibrate(self):
    print("\n### КАЛИБРОВКА КАМЕРЫ ###")
    print("ПОСТАВЬТЕ КУБИК В НАЧАЛО КООРДИНАТ РОБОТА И НАЖМИТЕ 'c' В ОКНЕ ZED Tracking")
    left_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    cv2.namedWindow("Calibration - Press 'c' when ready")
    while self.T_cam_to_base is None:
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(left_image, sl.VIEW.RIGHT)
            frame = left_image.get_data()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
                
            cv2.imshow("Calibration - Press 'c' when ready", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and ids is not None and len(ids) > 0:
                objPoints, imgPoints = self.board.matchImagePoints(corners, ids)
                if objPoints is not None and len(objPoints) >= 4:
                    _, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, self.camera_matrix, self.dist_coeffs)
                    # Seed the tracking loop's first frame with this pose so
                    # useExtrinsicGuess has a sane starting point immediately.
                    self.last_rvec = rvec.copy()
                    self.last_tvec = tvec.copy()
                    T_center_in_cam = self._get_transform_matrix(rvec, tvec)
                    
                    T_base_in_center = np.eye(4)
                    T_base_in_center[2, 3] = -19.0 # Сдвиг базы на 19мм
                    
                    T_base_in_cam = np.dot(T_center_in_cam, T_base_in_center)
                    self.T_cam_to_base = np.linalg.inv(T_base_in_cam)
                    print("Калибровка успешна!")
                    cv2.destroyWindow("Calibration - Press 'c' when ready")

  def _tracking_loop(self):
    left_image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    
    while self.running:
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(left_image, sl.VIEW.RIGHT)
            frame = left_image.get_data()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            now = time.time()
            
            if ids is not None and len(ids) > 0:
                objPoints, imgPoints = self.board.matchImagePoints(corners, ids)
                if objPoints is not None and len(objPoints) >= 4:
                    # Seed with the previous frame's pose (useExtrinsicGuess).
                    # This resolves the two-fold PnP pose ambiguity that
                    # occurs when only some faces of the cube are visible
                    # (e.g. fingers occluding part of it during a grasp) by
                    # biasing the iterative solver towards the temporally
                    # consistent solution instead of letting it flip between
                    # two ~equally-valid poses frame to frame.
                    if self.last_rvec is not None:
                        success, rvec, tvec = cv2.solvePnP(
                            objPoints, imgPoints, self.camera_matrix, self.dist_coeffs,
                            rvec=self.last_rvec.copy(), tvec=self.last_tvec.copy(),
                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                    else:
                        success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, self.camera_matrix, self.dist_coeffs)
                    if success:
                        T_center_in_cam = self._get_transform_matrix(rvec, tvec)
                        T_bottom_in_center = np.eye(4)
                        T_bottom_in_center[2, 3] = -19.0
                        T_bottom_in_cam = np.dot(T_center_in_cam, T_bottom_in_center)

                        T_target_in_base = np.dot(self.T_cam_to_base, T_bottom_in_cam)

                        new_pos = T_target_in_base[:3, 3] * 0.001
                        r_mat = R.from_matrix(T_target_in_base[:3, :3])
                        # Scipy: [x,y,z,w], Mujoco: [w,x,y,z]
                        qx, qy, qz, qw = r_mat.as_quat()
                        new_quat = np.array([qw, qx, qy, qz], dtype=np.float32)

                        with self.lock:
                            # Jump-rejection safety net: even with extrinsic-guess
                            # seeding, an occasional frame can still snap to the
                            # wrong solution. Drop implausibly large single-frame
                            # jumps unless they persist (then accept, in case the
                            # cube genuinely moved fast or was repositioned).
                            jump_m = float(np.linalg.norm(new_pos - self.pos))
                            is_first_frame = self.last_rvec is None
                            if (not is_first_frame and jump_m > self._JUMP_REJECT_M
                                    and self._consecutive_rejects < self._MAX_CONSECUTIVE_REJECTS):
                                self._consecutive_rejects += 1
                                # Don't update last_rvec/tvec either, so the next
                                # frame is still seeded from the last accepted pose.
                            else:
                                self._consecutive_rejects = 0
                                dt = max(now - self.last_time, 1e-4)
                                # Считаем линейную скорость (м/с)
                                self.linvel = (new_pos - self.pos) / dt

                                # Считаем угловую скорость (приближенно из разницы кватернионов)
                                # w = 2 * (q_new * q_old^-1) / dt (упрощенно через r_new * r_old^-1)
                                r_old = R.from_quat([self.quat[1], self.quat[2], self.quat[3], self.quat[0]])
                                r_delta = r_mat * r_old.inv()
                                self.angvel = r_delta.as_rotvec() / dt

                                self.pos = new_pos
                                self.quat = new_quat
                                self.last_time = now
                                self.last_rvec = rvec.copy()
                                self.last_tvec = tvec.copy()

  def get_state(self):
    with self.lock:
        return {
            "object_pos": self.pos.copy(),
            "cube_pos": self.pos.copy(),
            "target_pos": self.pos.copy(),
            
            "object_quat": self.quat.copy(),
            "cube_quat": self.quat.copy(),
            "target_quat": self.quat.copy(),
            
            "object_linvel": self.linvel.copy(),
            "cube_linvel": self.linvel.copy(),
            
            "object_angvel": self.angvel.copy(),
            "cube_angvel": self.angvel.copy(),
        }

  def close(self):
    self.running = False
    self.thread.join()
    self.zed.close()


def _sdk_joints16_to_sim_order(sdk16):
  """Reorder SDK joint_names [thumb(4), index/middle/ring/pinky(3 each)]
  into consts.JOINT_NAMES sim order [index, middle, ring, pinky, thumb(4)]."""
  sdk16 = np.asarray(sdk16, dtype=np.float32)
  thumb = sdk16[0:4]
  index_ = sdk16[4:7]
  middle = sdk16[7:10]
  ring = sdk16[10:13]
  pinky = sdk16[13:16]
  return np.concatenate([index_, middle, ring, pinky, thumb])


def _flatten_obs(obs, batch_shape):
  parts = []
  if not isinstance(obs, Mapping):
    return obs
  for k in sorted(obs.keys()):
    x = obs[k]
    if isinstance(x, Mapping):
      x = _flatten_obs(x, batch_shape)
      parts.append(np.reshape(x, batch_shape + (-1,)))
    else:
      x = np.asarray(x)
      parts.append(np.reshape(x, batch_shape + (-1,)))
  if not parts:
    return np.zeros(batch_shape + (0,))
  return np.concatenate(parts, axis=-1)


def _build_obs_template(env, jax):
  rng = jax.random.PRNGKey(0)
  state = env.reset(rng)
  return state.obs


def _configure_servo_dynamics(
    hand,
    *,
    speed: int | None,
    torque: int | None,
):
  if speed is None and torque is None:
    return
  for servo_id in range(7):
    if speed is not None:
      try:
        resp = hand.set_speed(servo_id, int(speed))
        logging.info("Set speed: %s", resp)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to set speed for id=%d: %s", servo_id, e)
    if torque is not None:
      try:
        resp = hand.set_torque(servo_id, int(torque))
        logging.info("Set torque: %s", resp)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to set torque for id=%d: %s", servo_id, e)


_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi


def _convert_hand_actuations_to_sim_sensors(
    *,
    actuations_deg: np.ndarray,
    ref_actuations_deg: np.ndarray,
    default_tendon_sim: np.ndarray,
    motor_pulley_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Approximate sim sensors from real hand actuations.

  In sim, the 6 tendon sensors are tendon lengths (meters) and the thumb
  abduction sensor is a joint angle (radians).

  We do not know the absolute tendon length on the real hand, so we anchor the
  mapping at a reference pose: ref_actuations_deg corresponds to
  default_tendon_sim.
  """

  # Hardware (SDK) actuation order (degrees):
  #   [thumb_cmc_abd, thumb_cmc_flex, thumb_tendon, index, middle, ring, pinky]
  # Sim actuator order (motor_targets):
  #   [index, middle, ring, pinky, thumb_cmc_abd, th1_tendon, th2_tendon]

  actuations_deg = np.asarray(actuations_deg, dtype=np.float32)
  ref_actuations_deg = np.asarray(ref_actuations_deg, dtype=np.float32)
  default_tendon_sim = np.asarray(default_tendon_sim, dtype=np.float32)

  # Thumb abduction is a joint in sim (radians).
  thumb_abd_rad = np.array(
      [default_tendon_sim[4] + (actuations_deg[0] - ref_actuations_deg[0]) * _DEG_TO_RAD],
      dtype=np.float32,
  )

  # Tendon lengths in sim are meters. We approximate delta length from motor
  # rotation via pulley radius: dL = r * dtheta.
  # motor_pulley_radius_mm is from SDK (9mm).
  radius_m = float(motor_pulley_radius_mm) / 1000.0

  def _delta_len_m(delta_deg: float) -> float:
    return float(delta_deg) * _DEG_TO_RAD * radius_m

  # Map real tendon-related channels to sim tendon sensors order.
  # NOTE: thumb tendon channel mapping is assumed:
  #   hardware[1] -> sim th1, hardware[2] -> sim th2.
  idx_if = default_tendon_sim[0] + _delta_len_m(actuations_deg[3] - ref_actuations_deg[3])
  idx_mf = default_tendon_sim[1] + _delta_len_m(actuations_deg[4] - ref_actuations_deg[4])
  idx_rf = default_tendon_sim[2] + _delta_len_m(actuations_deg[5] - ref_actuations_deg[5])
  idx_pf = default_tendon_sim[3] + _delta_len_m(actuations_deg[6] - ref_actuations_deg[6])
  idx_th1 = default_tendon_sim[5] + _delta_len_m(actuations_deg[1] - ref_actuations_deg[1])
  idx_th2 = default_tendon_sim[6] + _delta_len_m(actuations_deg[2] - ref_actuations_deg[2])

  tendon_m = np.array(
      [idx_if, idx_mf, idx_rf, idx_pf, idx_th1, idx_th2], dtype=np.float32
  )
  return tendon_m, thumb_abd_rad


def _convert_sim_motor_targets_to_hand_actuations(
    *,
    motor_targets_sim: np.ndarray,
    default_tendon_sim: np.ndarray,
    ref_actuations_deg: np.ndarray,
    motor_pulley_radius_mm: float,
    actuation_gain_deg: float,
    actuation_bias_deg: float,
    thumb_gain: float,
    finger_gain: float,
) -> np.ndarray:
  """Convert sim motor targets (m/rad) to hardware actuation targets (deg).

  We compute deltas in sim space relative to default_tendon_sim, then convert:
  - tendon delta length (meters) -> delta degrees via pulley radius
  - thumb abd delta angle (radians) -> delta degrees
  and anchor them at ref_actuations_deg.
  """
  motor_targets_sim = np.asarray(motor_targets_sim, dtype=np.float32)
  default_tendon_sim = np.asarray(default_tendon_sim, dtype=np.float32)
  ref_actuations_deg = np.asarray(ref_actuations_deg, dtype=np.float32)

  # Sim deltas
  d_if_m = float(motor_targets_sim[0] - default_tendon_sim[0])
  d_mf_m = float(motor_targets_sim[1] - default_tendon_sim[1])
  d_rf_m = float(motor_targets_sim[2] - default_tendon_sim[2])
  d_pf_m = float(motor_targets_sim[3] - default_tendon_sim[3])
  d_thumb_abd_rad = float(motor_targets_sim[4] - default_tendon_sim[4])
  d_th1_m = float(motor_targets_sim[5] - default_tendon_sim[5])
  d_th2_m = float(motor_targets_sim[6] - default_tendon_sim[6])

  radius_m = float(motor_pulley_radius_mm) / 1000.0

  def _delta_deg_from_len_m(d_len_m: float) -> float:
    # dtheta = dL / r
    if radius_m <= 0:
      return 0.0
    return (d_len_m / radius_m) * _RAD_TO_DEG

  d_if_deg = _delta_deg_from_len_m(d_if_m)
  d_mf_deg = _delta_deg_from_len_m(d_mf_m)
  d_rf_deg = _delta_deg_from_len_m(d_rf_m)
  d_pf_deg = _delta_deg_from_len_m(d_pf_m)
  d_th1_deg = _delta_deg_from_len_m(d_th1_m)
  d_th2_deg = _delta_deg_from_len_m(d_th2_m)
  d_thumb_abd_deg = d_thumb_abd_rad * _RAD_TO_DEG

  # Optional scaling for tuning (applied to deltas).
  d_thumb_abd_deg *= float(actuation_gain_deg) * float(thumb_gain)
  d_th1_deg *= float(actuation_gain_deg) * float(thumb_gain)
  d_th2_deg *= float(actuation_gain_deg) * float(thumb_gain)

  d_if_deg *= float(actuation_gain_deg) * float(finger_gain)
  d_mf_deg *= float(actuation_gain_deg) * float(finger_gain)
  d_rf_deg *= float(actuation_gain_deg) * float(finger_gain)
  d_pf_deg *= float(actuation_gain_deg) * float(finger_gain)

  # Build hardware actuation vector (degrees):
  #   [thumb_cmc_abd, thumb_cmc_flex, thumb_tendon, index, middle, ring, pinky]
  out = ref_actuations_deg.copy().astype(np.float32)
  out[0] = ref_actuations_deg[0] + d_thumb_abd_deg

  # NOTE: mapping assumption:
  #   hardware[1] corresponds to sim th1 tendon, hardware[2] to sim th2 tendon.
  out[1] = ref_actuations_deg[1] + d_th1_deg
  out[2] = ref_actuations_deg[2] + d_th2_deg

  out[3] = ref_actuations_deg[3] + d_if_deg
  out[4] = ref_actuations_deg[4] + d_mf_deg
  out[5] = ref_actuations_deg[5] + d_rf_deg
  out[6] = ref_actuations_deg[6] + d_pf_deg

  out += float(actuation_bias_deg)
  return out


def _verify_sim_real_roundtrip(
    *,
    default_tendon_sim: np.ndarray,
    action_scale: np.ndarray,
    ref_actuations_deg: np.ndarray,
    motor_pulley_radius_mm: float,
    actuation_lower_limits: np.ndarray,
    actuation_upper_limits: np.ndarray,
    tol_deg: float = 0.05,
) -> None:
  """Guarantee (not just assume) that sim<->real values are mixed in
  correctly: `_convert_sim_motor_targets_to_hand_actuations` and
  `_convert_hand_actuations_to_sim_sensors` must be exact inverses of each
  other for the SAME (default_tendon_sim, ref_actuations_deg, radius) used at
  runtime. This is exactly the kind of thing that silently breaks if a
  channel-ordering or sign convention drifts (as happened earlier with the
  SDK/sim joint-order mismatch) -- a visual/manual check would not catch it
  reliably, so this runs automatically on every startup.

  Raises RuntimeError if the round-trip error exceeds `tol_deg`.
  """
  rng = np.random.RandomState(0)
  test_actions = [
      np.zeros(7, dtype=np.float32),
      np.full(7, 1.0, dtype=np.float32),
      np.full(7, -1.0, dtype=np.float32),
      rng.uniform(-1.0, 1.0, size=7).astype(np.float32),
      rng.uniform(-1.0, 1.0, size=7).astype(np.float32),
  ]

  worst_err = 0.0
  for act in test_actions:
    motor_targets_sim = default_tendon_sim + act * action_scale

    # Forward: sim motor targets -> hardware degrees (gain=1, no tuning bias,
    # so this tests the core geometric conversion, not operator knobs).
    act_cmd_deg = _convert_sim_motor_targets_to_hand_actuations(
        motor_targets_sim=motor_targets_sim,
        default_tendon_sim=default_tendon_sim,
        ref_actuations_deg=ref_actuations_deg,
        motor_pulley_radius_mm=motor_pulley_radius_mm,
        actuation_gain_deg=1.0,
        actuation_bias_deg=0.0,
        thumb_gain=1.0,
        finger_gain=1.0,
    )
    # Clip to what the real hand could physically accept, same as the main
    # loop does -- round-trip is only meaningful within reachable range.
    act_cmd_deg_clipped = np.clip(act_cmd_deg, actuation_lower_limits, actuation_upper_limits)

    # Backward: hardware degrees -> sim sensor units (the same function used
    # every control step to build `state` from real actuations).
    tendon_m, thumb_abd_rad = _convert_hand_actuations_to_sim_sensors(
        actuations_deg=act_cmd_deg_clipped,
        ref_actuations_deg=ref_actuations_deg,
        default_tendon_sim=default_tendon_sim,
        motor_pulley_radius_mm=motor_pulley_radius_mm,
    )
    # Reassemble into sim order [index, middle, ring, pinky, thumb_abd, th1, th2].
    recovered_sim = np.array([
        tendon_m[0], tendon_m[1], tendon_m[2], tendon_m[3],
        thumb_abd_rad[0], tendon_m[4], tendon_m[5],
    ], dtype=np.float32)

    # Only compare channels that weren't clipped away from their forward
    # target -- a channel pinned at a hardware limit is *expected* to diverge
    # from the unclipped sim target, that's the limit doing its job, not a
    # round-trip bug.
    # `not_clipped` is in HARDWARE order [thumb_abd, th1, th2, index, middle,
    # ring, pinky]; reorder it into SIM order [index, middle, ring, pinky,
    # thumb_abd, th1, th2] to match `recovered_sim`/`motor_targets_sim` --
    # comparing them index-for-index without reordering was itself a bug (the
    # same class of channel-order mismatch this test exists to catch).
    not_clipped_hw = np.isclose(act_cmd_deg, act_cmd_deg_clipped, atol=1e-4)
    not_clipped = np.array([
        not_clipped_hw[3], not_clipped_hw[4], not_clipped_hw[5], not_clipped_hw[6],
        not_clipped_hw[0], not_clipped_hw[1], not_clipped_hw[2],
    ])

    radius_m = float(motor_pulley_radius_mm) / 1000.0
    err_raw = np.abs(recovered_sim - motor_targets_sim)
    # Sim-order units differ per channel: index/middle/ring/pinky/th1/th2 (idx
    # 0,1,2,3,5,6) are tendon lengths in METERS (-> deg via pulley radius);
    # thumb_abd (idx 4) is a joint angle in RADIANS (-> deg directly). Mixing
    # these up (dividing the radian error by radius_m too) was the second bug.
    err_deg = np.array(err_raw, dtype=np.float64)
    tendon_idx = [0, 1, 2, 3, 5, 6]
    err_deg[tendon_idx] = err_raw[tendon_idx] / radius_m * _RAD_TO_DEG
    err_deg[4] = err_raw[4] * _RAD_TO_DEG
    err_deg = np.where(not_clipped, err_deg, 0.0)
    worst_err = max(worst_err, float(np.max(err_deg)))

  if worst_err > tol_deg:
    raise RuntimeError(
        f"Sim<->real round-trip self-test FAILED: worst-case error "
        f"{worst_err:.4f} deg exceeds tolerance {tol_deg} deg. The forward "
        f"(_convert_sim_motor_targets_to_hand_actuations) and backward "
        f"(_convert_hand_actuations_to_sim_sensors) conversions are not "
        f"consistent for the current default_tendon_sim/ref_actuations_deg/"
        f"motor_pulley_radius_mm -- do NOT trust the observation built from "
        f"real hardware until this is fixed."
    )
  logging.info(
      "Sim<->real round-trip self-test PASSED (worst-case error %.4f deg, "
      "tolerance %.2f deg) across %d test action vectors.",
      worst_err, tol_deg, len(test_actions),
  )


def _make_obs_from_hand(
    *,
    obs_template,
    obs_history,
    last_act,
    actuations_deg,
    ref_actuations_deg,
    default_tendon_sim,
    motor_pulley_radius_mm,
    cube_state=None,
    tendon_noise_scale=0.0,
    joint_noise_scale=0.0,
    noise_level=1.0,
    joint_angles_rad_sim_order=None,
    joint_qvel_sim_order=None,
):
  if actuations_deg is None:
    tendon_m = np.zeros(6, dtype=np.float32)
    thumb_abd_rad = np.zeros(1, dtype=np.float32)
  else:
    tendon_m, thumb_abd_rad = _convert_hand_actuations_to_sim_sensors(
        actuations_deg=np.asarray(actuations_deg, dtype=np.float32),
        ref_actuations_deg=np.asarray(ref_actuations_deg, dtype=np.float32),
        default_tendon_sim=np.asarray(default_tendon_sim, dtype=np.float32),
        motor_pulley_radius_mm=float(motor_pulley_radius_mm),
    )

  # Match the synthetic sensor noise injected at train time (see rotate_z.py
  # _get_obs). On real hardware the tendon readings become bit-for-bit
  # static once the fingers stop physically moving, which (unlike sim, where
  # this noise keeps the observation from ever being exactly constant)
  # collapses the policy into a repeating, frozen action. Re-adding the same
  # noise distribution here keeps the real observation in-distribution.
  if tendon_noise_scale > 0.0:
    tendon_m = tendon_m + (
        (2.0 * np.random.uniform(size=tendon_m.shape) - 1.0)
        * float(noise_level) * float(tendon_noise_scale)
    ).astype(np.float32)
  if joint_noise_scale > 0.0:
    thumb_abd_rad = thumb_abd_rad + (
        (2.0 * np.random.uniform(size=thumb_abd_rad.shape) - 1.0)
        * float(noise_level) * float(joint_noise_scale)
    ).astype(np.float32)

  state_vec = np.concatenate([tendon_m, thumb_abd_rad, last_act], axis=0)
  # История наблюдений (как в симе): сдвиг и вставка нового состояния.
  state_size = state_vec.shape[0]
  obs_history = np.roll(obs_history, state_size)
  obs_history[:state_size] = state_vec

  if isinstance(obs_template, Mapping):
    obs = {}
    for k, v in obs_template.items():
      if k == "state":
        obs[k] = obs_history.astype(np.float32)
      elif k == "privileged_state":
        # Reconstruct the same layout as rotate_z.py's _get_obs:
        #   [state(14), joint_angles(16), qvel(16), joint_torques(7),
        #    fingertip_positions(15), cube_pos_error(3), cube_quat(4),
        #    cube_angvel(3), cube_linvel(3)]
        # joint_angles/qvel come from the real hand (see main loop); the
        # cube_* terms are zeroed to match how they were zeroed at train
        # time too. joint_torques/fingertip_positions are a known gap
        # (get_forward_kinematics() is unimplemented in the SDK) and are
        # left at zero rather than guessing a miscalibrated substitute.
        if joint_angles_rad_sim_order is not None:
          real_joint_angles = np.asarray(joint_angles_rad_sim_order, dtype=np.float32)
        else:
          real_joint_angles = np.zeros(16, dtype=np.float32)
        if joint_qvel_sim_order is not None:
          real_qvel = np.asarray(joint_qvel_sim_order, dtype=np.float32)
        else:
          real_qvel = np.zeros(16, dtype=np.float32)
        priv = np.concatenate([
            state_vec.astype(np.float32),          # 14
            real_joint_angles,                       # 16 (real)
            real_qvel,                                # 16 (real, finite-diff)
            np.zeros(7, dtype=np.float32),            # joint_torques (gap)
            np.zeros(15, dtype=np.float32),           # fingertip_positions (gap)
            np.zeros(3, dtype=np.float32),            # cube_pos_error (zeroed at train time too)
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # cube_quat
            np.zeros(3, dtype=np.float32),            # cube_angvel
            np.zeros(3, dtype=np.float32),            # cube_linvel
        ])
        if priv.shape == v.shape:
          obs[k] = priv
        else:
          logging.warning(
              "privileged_state shape mismatch: built %s, expected %s; zeroing.",
              priv.shape, v.shape,
          )
          obs[k] = np.zeros_like(v)
      elif cube_state is not None and k in cube_state:
        # Интеграция данных с камеры: подменяем np.zeros_like на реальные данные
        # с правильным shape, если такой ключ возвращается камерой.
        raw_val = np.asarray(cube_state[k], dtype=np.float32)
        if raw_val.shape == v.shape:
          obs[k] = raw_val
        else:
          # Защита от размерностей: если формат отличается, заполним нулями
          obs[k] = np.zeros_like(v)
      else:
        obs[k] = np.zeros_like(v)
    return obs, obs_history

  return obs_history.astype(np.float32), obs_history


def main(argv):
  del argv

  if _CHECKPOINT_DIR.value is None:
    raise flags.ValidationError("--checkpoint_dir is required")

  if bool(_DRY_RUN.value):
    logging.warning(
        "DRY RUN is enabled (--dry_run=true). Commands will NOT be sent to the hand. "
        "Use --dry_run=false to move the hardware."
    )

  if _JAX_PLATFORM.value == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

  import jax  # pylint: disable=import-error
  from brax.training import checkpoint  # pylint: disable=import-error
  from brax.training.agents.ppo import networks as ppo_networks  # pylint: disable=import-error
  from ml_collections import config_dict  # pylint: disable=import-error
  from mujoco_playground import registry  # pylint: disable=import-error
  from aero_open_sdk.aero_hand import AeroHand  # pylint: disable=import-error
  try:
    from aero_open_sdk.joints_to_actuations import MOTOR_PULLEY_RADIUS as _SDK_MOTOR_PULLEY_RADIUS  # pylint: disable=import-error
  except Exception:  # pylint: disable=broad-except
    _SDK_MOTOR_PULLEY_RADIUS = 9.0

  ckpt_root = epath.Path(_CHECKPOINT_DIR.value).resolve()
  if int(_CHECKPOINT_STEP.value) >= 0:
    step_dir = ckpt_root / str(int(_CHECKPOINT_STEP.value))
  else:
    step_dir = _latest_checkpoint_step_dir(ckpt_root)
  if step_dir is None or not step_dir.exists():
    raise ValueError(f"No checkpoints found in {ckpt_root}")
  logging.info("Using checkpoint step dir: %s", step_dir)

  env_name, env_cfg = _load_env_config(ckpt_root, registry=registry)
  env = registry.load(env_name, config=env_cfg)

  # Берем сим-скейлы для формирования motor_targets (как в rotate_z.py).
  action_scale = np.asarray(getattr(env._config, "action_scale", [1.0] * 7))
  default_tendon = getattr(env, "_default_tendon", np.zeros(7))
  default_tendon = np.asarray(default_tendon, dtype=np.float32)

  # Загрузка сети PPO из чекпоинта (обход бага load_config на null-полях).
  ckpt_config_path = _find_network_config_path(step_dir)
  ckpt_cfg = _load_ppo_config_safe(ckpt_config_path)
  networks = checkpoint.get_network(ckpt_cfg, ppo_networks.make_ppo_networks)
  make_policy = ppo_networks.make_inference_fn(networks)
  params = checkpoint.load(step_dir)
  inference_fn = make_policy(params, deterministic=bool(_DETERMINISTIC.value))
  jit_inference_fn = jax.jit(inference_fn)


  # Шаблон наблюдения нужен, чтобы сохранить формат (dict/flat) как в симе.
  obs_template = _build_obs_template(env, jax)
  history_len = int(getattr(env._config, "history_len", 1))
  obs_history = np.zeros(history_len * 14, dtype=np.float32)
  last_act = np.zeros(7, dtype=np.float32)

  # Reproduce the same synthetic observation noise used at train time (see
  # rotate_z.py _get_obs) so the real-hardware input distribution matches
  # what the policy was trained on. Without this, tendon readings become
  # exactly static once the fingers stop moving and the policy collapses
  # into a repeating, frozen action.
  noise_cfg = getattr(env._config, "noise_config", None)
  obs_noise_level = float(getattr(noise_cfg, "level", 0.0)) if noise_cfg else 0.0
  obs_tendon_noise_scale = (
      float(getattr(noise_cfg.scales, "tendon_length", 0.0))
      if noise_cfg and hasattr(noise_cfg, "scales") else 0.0
  )
  obs_joint_noise_scale = (
      float(getattr(noise_cfg.scales, "joint_pos", 0.0))
      if noise_cfg and hasattr(noise_cfg, "scales") else 0.0
  )
  logging.info(
      "Obs noise (matching training): level=%.3f tendon_scale=%.4f joint_scale=%.4f",
      obs_noise_level, obs_tendon_noise_scale, obs_joint_noise_scale,
  )
  
  # === Интеграция ZED камеры ===
  # Создаем трекер. Он заблокирует выполнение до тех пор, пока вы не нажмете 'c' 
  # для калибровки в окне OpenCV. После этого он запустит фоновый поток обновления 
  # координат (чтобы не тормозить RL-управление рукой).
  cube_tracker = ZedCubeTracker()

  # Подключение к руке.
  hand = AeroHand(port=_PORT.value)
  motor_pulley_radius_mm = float(_SDK_MOTOR_PULLEY_RADIUS)
  logging.info(
      "Hand actuation limits (deg): lower=%s upper=%s",
      np.round(np.asarray(hand.actuation_lower_limits, dtype=np.float32), 3),
      np.round(np.asarray(hand.actuation_upper_limits, dtype=np.float32), 3),
  )
  if _DO_HOMING.value:
    logging.info(
        "Sending homing command... (can take up to %.1fs)",
        float(_HOMING_TIMEOUT_S.value),
    )
    hand.send_homing(timeout_s=float(_HOMING_TIMEOUT_S.value))

  # Configure servo dynamics (speed/torque) to better match homing behavior.
  # Homing uses WritePosEx(..., speed=2400, ..., torque~1000..1023) in firmware.
  if _DRY_RUN.value and (
      _MATCH_HOMING_DYNAMICS.value
      or _SERVO_SPEED.value is not None
      or _SERVO_TORQUE.value is not None
  ):
    logging.warning(
        "--dry_run=true: skipping servo speed/torque configuration. "
        "Use --dry_run=false to apply --servo_speed/--servo_torque."
    )
  else:
    speed = _SERVO_SPEED.value
    torque = _SERVO_TORQUE.value
    if _MATCH_HOMING_DYNAMICS.value:
      if speed is None:
        speed = 2400
      if torque is None:
        torque = 1000
    _configure_servo_dynamics(hand, speed=speed, torque=torque)

  # Частота управления: влияет на плавность и безопасность.
  if float(_RATE_HZ.value) > 0:
    dt = 1.0 / float(_RATE_HZ.value)
  else:
    dt = float(getattr(env._config, "ctrl_dt", 0.05))

  effective_hz = 1.0 / max(dt, 1e-9)
  max_steps = int(_MAX_STEPS.value)
  logging.info(
      "Control loop: dt=%.4fs (%.1f Hz), max_steps=%d (~%.1f s)",
      dt,
      effective_hz,
      max_steps,
      max_steps * dt,
  )
  if float(_RATE_HZ.value) > 0 and float(_RATE_HZ.value) > 60.0:
    logging.warning(
        "--rate_hz=%.1f is quite high; serial I/O (get_actuations/set_actuations) "
        "may stall or the hand may appear to freeze. Consider <=50Hz or omit --rate_hz "
        "to use env ctrl_dt.",
        float(_RATE_HZ.value),
    )

  rng = jax.random.PRNGKey(0)
  last_cmd = None
  last_actuations_deg = None
  last_actuations_warn_t = 0.0
  last_joint_angles_rad = None

  # Reference pose mapping: we will treat the current hand pose as the
  # real-world equivalent of sim default_tendon.
  ref_actuations_deg = None

  # Возврат в нейтральное положение перед стартом.
  if _MOVE_TO_NEUTRAL.value:
    # Neutral pose is defined in *hardware degrees* for safety.
    start_cmd = hand.get_actuations()
    if start_cmd is None:
      start_cmd = np.zeros(7, dtype=np.float32)
    start_cmd = np.asarray(start_cmd, dtype=np.float32)

    # Always apply both targets explicitly (do NOT gate on != 0.0: 0.0 is a
    # legitimate, intended target -- the previous `!= 0.0` check silently
    # made --move_to_neutral a no-op whenever the default 0.0 was used, and
    # never touched the thumb channels at all).
    neutral_cmd = start_cmd.copy()
    neutral_cmd[0:3] = float(_NEUTRAL_THUMB_DEG.value)
    neutral_cmd[3:7] = float(_NEUTRAL_FINGER_DEG.value)
    neutral_cmd = np.clip(
        neutral_cmd, hand.actuation_lower_limits, hand.actuation_upper_limits
    )
    if _DRY_RUN.value:
      logging.info(
          "Neutral cmd (deg)=%s (not sent because --dry_run=true)",
          np.round(neutral_cmd, 3),
      )
    else:
      ramp_steps = max(1, int(float(_NEUTRAL_DURATION_S.value) / dt))
      for i in range(ramp_steps):
        t = (i + 1) / ramp_steps
        cmd = start_cmd + t * (neutral_cmd - start_cmd)
        hand.set_actuations(cmd.tolist())
        time.sleep(dt)
      last_cmd = neutral_cmd

  # Capture reference actuations after neutral move.
  ref_actuations_deg = hand.get_actuations()
  if ref_actuations_deg is None:
    # Fall back to last_cmd (if any) or zeros.
    if last_cmd is not None:
      ref_actuations_deg = np.asarray(last_cmd, dtype=np.float32)
    else:
      ref_actuations_deg = np.zeros(7, dtype=np.float32)
  else:
    ref_actuations_deg = np.asarray(ref_actuations_deg, dtype=np.float32)

  # --- Reference-point (anchor) diagnostic ---------------------------------
  # The whole sim->real conversion is a DELTA scheme: motor_targets are
  # expressed relative to default_tendon_sim (the sim's fixed home pose), and
  # then re-anchored onto ref_actuations_deg (whatever the real hand's degrees
  # happen to be right now). Nothing here proves that point is physically
  # equivalent to the sim's home pose -- it's only as good as the neutral
  # move actually achieving its commanded target (verified below) plus the
  # (unverified) assumption that hardware's homed/neutral pose corresponds to
  # the sim's home keyframe. See the model card / README for the fuller
  # writeup; this just surfaces drift so bad anchors don't fail silently.
  if bool(_MOVE_TO_NEUTRAL.value):
    commanded_neutral = np.array(
        [float(_NEUTRAL_THUMB_DEG.value)] * 3 + [float(_NEUTRAL_FINGER_DEG.value)] * 4,
        dtype=np.float32,
    )
    anchor_dev = ref_actuations_deg - commanded_neutral
    max_dev = float(np.max(np.abs(anchor_dev)))
    logging.info(
        "Reference pose (ref_actuations_deg, deg) = %s | commanded neutral = %s | "
        "max deviation from commanded target = %.2f deg",
        np.round(ref_actuations_deg, 3), commanded_neutral, max_dev,
    )
    if max_dev > 5.0:
      logging.warning(
          "Reference pose is >%.1fdeg away from the commanded neutral target "
          "(%.2fdeg). The hand may not have physically reached neutral (serial "
          "read/write issue, mechanical resistance, or --dry_run=true skipped "
          "the actual move) -- the sim<->real anchor for this run may be "
          "unreliable and results may not compare against other runs.",
          5.0, max_dev,
      )
  if not bool(_DO_HOMING.value):
    logging.warning(
        "--do_homing=false: the reference pose was NOT established via a "
        "calibrated hardware homing routine, only via --move_to_neutral (a "
        "software target, not a verified mechanical zero). If behavior is "
        "inconsistent between sessions, re-run with --do_homing=true first."
    )

  # Guarantee (not assume) that sim<->real values mix in correctly: the
  # forward and backward conversion functions must be exact inverses for the
  # actual default_tendon/action_scale/ref_actuations_deg this run is using.
  # Raises and aborts before touching the hand if this ever fails.
  _verify_sim_real_roundtrip(
      default_tendon_sim=default_tendon,
      action_scale=action_scale,
      ref_actuations_deg=ref_actuations_deg,
      motor_pulley_radius_mm=motor_pulley_radius_mm,
      actuation_lower_limits=np.asarray(hand.actuation_lower_limits, dtype=np.float32),
      actuation_upper_limits=np.asarray(hand.actuation_upper_limits, dtype=np.float32),
  )

  logging.info("Running policy on hand (dry_run=%s)", _DRY_RUN.value)
  # Основной цикл: наблюдение -> политика -> команда -> отправка.
  for step_idx in range(int(_MAX_STEPS.value)):
    actuations = hand.get_actuations() if _USE_HAND_STATE.value else None
    if _USE_HAND_STATE.value:
      if actuations is None:
        # Serial read may time out; keep running using last known state.
        now = time.monotonic()
        if now - last_actuations_warn_t > 1.0:
          logging.warning(
              "get_actuations() timed out; using last known actuations for obs"
          )
          last_actuations_warn_t = now
        actuations = last_actuations_deg
      else:
        last_actuations_deg = np.asarray(actuations, dtype=np.float32)

    # Reconstruct real joint_angles (16, sim JOINT_NAMES order, radians) via
    # the SDK's real kinematic coupling model, plus a finite-diff qvel — see
    # privileged_state discussion: these were being silently zeroed before.
    if actuations is not None:
      act_rad = [float(a) * _DEG_TO_RAD for a in actuations]
      compact7_rad = hand.actuations_to_joints_model.hand_joints(act_rad)
      sdk16_rad = hand.convert_seven_joints_to_sixteen(compact7_rad)
      joint_angles_rad_sim_order = _sdk_joints16_to_sim_order(sdk16_rad)
    elif last_joint_angles_rad is not None:
      joint_angles_rad_sim_order = last_joint_angles_rad
    else:
      joint_angles_rad_sim_order = np.zeros(16, dtype=np.float32)

    if last_joint_angles_rad is not None:
      joint_qvel_sim_order = (joint_angles_rad_sim_order - last_joint_angles_rad) / dt
    else:
      joint_qvel_sim_order = np.zeros(16, dtype=np.float32)
    last_joint_angles_rad = joint_angles_rad_sim_order

    obs, obs_history = _make_obs_from_hand(
        obs_template=obs_template,
        obs_history=obs_history,
        last_act=last_act,
        actuations_deg=actuations,
        ref_actuations_deg=ref_actuations_deg,
        default_tendon_sim=default_tendon,
        motor_pulley_radius_mm=motor_pulley_radius_mm,
        cube_state=cube_tracker.get_state(),
        tendon_noise_scale=obs_tendon_noise_scale,
        joint_noise_scale=obs_joint_noise_scale,
        joint_angles_rad_sim_order=joint_angles_rad_sim_order,
        joint_qvel_sim_order=joint_qvel_sim_order,
        noise_level=obs_noise_level,
    )

    # PPO actor consumes the obs dict directly (policy_obs_key="state"
    # internally), unlike SAC which needed the full flattened concatenation.
    rng, act_key = jax.random.split(rng)
    act = np.asarray(jit_inference_fn(obs, act_key)[0])
    act = np.clip(act, -1.0, 1.0)

    # Перенос sim->real:
    # 1) motor_targets как в симе (tendon lengths in meters + thumb abd in radians),
    # 2) convert deltas into hardware actuator degrees using pulley radius,
    # 3) anchor at ref_actuations_deg.
    motor_targets = default_tendon + act * action_scale
    act_cmd = _convert_sim_motor_targets_to_hand_actuations(
        motor_targets_sim=motor_targets,
        default_tendon_sim=default_tendon,
        ref_actuations_deg=ref_actuations_deg,
      motor_pulley_radius_mm=motor_pulley_radius_mm,
        actuation_gain_deg=float(_ACT_GAIN_DEG.value),
        actuation_bias_deg=float(_ACT_BIAS_DEG.value),
        thumb_gain=float(_THUMB_GAIN.value),
        finger_gain=float(_FINGER_GAIN.value),
    )

    act_cmd_raw = np.asarray(act_cmd, dtype=np.float32)

    # Калибровка лимитов: защищает от выхода за пределы актуаторов.
    act_cmd_clipped = np.clip(
        act_cmd_raw, hand.actuation_lower_limits, hand.actuation_upper_limits
    ).astype(np.float32)
    clipped_mask = ~np.isclose(act_cmd_raw, act_cmd_clipped, atol=1e-4)

    # Лимит скорости изменения (сильно влияет на стабильность на реальном железе).
    if last_cmd is not None and float(_MAX_ACT_STEP_DEG.value) > 0:
      max_step = float(_MAX_ACT_STEP_DEG.value)
      delta = np.clip(act_cmd_clipped - last_cmd, -max_step, max_step)
      act_cmd_final = (last_cmd + delta).astype(np.float32)
      slew_mask = ~np.isclose(act_cmd_clipped, act_cmd_final, atol=1e-4)
    else:
      act_cmd_final = act_cmd_clipped
      slew_mask = np.zeros_like(act_cmd_final, dtype=bool)

    act_cmd = act_cmd_final

    if int(_DEBUG_EVERY.value) > 0 and step_idx % int(_DEBUG_EVERY.value) == 0:
      if np.any(clipped_mask):
        logging.warning(
            "act_cmd clipped to limits (idx=%s) raw=%s clipped=%s",
            np.where(clipped_mask)[0].tolist(),
            np.round(act_cmd_raw, 3),
            np.round(act_cmd_clipped, 3),
        )
      if np.any(slew_mask):
        logging.warning(
            "act_cmd slew-limited (idx=%s) clipped=%s final=%s max_step_deg=%.3f",
            np.where(slew_mask)[0].tolist(),
            np.round(act_cmd_clipped, 3),
            np.round(act_cmd, 3),
            float(_MAX_ACT_STEP_DEG.value),
        )

    if _DRY_RUN.value:
      if step_idx % 20 == 0:
        logging.info("act_cmd(deg)=%s", np.round(act_cmd, 3))
    else:
      hand.set_actuations(act_cmd.tolist())

    if int(_DEBUG_EVERY.value) > 0 and step_idx % int(_DEBUG_EVERY.value) == 0:
      logging.info(
          "step=%d act=%s motor_targets=%s act_cmd=%s actuations=%s",
          step_idx,
          np.round(act, 3),
          np.round(motor_targets, 3),
          np.round(act_cmd, 3),
          "None" if actuations is None else np.round(np.asarray(actuations, dtype=np.float32), 3),
      )
      
      # Вывод текущих данных позиционирования кубика
      tracker_state = cube_tracker.get_state()
      r_mat = R.from_quat([tracker_state["cube_quat"][1], tracker_state["cube_quat"][2], tracker_state["cube_quat"][3], tracker_state["cube_quat"][0]])
      euler_deg = r_mat.as_euler('xyz', degrees=True)
      logging.info(
          "CUBE TRACKING | Pos(X,Y,Z): %s | Rot(XYZ deg): [%.1f, %.1f, %.1f]",
          np.round(tracker_state["cube_pos"], 3),
          euler_deg[0], euler_deg[1], euler_deg[2]
      )

    last_cmd = act_cmd
    last_act = act.astype(np.float32)
    time.sleep(dt)

  cube_tracker.close()
  hand.close()
  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
