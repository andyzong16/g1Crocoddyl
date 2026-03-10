"""
Batch generation script for two-step motion dataset.

Generates a single NPZ file containing trajectories with two consecutive steps:
- Each trajectory has: wait_before + step1 + wait_mid + step2 + wait_after

Dataset fields:
- q: (n, nq) - joint positions with base
- qd: (n, nv) - joint velocities
- T_blf: (n, 4) - body frame to left foot frame transform (x, y, z, yaw)
- T_brf: (n, 4) - body frame to right foot frame transform (x, y, z, yaw)
- T_stsw: (n, 4) - stance foot to swing foot transform (x, y, z, yaw)
- p_wcom: (n, 3) - CoM position in world frame
- T_wbase: (n, 7) - base transform in world frame (x, y, z, qw, qx, qy, qz)
- v_b: (n, 6) - base velocity in base frame (linear xyz, angular xyz)
- cmd_footstep: (n, 4) - [x, y, sin(yaw), cos(yaw)] in stance foot frame
- cmd_stance: (n, 1) - 0=left stance, 1=right stance
- cmd_countdown: (n, 1) - countdown timer: 0 during wait, 1->0 during step
- traj: (k,) - starting indices of each trajectory
- traj_dt: float - time step between frames
"""

import numpy as np
import pinocchio
import crocoddyl
import matplotlib.pyplot as plt
from step import SimpleBipedGaitProblem
import psutil
import gc
import os
try:
    import meshcat
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "stepping_dataset.npz")
TIME_STEP = 0.02
STEP_KNOTS = 20
SUPPORT_KNOTS = 20  # Increased from 10 for smoother and more accurate COM transitions
TRANSITION_KNOTS = 20  # Knots for post-swing COM centering phase
COM_SHIFT_RATIO = 0.5  # Ratio of COM shift towards center during swing (0.8 = 80%)
INITIAL_COM_SHIFT = 0.5  # Ratio of COM shift towards stance foot in initial phase (0.8->0.9 for more shift)
WITHDISPLAY = True
PLOT = 0
CHECKPOINT_FREQUENCY = 0  # Save checkpoint every N successful trajectories (0 to disable)

# Step generation parameters
STEP_HEIGHT = 0.125  # Step height in meters
WAIT_TIME_RANGE = (0.8, 1.0)  # Waiting period before step (seconds)
MID_WAIT_TIME_RANGE = (0.3, 0.6)  # Waiting period between two steps (seconds)
# Grid sampling parameters
GRID_X_STEPS = 5  # Number of steps in x direction
GRID_Y_STEPS = 3  # Number of steps in y direction
GRID_YAW_STEPS = 5  # Number of steps in yaw direction
X_STEP_UNIT = 0.1
Y_STEP_UNIT = 0.05
Y_OFFSET = 0.17
YAW_STEP_UNIT = 0.2

# Solver parameters
MAX_ITERATIONS = 600  # Increased for better convergence with higher accuracy requirements
SOLVER_THRESHOLD = 1e-5  # Tightened threshold for more precise solutions


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


def load_robot():
    """Load the robot model and set up initial configuration."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "model", "T1_7dof_arms_with_gripper.urdf")
    package_dir = os.path.join(script_dir, "model")

    robot = pinocchio.RobotWrapper.BuildFromURDF(
        urdf_path,
        package_dirs=[package_dir],
        root_joint=pinocchio.JointModelFreeFlyer(),
    )

    half_sitting = np.array(
        [
            0,
            0,
            0.655,  # base position (updated for COM height ~0.57m)
            0,
            0,
            0,
            1,  # base orientation (quaternion)
            0,
            0,  # torso joints
            0.2,
            -1.35,
            0,
            -0.5,
            0.0,
            0.0,
            0.0,  # left arm
            0.2,
            1.35,
            0,
            0.5,
            0.0,
            0.0,
            0.0,  # right arm
            0,  # head
            -0.3,  # left hip pitch
            0.007658,
            0,
            0.6,  # left knee
            -0.3,  # left ankle pitch
            0,  # left leg
            -0.3,  # right hip pitch
            -0.007658,
            0,
            0.6,  # right knee
            -0.3,  # right ankle pitch
            0,  # right leg
        ]
    )
    robot.model.referenceConfigurations["half_sitting"] = half_sitting

    return robot


def rotation_matrix_to_yaw(R):
    """Extract yaw angle from rotation matrix."""
    return np.arctan2(R[1, 0], R[0, 0])


def transform_to_stance_frame(target_pos, stance_pos, stance_R, swing_R):
    """
    Transform target position from world frame to stance foot frame.

    Args:
        target_pos: [x, y, z] swing foot position in world frame
        stance_pos: [x, y, z] stance foot position in world frame
        stance_R: 3x3 rotation matrix of stance foot in world frame
        swing_R: 3x3 rotation matrix of swing foot in world frame

    Returns:
        [x, y, z, yaw] in stance foot frame
    """
    # Transform position to stance frame
    p_world = target_pos - stance_pos
    p_stance = stance_R.T @ p_world

    # Compute yaw difference between swing and stance foot
    swing_stance_R = stance_R.T @ swing_R
    swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)

    return np.array([p_stance[0], p_stance[1], 0.0, swing_stance_yaw])


def generate_grid_samples(
    lfPos0, rfPos0, grid_x_steps, grid_y_steps, grid_yaw_steps, x_step_unit, y_step_unit, y_offset, yaw_step_unit
):
    samples = []
    x_values_right = np.arange(-grid_x_steps/2, grid_x_steps/2) * x_step_unit 
    y_values_right = -np.arange(grid_y_steps) * y_step_unit - y_offset
    yaw_values_right = np.arange(-grid_yaw_steps/2, grid_yaw_steps/2) * yaw_step_unit

    #left swing, respected to right stance
    x_values_left = np.arange(-grid_x_steps/2, grid_x_steps/2) * x_step_unit 
    y_values_left = np.arange(grid_y_steps) * y_step_unit + y_offset
    yaw_values_left = np.arange(-grid_yaw_steps/2, grid_yaw_steps/2) * yaw_step_unit
    # First sequence: Right foot swings, then left foot swings
    for dx1 in x_values_right:
        for dy1 in y_values_right:
            for yaw1 in yaw_values_right:
                # Step 1: Right foot swings
                rf_step1 = lfPos0.copy()
                rf_step1[0] += dx1
                rf_step1[1] += dy1

                # Step 2: Left foot swings (displacement will be applied after step 1)
                for dx2 in x_values_left:
                    for dy2 in y_values_left:
                        for yaw2 in yaw_values_left:
                            samples.append(
                                {
                                    "step1": {
                                        "left_target": lfPos0.copy(),
                                        "right_target": rf_step1.copy(),
                                        "stance_foot": "left",
                                        "swing_foot": "right",
                                        "target_yaw": yaw1,
                                    },
                                    "step2_displacement": {
                                        "dx": dx2,
                                        "dy": dy2,
                                        "stance_foot": "right",
                                        "swing_foot": "left",
                                        "target_yaw": yaw2,
                                    },
                                }
                            )

    # Second sequence: Left foot swings, then right foot swings
    for dx1 in x_values_left:
        for dy1 in y_values_left:
            for yaw1 in yaw_values_left:
                # Step 1: Left foot swings
                lf_step1 = rfPos0.copy()
                lf_step1[0] += dx1
                lf_step1[1] += dy1

                # Step 2: Right foot swings (displacement will be applied after step 1)
                for dx2 in x_values_right:
                    for dy2 in y_values_right:
                        for yaw2 in yaw_values_right:
                            samples.append(
                                {
                                    "step1": {
                                        "left_target": lf_step1.copy(),
                                        "right_target": rfPos0.copy(),
                                        "stance_foot": "right",
                                        "swing_foot": "left",
                                        "target_yaw": yaw1,
                                    },
                                    "step2_displacement": {
                                        "dx": dx2,
                                        "dy": dy2,
                                        "stance_foot": "left",
                                        "swing_foot": "right",
                                        "target_yaw": yaw2,
                                    },
                                }
                            )

    return samples


def solve_stepping_problem(gait, x0, left_target, right_target, target_yaw=0.0, verbose=False):
    """Solve a stepping problem with extra safety checks."""
    try:
        # Validate initial state
        nq = gait.rmodel.nq
        if len(x0) != gait.state.nx:
            return None, False

        q0 = x0[:nq]
        if np.any(np.isnan(q0)) or np.any(np.isinf(q0)):
            return None, False

        # Validate targets are reasonable (not NaN, not too far)
        left_target = np.array(left_target)
        right_target = np.array(right_target)

        if np.any(np.isnan(left_target)) or np.any(np.isnan(right_target)):
            return None, False

        problem = gait.createSingleStepProblem(x0, left_target, right_target, TIME_STEP, STEP_KNOTS, SUPPORT_KNOTS, STEP_HEIGHT, target_yaw, TRANSITION_KNOTS, COM_SHIFT_RATIO, INITIAL_COM_SHIFT)

        solver = crocoddyl.SolverIntro(problem)
        solver.th_stop = SOLVER_THRESHOLD

        # Only add verbose callback if requested
        if verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])

        # Create independent copies of x0 for each time step to avoid memory issues
        xs = [x0.copy() for _ in range(solver.problem.T + 1)]
        us = solver.problem.quasiStatic([x0.copy() for _ in range(solver.problem.T)])

        # Solve the problem
        solver.solve(xs, us, MAX_ITERATIONS, False)

        success = solver.stop < SOLVER_THRESHOLD
        return solver, success
    except (RuntimeError, ValueError, ZeroDivisionError):
        return None, False
    except Exception:
        return None, False


def generate_waiting_frames(robot, gait, x0, num_frames, stance_foot):
    """
    Generate waiting frames where the robot stands still.

    Returns:
        dict with keys: q, qd, T_blf, T_brf, T_stsw, p_wcom, T_wbase, v_b, cmd_footstep, cmd_stance, cmd_countdown
    """
    nq = robot.model.nq
    nv = robot.model.nv

    q_data = np.tile(x0[:nq], (num_frames, 1))
    qd_data = np.zeros((num_frames, nv))
    T_blf_data = np.zeros((num_frames, 4))
    T_brf_data = np.zeros((num_frames, 4))
    T_stsw_data = np.zeros((num_frames, 4))
    p_wcom_data = np.zeros((num_frames, 3))
    T_wbase_data = np.zeros((num_frames, 7))
    # Base velocity in base frame (linear, angular)
    v_b_data = np.zeros((num_frames, 6))
    cmd_footstep_data = np.zeros((num_frames, 4))
    cmd_stance_data = np.zeros((num_frames, 1))
    cmd_countdown_data = np.zeros((num_frames, 1))  # All zeros during waiting

    # Compute kinematics for the static pose
    rdata = robot.model.createData()
    q = x0[:nq]

    # Determine stance foot
    pinocchio.forwardKinematics(robot.model, rdata, q)
    pinocchio.updateFramePlacements(robot.model, rdata)

    if stance_foot:
        stance_is_left = 0  # Right foot is stance
        stance_foot_id = gait.rfId
        swing_foot_id = gait.lfId
    else:
        stance_is_left = 1  # Left foot is stance
        stance_foot_id = gait.lfId
        swing_foot_id = gait.rfId

    # Get body transformation
    body_pos = q[:3]
    body_quat = q[3:7]
    body_R = pinocchio.Quaternion(body_quat[3], body_quat[0], body_quat[1], body_quat[2]).toRotationMatrix()

    # Get foot transforms
    lf_world_pos = rdata.oMf[gait.lfId].translation
    rf_world_pos = rdata.oMf[gait.rfId].translation
    lf_world_R = rdata.oMf[gait.lfId].rotation
    rf_world_R = rdata.oMf[gait.rfId].rotation

    lf_body_pos = body_R.T @ (lf_world_pos - body_pos)
    rf_body_pos = body_R.T @ (rf_world_pos - body_pos)

    lf_body_R = body_R.T @ lf_world_R
    rf_body_R = body_R.T @ rf_world_R
    lf_body_yaw = rotation_matrix_to_yaw(lf_body_R)
    rf_body_yaw = rotation_matrix_to_yaw(rf_body_R)

    # CoM
    com_world = pinocchio.centerOfMass(robot.model, rdata, q)

    # Stance and swing foot poses
    stance_pos = rdata.oMf[stance_foot_id].translation
    stance_R = rdata.oMf[stance_foot_id].rotation
    swing_pos = rdata.oMf[swing_foot_id].translation
    swing_R = rdata.oMf[swing_foot_id].rotation

    swing_stance_pos = stance_R.T @ (swing_pos - stance_pos)
    swing_stance_R = stance_R.T @ swing_R
    swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)

    # Footstep command - use current swing position and rotation since feet are stationary during waiting
    cmd_footstep = transform_to_stance_frame(swing_pos, stance_pos, stance_R, swing_R)

    # Fill all frames with the same data
    for i in range(num_frames):
        T_blf_data[i] = [lf_body_pos[0], lf_body_pos[1], lf_body_pos[2], lf_body_yaw]
        T_brf_data[i] = [rf_body_pos[0], rf_body_pos[1], rf_body_pos[2], rf_body_yaw]
        T_stsw_data[i] = [swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw]
        p_wcom_data[i] = com_world
        T_wbase_data[i] = [body_pos[0], body_pos[1], body_pos[2], body_quat[3], body_quat[0], body_quat[1], body_quat[2]]
        cmd_footstep_data[i] = cmd_footstep
        #detect the previous cmd_stance, if next stance foot is left, the previous foot is right
        cmd_stance_data[i, 0] = 0 if stance_is_left else 1

    # print("while waiting, stance foot:", cmd_stance_data[0,0])

    return {
        "q": q_data,
        "qd": qd_data,
        "T_blf": T_blf_data,
        "T_brf": T_brf_data,
        "T_stsw": T_stsw_data,
        "p_wcom": p_wcom_data,
        "T_wbase": T_wbase_data,
        "v_b": v_b_data,
        "cmd_footstep": cmd_footstep_data,
        "cmd_stance": cmd_stance_data,
        "cmd_countdown": cmd_countdown_data,
    }


def extract_trajectory_data(robot, solver, gait, left_target, right_target):
    """
    Extract trajectory data in required format.

    Returns:
        dict with keys: q, qd, T_blf, T_brf, T_stsw, p_wcom, T_wbase, v_b, cmd_footstep, cmd_stance, cmd_countdown
    """
    T = len(solver.xs)
    # print(T)
    nq = robot.model.nq
    nv = robot.model.nv

    q_data = np.zeros((T, nq))
    qd_data = np.zeros((T, nv))
    T_blf_data = np.zeros((T, 4))  # (x, y, z, yaw)
    T_brf_data = np.zeros((T, 4))  # (x, y, z, yaw)
    T_stsw_data = np.zeros((T, 4))  # (x, y, z, yaw) stance to swing
    p_wcom_data = np.zeros((T, 3))
    T_wbase_data = np.zeros((T, 7))  # (x, y, z, qw, qx, qy, qz)
    # Base velocity in base frame (linear, angular)
    v_b_data = np.zeros((T, 6))
    cmd_footstep_data = np.zeros((T, 4))
    cmd_stance_data = np.zeros((T, 1))
    cmd_countdown_data = np.zeros((T, 1))

    # Determine which foot is moving (stance foot is the one NOT moving)
    rdata = robot.model.createData()
    pinocchio.forwardKinematics(robot.model, rdata, solver.xs[0][:nq])
    pinocchio.updateFramePlacements(robot.model, rdata)

    lf_init = rdata.oMf[gait.lfId].translation.copy()
    rf_init = rdata.oMf[gait.rfId].translation.copy()

    # Get final foot positions and rotations from last state
    pinocchio.forwardKinematics(robot.model, rdata, solver.xs[-1][:nq])
    pinocchio.updateFramePlacements(robot.model, rdata)
    lf_final = rdata.oMf[gait.lfId].translation.copy()
    rf_final = rdata.oMf[gait.rfId].translation.copy()
    lf_final_R = rdata.oMf[gait.lfId].rotation.copy()
    rf_final_R = rdata.oMf[gait.rfId].rotation.copy()

    left_movement = np.linalg.norm(left_target - lf_init)
    right_movement = np.linalg.norm(right_target - rf_init)

    # Stance foot: 0 = left, 1 = right
    # If left foot moves more, left is swing, right is stance
    if left_movement > right_movement:
        stance_is_left = 0  # Right foot is stance
        swing_target = lf_final  # Use actual achieved position, not target
        swing_target_R = lf_final_R  # Use actual achieved rotation
        stance_foot_id = gait.rfId
        swing_foot_id = gait.lfId
    else:
        stance_is_left = 1  # Left foot is stance
        swing_target = rf_final  # Use actual achieved position, not target
        swing_target_R = rf_final_R  # Use actual achieved rotation
        stance_foot_id = gait.lfId
        swing_foot_id = gait.rfId

    for t in range(T):
        q = solver.xs[t][:nq]
        qd = solver.xs[t][nq:]

        # Store q and qd
        q_data[t] = q
        qd_data[t] = qd

        # Compute forward kinematics
        pinocchio.forwardKinematics(robot.model, rdata, q)
        pinocchio.updateFramePlacements(robot.model, rdata)

        # Get body (base) transformation
        body_pos = q[:3]
        body_quat = q[3:7]  # [x, y, z, w] in pinocchio
        # Convert quaternion to rotation matrix
        body_R = pinocchio.Quaternion(body_quat[3], body_quat[0], body_quat[1], body_quat[2]).toRotationMatrix()

        # Get foot transforms in world frame
        lf_world_pos = rdata.oMf[gait.lfId].translation
        rf_world_pos = rdata.oMf[gait.rfId].translation
        lf_world_R = rdata.oMf[gait.lfId].rotation
        rf_world_R = rdata.oMf[gait.rfId].rotation

        # Transform to body frame: p_body_to_foot = R_body^T @ (p_foot - p_body)
        lf_body_pos = body_R.T @ (lf_world_pos - body_pos)
        rf_body_pos = body_R.T @ (rf_world_pos - body_pos)

        # Get yaw angles from rotation matrices
        lf_body_R = body_R.T @ lf_world_R
        rf_body_R = body_R.T @ rf_world_R
        lf_body_yaw = rotation_matrix_to_yaw(lf_body_R)
        rf_body_yaw = rotation_matrix_to_yaw(rf_body_R)

        # Compute base velocity in base frame
        # qd[:3] is linear velocity in world frame, qd[3:6] is angular velocity in world frame
        v_world_linear = qd[:3]
        v_world_angular = qd[3:6]
        v_b_linear = body_R.T @ v_world_linear
        v_b_angular = body_R.T @ v_world_angular
        v_b_data[t] = np.concatenate([v_b_linear, v_b_angular])

        # Store body to foot transforms (x, y, z, yaw)
        T_blf_data[t] = np.array([lf_body_pos[0], lf_body_pos[1], lf_body_pos[2], lf_body_yaw])
        T_brf_data[t] = np.array([rf_body_pos[0], rf_body_pos[1], rf_body_pos[2], rf_body_yaw])

        # Compute CoM position in world frame
        com_world = pinocchio.centerOfMass(robot.model, rdata, q)
        p_wcom_data[t] = com_world

        # Store base transform in world frame (x, y, z, qw, qx, qy, qz)
        T_wbase_data[t] = np.array([body_pos[0], body_pos[1], body_pos[2], body_quat[3], body_quat[0], body_quat[1], body_quat[2]])

        # Get stance and swing foot poses
        stance_pos = rdata.oMf[stance_foot_id].translation
        stance_R = rdata.oMf[stance_foot_id].rotation
        swing_pos = rdata.oMf[swing_foot_id].translation
        swing_R = rdata.oMf[swing_foot_id].rotation

        # Compute stance to swing transform (x, y, z, yaw)
        swing_stance_pos = stance_R.T @ (swing_pos - stance_pos)
        swing_stance_R = stance_R.T @ swing_R
        swing_stance_yaw = rotation_matrix_to_yaw(swing_stance_R)
        T_stsw_data[t] = np.array([swing_stance_pos[0], swing_stance_pos[1], swing_stance_pos[2], swing_stance_yaw])

        # Compute cmd_footstep in stance frame using final swing foot position and rotation
        cmd_footstep_data[t] = transform_to_stance_frame(swing_target, stance_pos, stance_R, swing_target_R)

        # Stance indicator
        cmd_stance_data[t, 0] = 0 if stance_is_left else 1

        # Countdown: goes from 1 -> 0 during support + step phases, then 0 during transition
        # Support + Step phases: SUPPORT_KNOTS + STEP_KNOTS knots
        # Transition phase: TRANSITION_KNOTS knots (countdown should be 0)
        support_step_knots = SUPPORT_KNOTS + STEP_KNOTS
        if t < support_step_knots:
            progress = t / (support_step_knots - 1) if support_step_knots > 1 else 0
            cmd_countdown_data[t, 0] = 1 - progress
        else:
            cmd_countdown_data[t, 0] = 0  # Transition phase: countdown is 0

    return {
        "q": q_data,
        "qd": qd_data,
        "T_blf": T_blf_data,
        "T_brf": T_brf_data,
        "T_stsw": T_stsw_data,
        "p_wcom": p_wcom_data,
        "T_wbase": T_wbase_data,
        "v_b": v_b_data,
        "cmd_footstep": cmd_footstep_data,
        "cmd_stance": cmd_stance_data,
        "cmd_countdown": cmd_countdown_data,
    }


def create_display(robot):
    """Create a display instance for visualization."""
    if WITHDISPLAY:
        try:
            display = crocoddyl.MeshcatDisplay(robot)
            display.rate = -1
            display.freq = 1
            # Set friction cone color: [R, G, B, Alpha] where each is 0.0-1.0
            # Default is [0.0, 0.4, 0.79, 0.5] (blue, 50% opacity)
            display.frictionConeColor = [1.0, 0.0, 0.0, 0.95]  # Red with 90% opacity
            # Set white background
            display.robot.viz.viewer["/Background"].set_property("top_color", [1, 1, 1])
            display.robot.viz.viewer["/Background"].set_property("bottom_color", [1, 1, 1])
            print("[Meshcat] Visualization enabled - open browser at http://localhost:7000")
            return display
        except Exception as e:
            print(f"[Warning] Failed to create display: {e}")
            return None
    return None


def visualize_trajectory(display, solver):
    """Visualize a single trajectory using existing display."""
    if display is not None:
        display.displayFromSolver(solver)


def extract_com_from_trajectory(robot, q_trajectory, start_idx, end_idx):
    """Extract COM position from trajectory segment."""
    com_trajectory = []
    rdata = robot.model.createData()

    for i in range(start_idx, end_idx):
        q = q_trajectory[i]
        pinocchio.forwardKinematics(robot.model, rdata, q)
        com = pinocchio.centerOfMass(robot.model, rdata, q)
        com_trajectory.append(com)

    return np.array(com_trajectory)


def extract_feet_from_trajectory(robot, q_trajectory, start_idx, end_idx, foot_frame):
    """Extract foot position from trajectory segment."""
    foot_trajectory = []
    rdata = robot.model.createData()
    foot_id = robot.model.getFrameId(foot_frame)

    for i in range(start_idx, end_idx):
        q = q_trajectory[i]
        pinocchio.forwardKinematics(robot.model, rdata, q)
        pinocchio.updateFramePlacements(robot.model, rdata)
        foot_pos = rdata.oMf[foot_id].translation.copy()
        foot_trajectory.append(foot_pos)

    return np.array(foot_trajectory)


def extract_foot_velocity_from_trajectory(foot_positions, dt):
    """
    Compute foot velocity in world frame from position trajectory using central differences.

    Args:
        foot_positions: (N, 3) array of foot positions in world frame
        dt: time step between frames

    Returns:
        (N, 3) array of velocities. Raw extraction without post-processing.
    """
    velocities = np.zeros_like(foot_positions)

    if len(foot_positions) < 2:
        return velocities

    # Forward difference for first point
    velocities[0] = (foot_positions[1] - foot_positions[0]) / dt

    # Central difference for interior points (smoother than backward difference)
    if len(foot_positions) > 2:
        velocities[1:-1] = (foot_positions[2:] - foot_positions[:-2]) / (2 * dt)

    # Backward difference for last point
    velocities[-1] = (foot_positions[-1] - foot_positions[-2]) / dt

    return velocities


def save_checkpoint(checkpoint_num, all_q, all_qd, all_T_blf, all_T_brf, all_T_stsw,
                    all_p_wcom, all_T_wbase, all_v_b, all_cmd_footstep,
                    all_cmd_stance, all_cmd_countdown, traj_starts):
    """Save checkpoint of accumulated data."""
    checkpoint_file = os.path.join(SCRIPT_DIR, f"checkpoint_{checkpoint_num:04d}.npz")

    # Concatenate all data
    q = np.vstack(all_q)
    qd = np.vstack(all_qd)
    T_blf = np.vstack(all_T_blf)
    T_brf = np.vstack(all_T_brf)
    T_stsw = np.vstack(all_T_stsw)
    p_wcom = np.vstack(all_p_wcom)
    T_wbase = np.vstack(all_T_wbase)
    v_b = np.vstack(all_v_b)
    cmd_footstep = np.vstack(all_cmd_footstep)
    cmd_stance = np.vstack(all_cmd_stance)
    cmd_countdown = np.vstack(all_cmd_countdown)
    traj = np.array(traj_starts[:-1], dtype=np.int32)  # Exclude the last pending start

    np.savez_compressed(
        checkpoint_file,
        q=q, qd=qd, T_blf=T_blf, T_brf=T_brf, T_stsw=T_stsw,
        p_wcom=p_wcom, T_wbase=T_wbase, v_b=v_b,
        cmd_footstep=cmd_footstep, cmd_stance=cmd_stance, cmd_countdown=cmd_countdown,
        traj=traj, traj_dt=TIME_STEP,
    )
    print(f"  ✓ Checkpoint saved: {checkpoint_file} ({len(traj)} trajectories, {len(q)} timesteps)")


def plot_com_trajectory(robot, q_trajectory, traj_start, traj_end, traj_idx):
    """Plot COM trajectory for a single trajectory segment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"COM Trajectory Analysis - Trajectory {traj_idx}", fontsize=14, fontweight='bold')

    com_traj = extract_com_from_trajectory(robot, q_trajectory, traj_start, traj_end)
    lf_traj = extract_feet_from_trajectory(robot, q_trajectory, traj_start, traj_end, "left_foot_link")
    rf_traj = extract_feet_from_trajectory(robot, q_trajectory, traj_start, traj_end, "right_foot_link")

    time_steps = np.arange(len(com_traj))

    # XY plane view (top-down)
    ax = axes[0, 0]
    ax.plot(com_traj[:, 0], com_traj[:, 1], 'b-', linewidth=2, label='COM')
    ax.plot(lf_traj[:, 0], lf_traj[:, 1], 'r--', linewidth=2, label='Left Foot')
    ax.plot(rf_traj[:, 0], rf_traj[:, 1], 'g--', linewidth=2, label='Right Foot')
    ax.plot(com_traj[0, 0], com_traj[0, 1], 'bo', markersize=8, label='Start')
    ax.plot(com_traj[-1, 0], com_traj[-1, 1], 'bs', markersize=8, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-Down View (XY Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # X position over time
    ax = axes[0, 1]
    ax.plot(time_steps, com_traj[:, 0], 'b-', linewidth=2, label='COM X')
    ax.plot(time_steps, lf_traj[:, 0], 'r--', linewidth=1.5, alpha=0.7, label='LF X')
    ax.plot(time_steps, rf_traj[:, 0], 'g--', linewidth=1.5, alpha=0.7, label='RF X')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('X Position (m)')
    ax.set_title('Forward/Backward Motion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y position over time (Lateral Lean - CRITICAL)
    ax = axes[1, 0]
    ax.plot(time_steps, com_traj[:, 1], 'b-', linewidth=2.5, label='COM Y')
    ax.plot(time_steps, lf_traj[:, 1], 'r--', linewidth=1.5, alpha=0.7, label='LF Y')
    ax.plot(time_steps, rf_traj[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='RF Y')
    ax.fill_between(time_steps, lf_traj[:, 1], rf_traj[:, 1], alpha=0.1, color='gray')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Lateral Lean (CRITICAL)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z position over time (Height)
    ax = axes[1, 1]
    ax.plot(time_steps, com_traj[:, 2], 'b-', linewidth=2, label='COM Z')
    ax.plot(time_steps, lf_traj[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='LF Z')
    ax.plot(time_steps, rf_traj[:, 2], 'g--', linewidth=1.5, alpha=0.7, label='RF Z')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Height (m)')
    ax.set_title('Vertical Motion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main batch generation loop."""
    print("=" * 80)
    print("Stepping Motion Dataset Generator")
    print("=" * 80)

    # Print configuration parameters
    print("\n[Configuration Parameters]")
    print(f"TIME_STEP: {TIME_STEP}")
    print(f"STEP_KNOTS: {STEP_KNOTS}")
    print(f"SUPPORT_KNOTS: {SUPPORT_KNOTS}")
    print(f"TRANSITION_KNOTS: {TRANSITION_KNOTS}")
    print(f"COM_SHIFT_RATIO: {COM_SHIFT_RATIO}")
    print(f"INITIAL_COM_SHIFT: {INITIAL_COM_SHIFT}")
    print(f"WITHDISPLAY: {WITHDISPLAY}")
    print(f"CHECKPOINT_FREQUENCY: {CHECKPOINT_FREQUENCY}")
    print(f"STEP_HEIGHT: {STEP_HEIGHT}")
    print(f"WAIT_TIME_RANGE: {WAIT_TIME_RANGE}")
    print(f"MID_WAIT_TIME_RANGE: {MID_WAIT_TIME_RANGE}")
    print(f"GRID_X_STEPS: {GRID_X_STEPS}")
    print(f"GRID_Y_STEPS: {GRID_Y_STEPS}")
    print(f"GRID_YAW_STEPS: {GRID_YAW_STEPS}")
    print(f"X_STEP_UNIT: {X_STEP_UNIT}")
    print(f"Y_STEP_UNIT: {Y_STEP_UNIT}")
    print(f"Y_OFFSET: {Y_OFFSET}")
    print(f"YAW_STEP_UNIT: {YAW_STEP_UNIT}")

    # Load robot
    print("\n[1/4] Loading robot model...")
    robot = load_robot()

    # Initial state
    q0 = robot.model.referenceConfigurations["half_sitting"].copy()
    v0 = np.zeros(robot.model.nv)
    x0 = np.concatenate([q0, v0])

    # Get initial foot positions
    rightFoot = "right_foot_link"
    leftFoot = "left_foot_link"

    rdata = robot.model.createData()
    pinocchio.forwardKinematics(robot.model, rdata, q0)
    pinocchio.updateFramePlacements(robot.model, rdata)

    rfId = robot.model.getFrameId(rightFoot)
    lfId = robot.model.getFrameId(leftFoot)
    rfPos0 = rdata.oMf[rfId].translation.copy()
    lfPos0 = rdata.oMf[lfId].translation.copy()

    print(f"Initial right foot position: {rfPos0}")
    print(f"Initial left foot position: {lfPos0}")

    # Initialize gait problem
    print("\n[2/4] Initializing gait problem...")
    gait = SimpleBipedGaitProblem(robot.model, rightFoot, leftFoot, fwddyn=False)

    # Create display (if visualization enabled)
    display = create_display(robot)
    if display is not None:
        print("Visualization enabled")

    # Generate grid samples
    print("\n[3/4] Generating grid samples...")
    grid_samples = generate_grid_samples(
        lfPos0, rfPos0, GRID_X_STEPS, GRID_Y_STEPS, GRID_YAW_STEPS, X_STEP_UNIT, Y_STEP_UNIT, Y_OFFSET, YAW_STEP_UNIT
    )
    # Shuffle with a fixed seed for reproducibility
    shuffle_seed = 42
    np.random.seed(shuffle_seed)
    shuffle_indices = np.random.permutation(len(grid_samples))
    grid_samples = [grid_samples[i] for i in shuffle_indices]
    print(f"Total grid samples: {len(grid_samples)}")

    # Accumulate all trajectory data
    all_q = []
    all_qd = []
    all_T_blf = []
    all_T_brf = []
    all_T_stsw = []
    all_p_wcom = []
    all_T_wbase = []
    all_v_b = []
    all_cmd_footstep = []
    all_cmd_stance = []
    all_cmd_countdown = []
    traj_starts = [0]  # First trajectory starts at index 0

    successful_samples = 0
    left_foot_first_count = 0  # Count trajectories where left foot swings first
    right_foot_first_count = 0  # Count trajectories where right foot swings first

    for i, sample in enumerate(grid_samples):
        print(f"--- Sample {i + 1}/{len(grid_samples)} ---")

        step1 = sample["step1"]
        step2_disp = sample["step2_displacement"]

        print(f"Step 1: {step1['swing_foot']} swings (stance: {step1['stance_foot']})")
        print(f"  Left foot: {lfPos0[:2]} -> {step1['left_target'][:2]}")
        print(f"  Right foot: {rfPos0[:2]} -> {step1['right_target'][:2]}")

        # Generate random waiting time and frames (before stepping)
        wait_time_before = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_before = int(wait_time_before / TIME_STEP)
        first_stance = np.random.randint(2)
        # Generate waiting frames before first step
        try:
            wait_data_before = generate_waiting_frames(
                robot, gait, x0, wait_frames_before, first_stance
            )
        except Exception as e:
            print(f"✗ Failed to generate wait_data_before: {e}")
            continue

        # Solve first step
        solver1, success1 = solve_stepping_problem(gait, x0, step1["left_target"], step1["right_target"], step1["target_yaw"], verbose=False)

        if not success1 or solver1 is None:
            # Track memory on failure
            # if (i + 1) % 100 == 0:
            #     print(f"  [Memory on failure] {get_memory_usage():.2f} MB")
            continue

        # Visualize first step
        visualize_trajectory(display, solver1)

        # Extract first step data
        try:
            step1_data = extract_trajectory_data(robot, solver1, gait, step1["left_target"], step1["right_target"])
        except Exception as e:
            print(f"✗ Failed to extract step1 data: {e}")
            del solver1
            continue

        # Compute actual foot positions after step 1
        state_after_step1 = solver1.xs[-1].copy()  # Copy to avoid reference to solver
        q_after_step1 = state_after_step1[: robot.model.nq]

        # Clean up solver1 to free memory
        del solver1

        rdata_step1 = robot.model.createData()
        pinocchio.forwardKinematics(robot.model, rdata_step1, q_after_step1)
        pinocchio.updateFramePlacements(robot.model, rdata_step1)

        lf_after_step1 = rdata_step1.oMf[gait.lfId].translation.copy()
        rf_after_step1 = rdata_step1.oMf[gait.rfId].translation.copy()

        # Compute step 2 targets based on actual positions after step 1
        if step2_disp["swing_foot"] == "left":
            # Left foot swings, right foot is stance
            lf_step2_target = rf_after_step1.copy()
            lf_step2_target[0] += step2_disp["dx"]
            lf_step2_target[1] += step2_disp["dy"]
            rf_step2_target = rf_after_step1.copy()  # Stance foot stays
        else:
            # Right foot swings, left foot is stance
            rf_step2_target = lf_after_step1.copy()
            rf_step2_target[0] += step2_disp["dx"]
            rf_step2_target[1] += step2_disp["dy"]
            lf_step2_target = lf_after_step1.copy()  # Stance foot stays

        print(f"Step 2: {step2_disp['swing_foot']} swings (stance: {step2_disp['stance_foot']})")
        print(f"  Left foot: {lf_after_step1[:2]} -> {lf_step2_target[:2]}")
        print(f"  Right foot: {rf_after_step1[:2]} -> {rf_step2_target[:2]}")

        # Generate middle waiting time and frames (between steps)
        wait_time_mid = np.random.uniform(MID_WAIT_TIME_RANGE[0], MID_WAIT_TIME_RANGE[1])
        wait_frames_mid = int(wait_time_mid / TIME_STEP)
        prev_stance =  0 if step1['stance_foot'] == "left" else 1
        # Generate waiting frames between steps (using final state from step 1)
        try:
            wait_data_mid = generate_waiting_frames(
                robot, gait, state_after_step1, wait_frames_mid, prev_stance
            )
            # Override cmd_footstep and cmd_stance to match the previous step (step1)
            # wait_data_mid["cmd_footstep"][:] = step1_data["cmd_footstep"][-1]
            # wait_data_mid["cmd_stance"][:] = step1_data["cmd_stance"][-1]
        except Exception as e:
            print(f"✗ Failed to generate wait_data_mid: {e}")
            continue

        # Solve second step (starting from final state of step 1)
        solver2, success2 = solve_stepping_problem(gait, state_after_step1, lf_step2_target, rf_step2_target, step2_disp["target_yaw"], verbose=False)

        if not success2 or solver2 is None:
            continue

        # Visualize second step
        visualize_trajectory(display, solver2)

        # Extract second step data
        try:
            step2_data = extract_trajectory_data(robot, solver2, gait, lf_step2_target, rf_step2_target)
        except Exception as e:
            print(f"✗ Failed to extract step2 data: {e}")
            del solver2
            continue

        # Generate random waiting time and frames (after second step)
        wait_time_after = np.random.uniform(WAIT_TIME_RANGE[0], WAIT_TIME_RANGE[1])
        wait_frames_after = int(wait_time_after / TIME_STEP)

        # Generate waiting frames after second step (using final state from step 2)
        final_state = solver2.xs[-1].copy()  # Copy to avoid reference to solver
        prev_stance2 = 0 if step2_disp['stance_foot'] == "left" else 1
        try:
            wait_data_after = generate_waiting_frames(
                robot, gait, final_state, wait_frames_after, prev_stance2
            )
            # Override cmd_footstep and cmd_stance to match the previous step (step2)
            # wait_data_after["cmd_footstep"][:] = step2_data["cmd_footstep"][-1]
            # wait_data_after["cmd_stance"][:] = step2_data["cmd_stance"][-1]
        except Exception as e:
            print(f"✗ Failed to generate wait_data_after: {e}")
            del solver2
            continue

        # Clean up solver2 to free memory
        del solver2

        # Concatenate: waiting_before + step1 + waiting_mid + step2 + waiting_after
        all_q.append(wait_data_before["q"])
        all_q.append(step1_data["q"])
        all_q.append(wait_data_mid["q"])
        all_q.append(step2_data["q"])
        all_q.append(wait_data_after["q"])

        all_qd.append(wait_data_before["qd"])
        all_qd.append(step1_data["qd"])
        all_qd.append(wait_data_mid["qd"])
        all_qd.append(step2_data["qd"])
        all_qd.append(wait_data_after["qd"])

        all_T_blf.append(wait_data_before["T_blf"])
        all_T_blf.append(step1_data["T_blf"])
        all_T_blf.append(wait_data_mid["T_blf"])
        all_T_blf.append(step2_data["T_blf"])
        all_T_blf.append(wait_data_after["T_blf"])

        all_T_brf.append(wait_data_before["T_brf"])
        all_T_brf.append(step1_data["T_brf"])
        all_T_brf.append(wait_data_mid["T_brf"])
        all_T_brf.append(step2_data["T_brf"])
        all_T_brf.append(wait_data_after["T_brf"])

        all_T_stsw.append(wait_data_before["T_stsw"])
        all_T_stsw.append(step1_data["T_stsw"])
        all_T_stsw.append(wait_data_mid["T_stsw"])
        all_T_stsw.append(step2_data["T_stsw"])
        all_T_stsw.append(wait_data_after["T_stsw"])

        all_p_wcom.append(wait_data_before["p_wcom"])
        all_p_wcom.append(step1_data["p_wcom"])
        all_p_wcom.append(wait_data_mid["p_wcom"])
        all_p_wcom.append(step2_data["p_wcom"])
        all_p_wcom.append(wait_data_after["p_wcom"])

        all_T_wbase.append(wait_data_before["T_wbase"])
        all_T_wbase.append(step1_data["T_wbase"])
        all_T_wbase.append(wait_data_mid["T_wbase"])
        all_T_wbase.append(step2_data["T_wbase"])
        all_T_wbase.append(wait_data_after["T_wbase"])

        all_v_b.append(wait_data_before["v_b"])
        all_v_b.append(step1_data["v_b"])
        all_v_b.append(wait_data_mid["v_b"])
        all_v_b.append(step2_data["v_b"])
        all_v_b.append(wait_data_after["v_b"])

        all_cmd_footstep.append(wait_data_before["cmd_footstep"])
        all_cmd_footstep.append(step1_data["cmd_footstep"])
        all_cmd_footstep.append(wait_data_mid["cmd_footstep"])
        all_cmd_footstep.append(step2_data["cmd_footstep"])
        all_cmd_footstep.append(wait_data_after["cmd_footstep"])

        all_cmd_stance.append(wait_data_before["cmd_stance"])
        all_cmd_stance.append(step1_data["cmd_stance"])
        all_cmd_stance.append(wait_data_mid["cmd_stance"])
        all_cmd_stance.append(step2_data["cmd_stance"])
        all_cmd_stance.append(wait_data_after["cmd_stance"])

        all_cmd_countdown.append(wait_data_before["cmd_countdown"])
        all_cmd_countdown.append(step1_data["cmd_countdown"])
        all_cmd_countdown.append(wait_data_mid["cmd_countdown"])
        all_cmd_countdown.append(step2_data["cmd_countdown"])
        all_cmd_countdown.append(wait_data_after["cmd_countdown"])

        # Record next trajectory start index
        current_length = sum(len(q) for q in all_q)
        traj_starts.append(current_length)

        successful_samples += 1

        # Track which foot swings first
        if step1["swing_foot"] == "left":
            left_foot_first_count += 1
        else:
            right_foot_first_count += 1

        print(
            f"✓ Success! Wait before: {wait_frames_before}, Step1: {len(step1_data['q'])}, "
            + f"Wait mid: {wait_frames_mid}, Step2: {len(step2_data['q'])}, Wait after: {wait_frames_after}"
        )

        # Force garbage collection EVERY successful sample to free memory aggressively
        gc.collect()

        # Save checkpoint if enabled
        if CHECKPOINT_FREQUENCY > 0 and successful_samples % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(
                successful_samples, all_q, all_qd, all_T_blf, all_T_brf, all_T_stsw,
                all_p_wcom, all_T_wbase, all_v_b, all_cmd_footstep,
                all_cmd_stance, all_cmd_countdown, traj_starts
            )

        # Optionally plot this trajectory (every Nth sample to avoid too many plots)
        if PLOT:
            if successful_samples % PLOT == 0:  # Plot every 5th successful sample
                print(f"  Plotting trajectory {successful_samples}...")
                # Combine all data for this trajectory
                traj_q = np.vstack([
                    wait_data_before["q"], step1_data["q"], wait_data_mid["q"],
                    step2_data["q"], wait_data_after["q"]
                ])
                traj_v_b = np.vstack([
                    wait_data_before["v_b"], step1_data["v_b"], wait_data_mid["v_b"],
                    step2_data["v_b"], wait_data_after["v_b"]
                ])
                traj_cmd_countdown = np.vstack([
                    wait_data_before["cmd_countdown"], step1_data["cmd_countdown"], wait_data_mid["cmd_countdown"],
                    step2_data["cmd_countdown"], wait_data_after["cmd_countdown"]
                ])
                traj_com_rpy = np.vstack([
                    wait_data_before["qd"], step1_data["qd"], wait_data_mid["qd"],
                    step2_data["qd"], wait_data_after["qd"]
                ])
                traj_T_wbase = np.vstack([
                    wait_data_before["T_wbase"], step1_data["T_wbase"], wait_data_mid["T_wbase"],
                    step2_data["T_wbase"], wait_data_after["T_wbase"]
                ])
                # Extract COM and feet
                com_traj = extract_com_from_trajectory(robot, traj_q, 0, len(traj_q))
                lf_traj = extract_feet_from_trajectory(robot, traj_q, 0, len(traj_q), "left_foot_link")
                rf_traj = extract_feet_from_trajectory(robot, traj_q, 0, len(traj_q), "right_foot_link")

                # Compute foot velocities in world frame
                lf_vel = extract_foot_velocity_from_trajectory(lf_traj, TIME_STEP)
                rf_vel = extract_foot_velocity_from_trajectory(rf_traj, TIME_STEP)

                # Get all trajectory data
                traj_T_blf = np.vstack([
                    wait_data_before["T_blf"], step1_data["T_blf"], wait_data_mid["T_blf"],
                    step2_data["T_blf"], wait_data_after["T_blf"]
                ])
                traj_T_brf = np.vstack([
                    wait_data_before["T_brf"], step1_data["T_brf"], wait_data_mid["T_brf"],
                    step2_data["T_brf"], wait_data_after["T_brf"]
                ])
                traj_T_stsw = np.vstack([
                    wait_data_before["T_stsw"], step1_data["T_stsw"], wait_data_mid["T_stsw"],
                    step2_data["T_stsw"], wait_data_after["T_stsw"]
                ])
                traj_cmd_footstep = np.vstack([
                    wait_data_before["cmd_footstep"], step1_data["cmd_footstep"], wait_data_mid["cmd_footstep"],
                    step2_data["cmd_footstep"], wait_data_after["cmd_footstep"]
                ])
                traj_cmd_stance = np.vstack([
                    wait_data_before["cmd_stance"], step1_data["cmd_stance"], wait_data_mid["cmd_stance"],
                    step2_data["cmd_stance"], wait_data_after["cmd_stance"]
                ])

                # Plot
                fig, axes = plt.subplots(10, 2, figsize=(14, 40))
                fig.suptitle(f"Trajectory Analysis - Sample {i+1}, Successful #{successful_samples}", fontsize=14, fontweight='bold')
                time_steps = np.arange(len(com_traj))

                # Extract hip roll joint angles (config indices: 25=left, 31=right)
                left_hip_roll = traj_q[:, 25]
                right_hip_roll = traj_q[:, 31]

                # Extract yaw angles for base and feet
                base_yaw_traj = np.zeros(len(traj_q))
                lf_yaw_traj = np.zeros(len(traj_q))
                rf_yaw_traj = np.zeros(len(traj_q))
                for t in range(len(traj_q)):
                    q = traj_q[t]
                    pinocchio.forwardKinematics(robot.model, rdata, q)
                    pinocchio.updateFramePlacements(robot.model, rdata)

                    # Base yaw from quaternion
                    base_quat = q[3:7]  # [x, y, z, w]
                    base_R = pinocchio.Quaternion(base_quat[3], base_quat[0], base_quat[1], base_quat[2]).toRotationMatrix()
                    base_yaw_traj[t] = rotation_matrix_to_yaw(base_R)

                    # Feet yaw
                    lf_yaw_traj[t] = rotation_matrix_to_yaw(rdata.oMf[gait.lfId].rotation)
                    rf_yaw_traj[t] = rotation_matrix_to_yaw(rdata.oMf[gait.rfId].rotation)

                avg_feet_yaw = (lf_yaw_traj + rf_yaw_traj) / 2.0

                # Helper function to add countdown as dashed lines
                def add_countdown_background(ax, time_steps, countdown):
                    """Add dashed line overlay for countdown (active stepping phase)"""
                    # Plot countdown as dashed lines on secondary y-axis
                    ax2 = ax.twinx()
                    ax2.plot(time_steps, countdown[:, 0], 'k--', linewidth=1.5, alpha=0.5, label='Countdown')
                    ax2.set_ylabel('Countdown', alpha=0.5)
                    ax2.set_ylim(-0.1, 1.1)
                    ax2.tick_params(axis='y', labelcolor='black', labelsize=8)
                    ax2.spines['right'].set_alpha(0.3)

                # XY plane (top-down view)
                ax = axes[0, 0]
                ax.plot(com_traj[:, 0], com_traj[:, 1], 'b-', linewidth=2, label='COM')
                ax.plot(lf_traj[:, 0], lf_traj[:, 1], 'r--', linewidth=2, label='Left Foot')
                ax.plot(rf_traj[:, 0], rf_traj[:, 1], 'g--', linewidth=2, label='Right Foot')
                ax.plot(com_traj[0, 0], com_traj[0, 1], 'bo', markersize=8, label='Start')
                ax.plot(com_traj[-1, 0], com_traj[-1, 1], 'bs', markersize=8, label='End')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('Top-Down View (XY Plane)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axis('equal')

                # X position over time
                ax = axes[0, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, com_traj[:, 0], 'b-', linewidth=2, label='COM X')
                ax.plot(time_steps, lf_traj[:, 0], 'r--', linewidth=1.5, alpha=0.7, label='LF X')
                ax.plot(time_steps, rf_traj[:, 0], 'g--', linewidth=1.5, alpha=0.7, label='RF X')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('X Position (m)')
                ax.set_title('Forward/Backward Motion')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Y position over time (lateral lean - CRITICAL)
                ax = axes[1, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, com_traj[:, 1], 'b-', linewidth=2.5, label='COM Y')
                ax.plot(time_steps, lf_traj[:, 1], 'r--', linewidth=1.5, alpha=0.7, label='LF Y')
                ax.plot(time_steps, rf_traj[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='RF Y')
                ax.fill_between(time_steps, lf_traj[:, 1], rf_traj[:, 1], alpha=0.1, color='gray')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Y Position (m)')
                ax.set_title('Lateral Lean (CRITICAL)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Z position over time (height)
                ax = axes[1, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_T_wbase[:, 2], 'b-', linewidth=2, label='Base Z')
                ax.plot(time_steps, lf_traj[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='LF Z')
                ax.plot(time_steps, rf_traj[:, 2], 'g--', linewidth=1.5, alpha=0.7, label='RF Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Height (m)')
                ax.set_title('Vertical Motion (Base Height)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Left foot velocity (world frame)
                ax = axes[2, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, lf_vel[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='LF Vel X')
                ax.plot(time_steps, lf_vel[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='LF Vel Y')
                ax.plot(time_steps, lf_vel[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='LF Vel Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title('Left Foot Velocity (World Frame)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Right foot velocity (world frame)
                ax = axes[2, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, rf_vel[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='RF Vel X')
                ax.plot(time_steps, rf_vel[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='RF Vel Y')
                ax.plot(time_steps, rf_vel[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='RF Vel Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title('Right Foot Velocity (World Frame)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Base linear velocity (base frame)
                ax = axes[3, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                base_vel_linear = traj_v_b[:, :3]
                ax.plot(time_steps, base_vel_linear[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='Base Vel X')
                ax.plot(time_steps, base_vel_linear[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='Base Vel Y')
                ax.plot(time_steps, base_vel_linear[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='Base Vel Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title('Base Linear Velocity (Base Frame)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Base angular velocity (base frame)
                ax = axes[3, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                base_vel_angular = traj_v_b[:, 3:]
                ax.plot(time_steps, base_vel_angular[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='Base ω X')
                ax.plot(time_steps, base_vel_angular[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='Base ω Y')
                ax.plot(time_steps, base_vel_angular[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='Base ω Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Angular Velocity (rad/s)')
                ax.set_title('Base Angular Velocity (Base Frame)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # COM velocity (world frame)
                ax = axes[4, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                com_vel = extract_foot_velocity_from_trajectory(com_traj, TIME_STEP)
                ax.plot(time_steps, com_vel[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='COM Vel X')
                ax.plot(time_steps, com_vel[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='COM Vel Y')
                ax.plot(time_steps, com_vel[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='COM Vel Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title('COM Velocity (World Frame)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # COM RPY
                ax = axes[4, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_com_rpy[:, 3], 'r-', linewidth=1.5, alpha=0.7, label='COM Roll')
                ax.plot(time_steps, traj_com_rpy[:, 4], 'g-', linewidth=1.5, alpha=0.7, label='COM Pitch')
                ax.plot(time_steps, traj_com_rpy[:, 5], 'b-', linewidth=1.5, alpha=0.7, label='COM Yaw')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Radian')
                ax.set_title('COM RPY')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Hip Roll Angles
                ax = axes[5, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, left_hip_roll, 'r-', linewidth=2, label='Left Hip Roll')
                ax.plot(time_steps, right_hip_roll, 'b-', linewidth=2, label='Right Hip Roll')
                ax.axhline(y=0.007658, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Left Target')
                ax.axhline(y=-0.007658, color='b', linestyle='--', linewidth=1, alpha=0.5, label='Right Target')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Hip Roll Angle (rad)')
                ax.set_title('Hip Roll Joint Angles')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Hip Roll Deviation from Initial
                ax = axes[5, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                left_deviation = left_hip_roll - 0.007658
                right_deviation = right_hip_roll - (-0.007658)
                ax.plot(time_steps, left_deviation * 1000, 'r-', linewidth=2, label='Left Hip Roll Deviation')
                ax.plot(time_steps, right_deviation * 1000, 'b-', linewidth=2, label='Right Hip Roll Deviation')
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Deviation (milliradians)')
                ax.set_title('Hip Roll Deviation from Initial Pose')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # T_blf (Body to Left Foot Transform)
                ax = axes[6, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_T_blf[:, 0], 'r-', linewidth=1.5, label='X')
                ax.plot(time_steps, traj_T_blf[:, 1], 'g-', linewidth=1.5, label='Y')
                ax.plot(time_steps, traj_T_blf[:, 2], 'b-', linewidth=1.5, label='Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Position (m)')
                ax.set_title('T_blf: Body to Left Foot Position')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # T_blf Yaw
                ax = axes[6, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_T_blf[:, 3], 'k-', linewidth=2, label='Yaw')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Yaw (rad)')
                ax.set_title('T_blf: Body to Left Foot Yaw')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # T_brf (Body to Right Foot Transform)
                ax = axes[7, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_T_brf[:, 0], 'r-', linewidth=1.5, label='X')
                ax.plot(time_steps, traj_T_brf[:, 1], 'g-', linewidth=1.5, label='Y')
                ax.plot(time_steps, traj_T_brf[:, 2], 'b-', linewidth=1.5, label='Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Position (m)')
                ax.set_title('T_brf: Body to Right Foot Position')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # T_brf Yaw
                ax = axes[7, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_T_brf[:, 3], 'k-', linewidth=2, label='Yaw')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Yaw (rad)')
                ax.set_title('T_brf: Body to Right Foot Yaw')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # T_stsw (Stance to Swing Transform)
                ax = axes[8, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_T_stsw[:, 0], 'r-', linewidth=1.5, label='X')
                ax.plot(time_steps, traj_T_stsw[:, 1], 'g-', linewidth=1.5, label='Y')
                ax.plot(time_steps, traj_T_stsw[:, 2], 'b-', linewidth=1.5, label='Z')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Position (m)')
                ax.set_title('T_stsw: Stance to Swing Foot Position')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # cmd_footstep and cmd_stance
                ax = axes[8, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, traj_cmd_footstep[:, 0], 'r-', linewidth=1.5, alpha=0.7, label='Cmd X')
                ax.plot(time_steps, traj_cmd_footstep[:, 1], 'g-', linewidth=1.5, alpha=0.7, label='Cmd Y')
                ax.plot(time_steps, traj_cmd_stance[:, 0] * 0.1, 'k--', linewidth=2, label='Cmd Stance (×0.1)')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.set_title('cmd_footstep (X, Y) & cmd_stance')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Yaw angles - Base and Feet
                ax = axes[9, 0]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                ax.plot(time_steps, base_yaw_traj, 'b-', linewidth=2.5, label='Base Yaw')
                ax.plot(time_steps, lf_yaw_traj, 'r--', linewidth=1.5, alpha=0.7, label='Left Foot Yaw')
                ax.plot(time_steps, rf_yaw_traj, 'g--', linewidth=1.5, alpha=0.7, label='Right Foot Yaw')
                ax.plot(time_steps, avg_feet_yaw, 'k:', linewidth=2.5, label='Avg Feet Yaw')
                # Mark phase boundaries
                phase1_end = wait_frames_before + SUPPORT_KNOTS
                phase2_end = wait_frames_before + SUPPORT_KNOTS + STEP_KNOTS + 1
                phase3_end = len(traj_q) - wait_frames_after - 1
                ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=phase2_end, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=phase3_end, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='End 2nd DS')
                # Mark end of 2nd double support
                ax.plot(phase3_end, base_yaw_traj[phase3_end], 'bo', markersize=8)
                ax.plot(phase3_end, avg_feet_yaw[phase3_end], 'ko', markersize=8)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Yaw (rad)')
                ax.set_title('Yaw Angles: Base vs Feet (Verification)')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)

                # Yaw tracking error
                ax = axes[9, 1]
                add_countdown_background(ax, time_steps, traj_cmd_countdown)
                yaw_error = (base_yaw_traj - avg_feet_yaw) * 1000  # Convert to milliradians
                ax.plot(time_steps, yaw_error, 'r-', linewidth=2, label='Base Yaw - Avg Feet Yaw')
                ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
                ax.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=phase2_end, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(x=phase3_end, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='End 2nd DS')
                # Mark end of 2nd double support with annotation
                ax.plot(phase3_end, yaw_error[phase3_end], 'ro', markersize=8)
                ax.annotate(f'{yaw_error[phase3_end]:.2f} mrad',
                           xy=(phase3_end, yaw_error[phase3_end]),
                           xytext=(phase3_end-5, yaw_error[phase3_end] + 2),
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Error (milliradians)')
                ax.set_title('Yaw Tracking Error at End of 2nd Double Support')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_file = os.path.join(SCRIPT_DIR, f"velocity_sample_{successful_samples}.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"  Saved: {plot_file}")
                plt.close(fig)

    # Remove last element (it's one past the end)
    traj_starts = traj_starts[:-1]

    # Concatenate all trajectories
    print("\n[4/4] Saving dataset...")
    q = np.vstack(all_q)
    qd = np.vstack(all_qd)
    T_blf = np.vstack(all_T_blf)
    T_brf = np.vstack(all_T_brf)
    T_stsw = np.vstack(all_T_stsw)
    p_wcom = np.vstack(all_p_wcom)
    T_wbase = np.vstack(all_T_wbase)
    v_b = np.vstack(all_v_b)
    cmd_footstep = np.vstack(all_cmd_footstep)
    cmd_stance = np.vstack(all_cmd_stance)
    cmd_countdown = np.vstack(all_cmd_countdown)
    traj = np.array(traj_starts, dtype=np.int32)

    # Save single NPZ file
    np.savez_compressed(
        OUTPUT_FILE,
        q=q,
        qd=qd,
        T_blf=T_blf,
        T_brf=T_brf,
        T_stsw=T_stsw,
        p_wcom=p_wcom,
        T_wbase=T_wbase,
        v_b=v_b,
        cmd_footstep=cmd_footstep,
        cmd_stance=cmd_stance,
        cmd_countdown=cmd_countdown,
        traj=traj,
        traj_dt=TIME_STEP,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Successful trajectories: {successful_samples}/{len(grid_samples)}")
    print(f"  - Left foot first: {left_foot_first_count}")
    print(f"  - Right foot first: {right_foot_first_count}")
    print(f"Total timesteps: {len(q)}")
    print(f"Output file: {OUTPUT_FILE}")
    if CHECKPOINT_FREQUENCY > 0:
        print(f"Checkpoints saved every {CHECKPOINT_FREQUENCY} trajectories")
    print(f"Grid configuration: {GRID_X_STEPS}x{GRID_Y_STEPS}x{GRID_YAW_STEPS} (x × y × yaw)")
    print(f"Step height: {STEP_HEIGHT:.2f} m")
    print(f"Wait time range (before & after): {WAIT_TIME_RANGE[0]:.2f} - {WAIT_TIME_RANGE[1]:.2f} s")
    print(f"Mid wait time range (between steps): {MID_WAIT_TIME_RANGE[0]:.2f} - {MID_WAIT_TIME_RANGE[1]:.2f} s")
    print("\nDataset contents:")
    print(f"  q:             {q.shape}")
    print(f"  qd:            {qd.shape}")
    print(f"  T_blf:         {T_blf.shape} (x, y, z, yaw)")
    print(f"  T_brf:         {T_brf.shape} (x, y, z, yaw)")
    print(f"  T_stsw:        {T_stsw.shape} (x, y, z, yaw)")
    print(f"  p_wcom:        {p_wcom.shape} (x, y, z)")
    print(f"  T_wbase:       {T_wbase.shape} (x, y, z, qw, qx, qy, qz)")
    print(f"  v_b:           {v_b.shape} (linear xyz, angular xyz)")
    print(f"  cmd_footstep:  {cmd_footstep.shape} (x, y, z, yaw)")
    print(f"  cmd_stance:    {cmd_stance.shape} (0 lf stance, 1 rf stance)")
    print(f"  cmd_countdown: {cmd_countdown.shape} (0 during wait, 1->0 during step)")
    print(f"  traj:          {traj.shape} (trajectory start indices)")
    print(f"  traj_dt:       {TIME_STEP:.6f} (time step)")
    print("=" * 80)

    # Plot random trajectory COM
    if successful_samples > 0:
        print("\nPlotting random trajectory COM...")
        random_traj_idx = np.random.randint(0, len(traj))
        traj_start = traj[random_traj_idx]
        traj_end = traj[random_traj_idx + 1] if random_traj_idx + 1 < len(traj) else len(q)

        fig = plot_com_trajectory(robot, q, traj_start, traj_end, random_traj_idx)
        plot_filename = os.path.join(SCRIPT_DIR, f"com_trajectory_random_traj_{random_traj_idx}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved random trajectory plot: {plot_filename}")
        plt.close(fig)


if __name__ == "__main__":
    main()