import os
import signal
import sys
import time

import numpy as np
import pinocchio
import matplotlib.pyplot as plt

import crocoddyl
from g1CrocoddylDigit.utils.g1_jumping_turn_250415 import SimpleBipedGaitProblem

WITHDISPLAY = True
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

g1_urdf = "/home/nzong8/crocoddyl/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
g1_mesh = "/home/nzong8/crocoddyl/unitree_ros/robots/g1_description/"

robot = pinocchio.RobotWrapper.BuildFromURDF(
    g1_urdf,
    [g1_mesh],
    root_joint=pinocchio.JointModelFreeFlyer()
)

print("Joint names:", [robot.model.names[i] for i in range(robot.model.njoints)])
half_sitting = np.array([
    # base position
    0.0, 0.0, 0.60,
    # base orientation quaternion
    0.0, 0.0, 0.0, 1.0,
    # LEFT LEG
    -0.5, 0.0, 0.0, 1.0, -0.5, 0.0,
    # RIGHT LEG
    -0.5, 0.0, 0.0, 1.0, -0.5, 0.0,
    # WAIST
    0.0, 0.0, 0.1,
    # LEFT ARM
    0.1, 0.15, 0.0, -0.3, 0.0, 0.0, 0.0,
    # RIGHT ARM
    0.1, -0.15, 0.0, -0.3, 0.0, 0.0, 0.0,
])
robot.model.referenceConfigurations["half_sitting"] = half_sitting

# Defining the initial state of the robot

rightHand = "right_rubber_hand"
leftHand  = "left_rubber_hand"
rightFoot = "right_ankle_roll_link"
leftFoot = "left_ankle_roll_link"
gait = SimpleBipedGaitProblem(
    robot.model, rightHand, leftHand, rightFoot, leftFoot, fwddyn=True
)

x0 = gait.x0.copy()

def extract_com_trajectory(robot, solver):
    """Extract COM position from solver trajectory."""
    com_trajectory = []
    rdata = robot.model.createData()

    for xs in solver.xs:
        q = xs[:robot.model.nq]
        pinocchio.forwardKinematics(robot.model, rdata, q)
        com = pinocchio.centerOfMass(robot.model, rdata, q)
        com_trajectory.append(com)

    return np.array(com_trajectory)


def extract_foot_positions(robot, solver, foot_frame):
    """Extract foot position from solver trajectory."""
    foot_trajectory = []
    rdata = robot.model.createData()
    foot_id = robot.model.getFrameId(foot_frame)

    for xs in solver.xs:
        q = xs[:robot.model.nq]
        pinocchio.forwardKinematics(robot.model, rdata, q)
        pinocchio.updateFramePlacements(robot.model, rdata)
        foot_pos = rdata.oMf[foot_id].translation.copy()
        foot_trajectory.append(foot_pos)

    return np.array(foot_trajectory)


def plot_com_trajectory(com_trajectory, lf_trajectory, rf_trajectory, phase_name="Step"):
    """Plot COM position relative to feet during stepping."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"COM Trajectory - {phase_name}", fontsize=14, fontweight='bold')

    time_steps = np.arange(len(com_trajectory))

    # XY plane view (top-down)
    ax = axes[0, 0]
    ax.plot(com_trajectory[:, 0], com_trajectory[:, 1], 'b-', linewidth=2, label='COM')
    ax.plot(lf_trajectory[:, 0], lf_trajectory[:, 1], 'r--', linewidth=2, label='Left Foot')
    ax.plot(rf_trajectory[:, 0], rf_trajectory[:, 1], 'g--', linewidth=2, label='Right Foot')
    # Mark start and end
    ax.plot(com_trajectory[0, 0], com_trajectory[0, 1], 'bo', markersize=8, label='COM Start')
    ax.plot(com_trajectory[-1, 0], com_trajectory[-1, 1], 'bs', markersize=8, label='COM End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-Down View (XY Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # X position over time
    ax = axes[0, 1]
    ax.plot(time_steps, com_trajectory[:, 0], 'b-', linewidth=2, label='COM X')
    ax.plot(time_steps, lf_trajectory[:, 0], 'r--', linewidth=1.5, label='Left Foot X')
    ax.plot(time_steps, rf_trajectory[:, 0], 'g--', linewidth=1.5, label='Right Foot X')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('X Position (m)')
    ax.set_title('X Position Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Y position over time
    ax = axes[1, 0]
    ax.plot(time_steps, com_trajectory[:, 1], 'b-', linewidth=2, label='COM Y')
    ax.plot(time_steps, lf_trajectory[:, 1], 'r--', linewidth=1.5, label='Left Foot Y')
    ax.plot(time_steps, rf_trajectory[:, 1], 'g--', linewidth=1.5, label='Right Foot Y')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Y Position Over Time (Lateral Lean)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z position over time
    ax = axes[1, 1]
    ax.plot(time_steps, com_trajectory[:, 2], 'b-', linewidth=2, label='COM Z')
    ax.plot(time_steps, lf_trajectory[:, 2], 'r--', linewidth=1.5, label='Left Foot Z')
    ax.plot(time_steps, rf_trajectory[:, 2], 'g--', linewidth=1.5, label='Right Foot Z')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Z Position (m)')
    ax.set_title('Z Position Over Time (Height)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Get current foot positions for target specification
q0_tmp = x0[:robot.model.nq]
rdata = robot.model.createData()
pinocchio.forwardKinematics(robot.model, rdata, q0_tmp)
pinocchio.updateFramePlacements(robot.model, rdata)
rfId = robot.model.getFrameId(rightFoot)
lfId = robot.model.getFrameId(leftFoot)
rfPos0 = rdata.oMf[rfId].translation.copy()
lfPos0 = rdata.oMf[lfId].translation.copy()

print("Initial foot positions:")
print(f"  Right foot ({rightFoot}): {rfPos0}")
print(f"  Left foot ({leftFoot}): {lfPos0}")

# Setting up all tasks with explicit foot targets
GAITPHASES = [
    {
        "jumping": {
            "jumpHeight":       0.35,
            "jumpLength":       0.3,
            "residual":         0.05,
            "jumpingangle":     np.radians(0),   
            "timeStep":         0.01,            
            "standingKnots":    10,
            "pretakeoffKnots":  30,
            "groundKnots":      15,
            "flyingKnots":      30,
        }
    },
]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "jumping":
            solver[i] = crocoddyl.SolverFDDP(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["residual"],
                    value["jumpingangle"],
                    value["timeStep"],
                    value["standingKnots"],
                    value["pretakeoffKnots"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
    solver[i].th_stop = 1e-7

    # Added the callback functions
    print("*** SOLVE " + key + " ***")
    if WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 500, False)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]

    # Extract and plot COM trajectory
    print("\nExtracting COM trajectory for plotting...")
    com_traj = extract_com_trajectory(robot, solver[i])
    lf_traj = extract_foot_positions(robot, solver[i], "left_ankle_roll_link")
    rf_traj = extract_foot_positions(robot, solver[i], "right_ankle_roll_link")

    phase_name = next(iter(GAITPHASES[i].keys()))
    fig = plot_com_trajectory(com_traj, lf_traj, rf_traj, phase_name=phase_name)
    plt.savefig(f"com_trajectory_{i}_{phase_name}.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot: com_trajectory_{i}_{phase_name}.png")
    plt.close(fig)

# Export trajectories to npz
q_trajectory         = []
root_pose_trajectory = []

for i in range(len(GAITPHASES)):
    for xs in solver[i].xs:
        q         = xs[:robot.model.nq]
        root_pos  = q[:3]
        root_quat = q[3:7]
        root_pose = np.concatenate([[root_quat[3]], root_quat[:3], root_pos])
        q_trajectory.append(q[7:])
        root_pose_trajectory.append(root_pose)

q_trajectory         = np.array(q_trajectory)
root_pose_trajectory = np.array(root_pose_trajectory)
phase_trajectory     = np.linspace(0.0, 1.0, num=q_trajectory.shape[0])

np.savez(
    "trajectory.npz",
    q=q_trajectory,
    phase=phase_trajectory,
    root_pose=root_pose_trajectory
)
print(
    f"Exported trajectory: q{q_trajectory.shape}, phase{phase_trajectory.shape}, root_pose{root_pose_trajectory.shape}"
)

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [3.0, 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        time.sleep(1.0)
