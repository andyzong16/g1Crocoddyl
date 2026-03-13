"""
Quick script to list robot joint names with their configuration indices.
"""

import numpy as np
import pinocchio
import os

def main():
    # Load robot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "/home/nzong8/crocoddyl/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf")
    package_dir = os.path.join(script_dir, "/home/nzong8/crocoddyl/unitree_ros/robots/g1_description/meshes")

    robot = pinocchio.RobotWrapper.BuildFromURDF(
        urdf_path,
        package_dirs=[package_dir],
        root_joint=pinocchio.JointModelFreeFlyer(),
    )

    print("\n" + "=" * 80)
    print("ROBOT CONFIGURATION INDICES AND JOINT NAMES")
    print("=" * 80)
    print(f"Total configuration dimension: {robot.model.nq}\n")

    current_idx = 0

    for i, name in enumerate(robot.model.names[1:], 1):  # Skip universe
        joint = robot.model.joints[i]
        nq = joint.nq
        idx_q = joint.idx_q

        if nq == 0:
            continue

        if nq == 1:
            print(f"Index {idx_q:2d}      : {name}")
        else:
            print(f"Index {idx_q:2d}-{idx_q+nq-1:2d}  : {name} ({nq} DOF)")

    print("\n" + "=" * 80)
    print("QUICK REFERENCE")
    print("=" * 80)
    print("q[0:3]   - Base position (x, y, z)")
    print("q[3:7]   - Base orientation (qx, qy, qz, qw)")
    print("q[7:9]   - Torso")
    print("q[9:16]  - Left arm")
    print("q[16:23] - Right arm")
    print("q[23]    - Head")
    print("q[24:30] - Left leg")
    print("q[30:36] - Right leg")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
