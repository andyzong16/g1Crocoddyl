import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio
from datetime import datetime
import crocoddyl
from utils.digit_jumping_turn_250415 import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)
WITHDISPLAY = True
WITHPLOT = True

urdf_filename = '/home/fliu305/Digit_Manipulation/DigitURDF/digit_description/urdf/digit_fixed.urdf'
# # Now load the model (using pinocchio)
digit_legs = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(filename=str(urdf_filename),root_joint=pinocchio.JointModelFreeFlyer())
robot_model = digit_legs.model
print(robot_model)
rdata = robot_model.createData()
state = crocoddyl.StateMultibody(robot_model)

rightHand = "right_hand"
leftHand = "left_hand"
rightFoot = "right_foot_bottom"
leftFoot = "left_foot_bottom"



## Defining the initial state of the robot
q0 = np.array([ 
                5.25165146e-03, -5.53016165e-05,  1.02603102e+00,  
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  
                
                3.74471076e-01, 0.00000000e+00,   3.11110479e-01,  3.44378687e-01,
                -3.18277045e-01, 1.25446658e-01,  -3.02673931e-04, -7.73557749e-02, 1.14510707e+00, 1.28854386e-03,  -4.27549208e-02, 
                -3.74830602e-01,  1.07378655e-04, -3.11182384e-01, -3.44234876e-01, 3.18444824e-01, -1.25230803e-01,  3.40205933e-04,  
                7.73222191e-02, -1.14532758e+00, -1.22718463e-03,  4.27884766e-02
            ])





v0 = pinocchio.utils.zero(digit_legs.model.nv)
x0 = np.concatenate([q0, v0])

pinocchio.forwardKinematics(robot_model, rdata, q0)
pinocchio.updateFramePlacements(robot_model, rdata)
l_f_p = np.array(rdata.oMf[robot_model.getFrameId(leftFoot)].translation.T.flat)
r_f_p = np.array(rdata.oMf[robot_model.getFrameId(rightFoot)].translation.T.flat)
l_h_p = np.array(rdata.oMf[robot_model.getFrameId(leftHand)].translation.T.flat)
r_h_p = np.array(rdata.oMf[robot_model.getFrameId(rightHand)].translation.T.flat)
print('left foot:',l_f_p)
print('right foot:',r_f_p)
print('left hand:',l_h_p)
print('right hand:',r_h_p)


# display = crocoddyl.MeshcatDisplay(digit_legs, 4, 4, False)

# Setting up the 3d walking problem




gait = SimpleBipedGaitProblem(digit_legs.model, rightHand, leftHand, rightFoot, leftFoot)

# Setting up all tasks
GAITPHASES = [
    {
        "walking": {
            "jumpHeight":  0.2,
            "jumpLength":  0.1,
            "residual": 0.05,
            "jumpingangle": np.radians(60),
            "timeStep": 0.005,
            "standingKnots": 200,
            "pretakeoffKnots": 300,
            "groundKnots": 50,
            "flyingKnots": 41,
        }
    },

]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
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
            solver[i].th_stop = 1e-8

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

    # Solving the problem with the DDP solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 1000, False)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]

    pinocchio.forwardKinematics(robot_model, rdata, x0[: state.nq])
    pinocchio.updateFramePlacements(robot_model, rdata)
    com = pinocchio.centerOfMass(robot_model, rdata, x0[: state.nq])
    
    tsId = robot_model.getFrameId("torso")
    torsoPos = rdata.oMf[tsId].translation
    print('com',com)
        
    rdata = robot_model.createData()

    xT = solver[i].xs[-1]
    pinocchio.forwardKinematics(robot_model, rdata, xT[: state.nq])
    pinocchio.updateFramePlacements(robot_model, rdata)
    com = pinocchio.centerOfMass(robot_model, rdata, xT[: state.nq])
    finalPosEff = np.array(
        rdata.oMf[robot_model.getFrameId("left_hand")].translation.T.flat
    )
    print('left hand:',np.array(rdata.oMf[robot_model.getFrameId("left_hand")].translation.T.flat))
    print('right hand:',np.array(rdata.oMf[robot_model.getFrameId("right_hand")].translation.T.flat))
    print('left foot:',np.array(rdata.oMf[robot_model.getFrameId("left_foot_bottom")].translation.T.flat))
    print('right foot:',np.array(rdata.oMf[robot_model.getFrameId("right_foot_bottom")].translation.T.flat))
    print(f"Distance to default state = {np.linalg.norm(x0 - np.array(xT.flat)):.3E}")


    # Plotting the entire motion




xs_traj = np.concatenate([np.array(solver[i].xs) for i in range(len(solver))], axis=0)
print(xs_traj.shape)


p_index = [0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 14,16,28,29,30,31,32,33,
            34,35,36,41,43,55,56,57,58,59,60]
v_index = [0, 1, 2, 3, 4, 5,
            6, 7, 8, 12,14,24,25,26,27,28,29,
            30,31,32,36,38,48,49,50,51,52,53]

digit_ref_traj_qpos = np.zeros((len(xs_traj),61))
digit_ref_traj_qvel = np.zeros((len(xs_traj),54))
digit_ref_traj_ee_pos = np.zeros((len(xs_traj),12))
digit_ref_traj_torque = np.zeros((len(xs_traj),20))
digit_ref_traj_com = np.zeros((len(xs_traj),3))

digit_ref_keypoints = np.zeros((len(xs_traj),15))




start_index = 0
for i in range(len(solver)):
    for j in range(len(solver[i].xs)):
        x0 = solver[i].xs[j]
        pinocchio.forwardKinematics(robot_model, rdata, x0[: state.nq])
        pinocchio.updateFramePlacements(robot_model, rdata)
        
        lh_p = np.array(rdata.oMf[robot_model.getFrameId("left_hand")].translation.T.flat)
        rh_p = np.array(rdata.oMf[robot_model.getFrameId("right_hand")].translation.T.flat)
        lf_p = np.array(rdata.oMf[robot_model.getFrameId("left_foot_bottom")].translation.T.flat)
        rf_p = np.array(rdata.oMf[robot_model.getFrameId("right_foot_bottom")].translation.T.flat)
        com = pinocchio.centerOfMass(robot_model, rdata, x0[: state.nq])
        torso = np.array(rdata.oMf[robot_model.getFrameId("torso")].translation.T.flat)
        
        digit_ref_traj_ee_pos[start_index,:] = [lh_p[0], lh_p[1], lh_p[2],
                                                rh_p[0], rh_p[1], rh_p[2],
                                                lf_p[0], lf_p[1], lf_p[2],
                                                rf_p[0], rf_p[1], rf_p[2]]
        digit_ref_traj_com[start_index,:] = com
        # digit_ref_keypoints[start_index,:] = [lh_p[0], lh_p[1], lh_p[2],
        #                                        rh_p[0], rh_p[1], rh_p[2],
        #                                        lf_p[0], lf_p[1], lf_p[2],
        #                                        rf_p[0], rf_p[1], rf_p[2],
        #                                        torso[0],torso[1],torso[2]]
        
        start_index += 1

digit_ref_traj_qpos[:, p_index] += xs_traj[:, :len(p_index)]
digit_ref_traj_qvel[:, v_index] += xs_traj[:, len(p_index):len(p_index)+len(v_index)]





# motor A left coefficients
lA00 =     0.01785;  
lA10 =     -0.9256;  
lA01 =      0.2938;  
lA20 =    -0.08362;  
lA11 =       0.103;  
lA02 =     0.06534;  
lA30 =     0.02975;  
lA21 =    -0.02949;  
lA12 =    -0.01311;  
lA03 =    -0.03942;  
lA40 =    -0.03918;  
lA31 =     0.06356;  
lA22 =     -0.0451;  
lA13 =    -0.02977;  
lA04 =   -0.003042;  
#   //motor B left coefficients
lB00 =    -0.01785;  
lB10 =      0.9257;  
lB01 =      0.2972;  
lB20 =     0.08384;  
lB11 =      0.1044;  
lB02 =    -0.06483;  
lB30 =    -0.02988;  
lB21 =    -0.02979;  
lB12 =     0.01411;  
lB03 =      -0.039;  
lB40 =     0.04013;  
lB31 =     0.06584;  
lB22 =     0.04692;  
lB13 =    -0.02893;  
lB04 =    0.003069;  
# //motor A right coefficients            
rA00 =    -0.01785;  
rA10 =     -0.9255;  
rA01 =      0.2938;  
rA20 =     0.08367;  
rA11 =     -0.1029;  
rA02 =    -0.06529;  
rA30 =      0.0297;  
rA21 =    -0.02936;  
rA12 =    -0.01315;  
rA03 =    -0.03937;  
rA40 =     0.03896;  
rA31 =    -0.06342;  
rA22 =     0.04496;  
rA13 =     0.02929;  
rA04 =    0.002823;  
# //motor B right coefficients            
rB00 =     0.01785; 
rB10 =      0.9257; 
rB01 =      0.2972; 
rB20 =    -0.08391; 
rB11 =     -0.1045; 
rB02 =     0.06483; 
rB30 =    -0.02982; 
rB21 =    -0.02973; 
rB12 =     0.01419; 
rB03 =    -0.03903; 
rB40 =    -0.03976; 
rB31 =    -0.06553; 
rB22 =    -0.04701; 
rB13 =     0.02931; 
rB04 =   -0.003061; 

for i in range(len(xs_traj)): # calculate left toe AB
    x = digit_ref_traj_qpos[i,28] # left-toe-pitch
    y = digit_ref_traj_qpos[i,29] # left-toe-roll

    digit_ref_traj_qpos[i,18] = lA00 + lA10*x + lA01*y + lA20*pow(x,2) + lA11*x*y + lA02*pow(y,2) + lA30*pow(x,3) + lA21*pow(x,2)*y + lA12*x*pow(y,2) + lA03*pow(y,3) + lA40*pow(x,4) + lA31*pow(x,3)*y + lA22*pow(x,2)*pow(y,2) + lA13*x*pow(y,3) + lA04*pow(y,4)
    digit_ref_traj_qpos[i,23] = lB00 + lB10*x + lB01*y + lB20*pow(x,2) + lB11*x*y + lB02*pow(y,2) + lB30*pow(x,3) + lB21*pow(x,2)*y + lB12*x*pow(y,2) + lB03*pow(y,3) + lB40*pow(x,4) + lB31*pow(x,3)*y + lB22*pow(x,2)*pow(y,2) + lB13*x*pow(y,3) + lB04*pow(y,4)

for i in range(len(xs_traj)): # calculate right toe AB
    x = digit_ref_traj_qpos[i,55] # right-toe-pitch
    y = digit_ref_traj_qpos[i,56] # right-toe-roll

    digit_ref_traj_qpos[i,45] = rA00 + rA10*x + rA01*y + rA20*pow(x,2) + rA11*x*y + rA02*pow(y,2) + rA30*pow(x,3) + rA21*pow(x,2)*y + rA12*x*pow(y,2) + rA03*pow(y,3) + rA40*pow(x,4) + rA31*pow(x,3)*y + rA22*pow(x,2)*pow(y,2) + rA13*x*pow(y,3) + rA04*pow(y,4)
    digit_ref_traj_qpos[i,50] = rB00 + rB10*x + rB01*y + rB20*pow(x,2) + rB11*x*y + rB02*pow(y,2) + rB30*pow(x,3) + rB21*pow(x,2)*y + rB12*x*pow(y,2) + rB03*pow(y,3) + rB40*pow(x,4) + rB31*pow(x,3)*y + rB22*pow(x,2)*pow(y,2) + rB13*x*pow(y,3) + rB04*pow(y,4)

for i in range(1,len(xs_traj)): # calculate toe AB velocity
    digit_ref_traj_qvel[i,16] = (digit_ref_traj_qpos[i,18]-digit_ref_traj_qpos[i-1,18])/GAITPHASES[0]['walking']['timeStep']
    digit_ref_traj_qvel[i,20] = (digit_ref_traj_qpos[i,23]-digit_ref_traj_qpos[i-1,23])/GAITPHASES[0]['walking']['timeStep']
    digit_ref_traj_qvel[i,40] = (digit_ref_traj_qpos[i,45]-digit_ref_traj_qpos[i-1,45])/GAITPHASES[0]['walking']['timeStep']
    digit_ref_traj_qvel[i,44] = (digit_ref_traj_qpos[i,50]-digit_ref_traj_qpos[i-1,50])/GAITPHASES[0]['walking']['timeStep']



digit_ref_traj = np.concatenate((digit_ref_traj_qpos,
                                    digit_ref_traj_qvel,
                                    digit_ref_traj_ee_pos,
                                    digit_ref_traj_torque,
                                    digit_ref_traj_com,
                                    ),axis=1)


dir_path = '/home/fliu305/'
filename = 'crocoddyl_ref_jumping'
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename_with_time = f"{filename}_{current_time}"
ref_full_path = os.path.join(dir_path, filename_with_time)
np.savetxt(ref_full_path, digit_ref_traj, delimiter=',', fmt='%s', comments='')
print("shape",digit_ref_traj.shape)





# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(digit_legs, frameNames=[rightFoot, leftFoot])
    display.rate = -1
    display.freq = 1
    for j in range (2):
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i],factor=1)
