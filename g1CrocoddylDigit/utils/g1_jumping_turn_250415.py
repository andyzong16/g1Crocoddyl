import numpy as np
import pinocchio
import sys
import crocoddyl
from scipy.interpolate import CubicSpline


class SimpleBipedGaitProblem:
    """Build simple bipedal locomotion problems.

    This class aims to build simple locomotion problems used in the examples of
    Crocoddyl.
    The scope of this class is purely for academic reasons, and it does not aim to be
    used in any robotics application.
    We also do not consider it as part of the API, so changes in this class will not
    pass through a strict process of deprecation.
    Thus, we advice any user to DO NOT develop their application based on this class.
    """

    def __init__(
        self,
        rmodel,
        rightHand,
        leftHand,
        rightFoot,
        leftFoot,
        integrator="euler",
        control="zero",
        fwddyn=True,
    ):
        """Construct biped-gait problem.

        :param rmodel: robot model
        :param rightFoot: name of the right foot
        :param leftFoot: name of the left foot
        :param integrator: type of the integrator
            (options are: 'euler', and 'rk4')
        :param control: type of control parametrization
            (options are: 'zero', 'one', and 'rk4')
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.rhId = self.rmodel.getFrameId(rightHand)
        self.lhId = self.rmodel.getFrameId(leftHand)
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)
        self.tsId = self.rmodel.getFrameId("torso_link")
        
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn
        # Defining default state
        self.q0 = np.array([
            # base position — lower to reflect crouched posture
            0.0, 0.0, 0.60,

            # base orientation quaternion
            0.0, 0.0, 0.0, 1.0,

            # LEFT LEG — deeper crouch
            -0.5,    # hip_pitch (more flexion)
            0.0,     # hip_roll
            0.0,     # hip_yaw
            1.0,     # knee (deeper bend)
            -0.5,    # ankle_pitch (more dorsiflexion)
            0.0,     # ankle_roll

            # RIGHT LEG
            -0.5,
            0.0,
            0.0,
            1.0,
            -0.5,
            0.0,

            # WAIST — slight forward lean helps push-off
            0.0,   # yaw
            0.0,   # roll
            0.1,   # pitch

            # LEFT ARM — pulled back slightly for counter-swing
            0.1,    # shoulder_pitch
            0.15,   # shoulder_roll
            0.0,    # shoulder_yaw
            -0.3,   # elbow
            0.0,
            0.0,
            0.0,

            # RIGHT ARM
            0.1,
            -0.15,
            0.0,
            -0.3,
            0.0,
            0.0,
            0.0
        ])
        
        self.rmodel.defaultState = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)
        self.x0 = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])


    def createJumpingProblem(
        self,
        x0,
        jumpHeight,
        jumpLength,
        residual,
        jumpingangle,
        timeStep,
        standingKnots,
        pretakeoffKnots,
        groundKnots,
        flyingKnots,
        final=False,
    ):
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfId].translation.copy()
        lfFootPos0 = self.rdata.oMf[self.lfId].translation.copy()
        foot_distance = np.sqrt((rfFootPos0[0] - lfFootPos0[0])**2 + (rfFootPos0[1] - lfFootPos0[1])**2)
        rfFootPos0[2] = 0.0
        lfFootPos0[2] = 0.0
        comRef = (rfFootPos0 + lfFootPos0) / 2
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2]
        refTorso = self.rdata.oMf[self.tsId].translation.copy()

        self.rWeight = 1e1
        loco3dModel = []

        # standing
        standing_com = comRef.copy()
        standing_com[0] += 0.1
        target_hand = np.array([[0.5,  0.20, 1.3],
                                [0.5, -0.20, 1.3]])
        handTask = [
            [self.lhId, pinocchio.SE3(np.eye(3), target_hand[0])],
            [self.rhId, pinocchio.SE3(np.eye(3), target_hand[1])],
        ]
        standing = self.createSwingFootModel_standing(
            timeStep, standingKnots, [self.lfId, self.rfId], standing_com, handTask
        )

        # pretakeoff
        pretakeoff_com = comRef.copy()
        pretakeOff = self.createSwingFootModel_pretakeOff(
            timeStep, pretakeoffKnots, [self.lfId, self.rfId], comTask=pretakeoff_com,
        )

        # takeoff
        takeoff_com = comRef.copy()
        takeoff_com[2] += 0.25
        takeOff = self.createSwingFootModel_takeOff(
            timeStep, groundKnots, [self.lfId, self.rfId],
            comTask=takeoff_com,
            swingFootTask=None,
            handTask=None,
        )

        # flyingUp
        comRef_flying = comRef.copy()
        comRef_flying[0] += jumpLength * np.cos(jumpingangle)
        comRef_flying[1] += jumpLength * np.sin(jumpingangle)
        comRef_flying[2] += jumpHeight + 0.15

        jump_vec = np.array([
            jumpLength * np.cos(jumpingangle),
            jumpLength * np.sin(jumpingangle),
            jumpHeight,
        ])

        key_time_points = np.array([0, 0.2, 0.5, 0.8, 1])
        key_foot_angles = np.array([
            0,
            jumpingangle * 0.1,
            jumpingangle * 0.25,
            jumpingangle * 0.4,
            jumpingangle * 0.5,
        ])
        cs_angle = CubicSpline(key_time_points, key_foot_angles)
        time_interpolated = np.linspace(0, 1, flyingKnots)
        angle_interpolated = cs_angle(time_interpolated)

        flyingUpPhase = []
        for k in range(flyingKnots):
            frac = (k + 1) / flyingKnots
            com_up = comRef.copy()
            com_up[0] += jumpLength * np.cos(jumpingangle) * frac
            com_up[1] += jumpLength * np.sin(jumpingangle) * frac
            com_up[2] += (jumpHeight + 0.15) * frac
            flyingUpPhase += [self.createSwingFootModel_flyingUp(
                timeStep,
                [],
                comTask=com_up,
                swingFootTask=[
                    [self.lfId, pinocchio.SE3(np.eye(3), lfFootPos0 + jump_vec * frac)],
                    [self.rfId, pinocchio.SE3(np.eye(3), rfFootPos0 + jump_vec * frac)],
                ],
            )]

        # flyingDown — apex to landing
        flyingDownPhase = []
        for k in range(flyingKnots):
            frac = (k + 1) / flyingKnots 
            lf_target = lfFootPos0 + np.array([
                jumpLength * np.cos(jumpingangle),
                jumpLength * np.sin(jumpingangle),
                jumpHeight * (1.0 - frac),
            ])
            rf_target = rfFootPos0 + np.array([
                jumpLength * np.cos(jumpingangle),
                jumpLength * np.sin(jumpingangle),
                jumpHeight * (1.0 - frac),
            ])
            com_down = comRef.copy()
            com_down[0] += jumpLength * np.cos(jumpingangle)
            com_down[1] += jumpLength * np.sin(jumpingangle)
            com_down[2] += (jumpHeight + 0.15) * (1.0 - frac)
            flyingDownPhase += [self.createSwingFootModel_down(
                timeStep,
                [],
                comTask=com_down,
                swingFootTask=[
                    [self.lfId, pinocchio.SE3(np.eye(3), lf_target)],
                    [self.rfId, pinocchio.SE3(np.eye(3), rf_target)],
                ],
            )]

        # landing
        roll, pitch, yaw = 0., 0., jumpingangle
        rotation_matrix = pinocchio.utils.rpyToMatrix(roll, pitch, yaw)
        rotation_array = np.array(rotation_matrix)

        footTask = [
            [self.lfId, pinocchio.SE3(
                rotation_array,
                lfFootPos0 + np.array([
                    -0.5 * foot_distance * np.sin(jumpingangle) - residual * np.sin(jumpingangle) + jumpLength * np.cos(jumpingangle),
                    -0.5 * foot_distance * (1 - np.cos(jumpingangle)) + residual * np.cos(jumpingangle) + jumpLength * np.sin(jumpingangle),
                    0.0,
                ])
            )],
            [self.rfId, pinocchio.SE3(
                rotation_array,
                rfFootPos0 + np.array([
                    0.5 * foot_distance * np.sin(jumpingangle) + residual * np.sin(jumpingangle) + jumpLength * np.cos(jumpingangle),
                    0.5 * foot_distance * (1 - np.cos(jumpingangle)) - residual * np.cos(jumpingangle) + jumpLength * np.sin(jumpingangle),
                    0.0,
                ])
            )],
        ]

        torsoTask = [
            [self.tsId, pinocchio.SE3(rotation_array, refTorso)],
        ]

        landingPhase = [self.createFootSwitchModel([self.lfId, self.rfId], footTask, torsoTask, False)]

        if final is True:
            self.rWeight = 1e4

        landed = [
            self.createSwingFootModel(
                timeStep,
                [self.lfId, self.rfId],
                comTask=comRef + np.array([
                    jumpLength * np.cos(jumpingangle),
                    jumpLength * np.sin(jumpingangle),
                    0.0,
                ]),
                swingFootTask=footTask,
                torsoTask=torsoTask,
            )
            for k in range(groundKnots * 12)
        ]

        # loco3dModel += everything
        loco3dModel += pretakeOff
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        # Rescale terminal weights
        costs = loco3dModel[-1].differential.costs.costs.todict()
        for c in costs.values():
            c.weight *= timeStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem

    def createSwingFootModel_standing(self, timeStep, pretakeoffKnots, supportFootIds, comTask=None, swingHandTask=None):
        
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
        
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        costModel = crocoddyl.CostModelSum(self.state, nu)

        if isinstance(comTask, np.ndarray):
            w_com = np.array([1.0, 1.0, 0.0])
            activation_com = crocoddyl.ActivationModelWeightedQuad(w_com**2)
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, activation_com, comResidual)
            # costModel.addCost("comTrack", comTrack, 1e6)

        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.06, 0.03]))  # G1 foot half-extents
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e4)

        if swingHandTask is not None:
            w_hand_0 = np.array([1] * 3 + [0.0001] * 3)
            activation_hand = crocoddyl.ActivationModelWeightedQuad(w_hand_0**2)
            for i in swingHandTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                handTrack = crocoddyl.CostModelResidual(self.state, activation_hand, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_handTrack", handTrack, 1e6)

        # Cost for torque limit — pad bounds to nu for inv-dynamics mode
        u_ub = self.rmodel.effortLimit[6:]
        u_lb = -self.rmodel.effortLimit[6:]
        if not self._fwddyn:
            n_cf = nu - self.actuation.nu
            inf_pad = sys.float_info.max * np.ones(n_cf)
            u_ub = np.concatenate([u_ub, inf_pad])
            u_lb = np.concatenate([u_lb, -inf_pad])
        u_bounds = crocoddyl.ActivationBounds(u_lb, u_ub, 1.0)
        uLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(u_bounds)
        uLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
        torqueLimitCost = crocoddyl.CostModelResidual(self.state, uLimitActivation, uLimitResidual)
        costModel.addCost("torque_limit", torqueLimitCost, 1.0)

        # Adding the state limits penalization
        x_lb = np.concatenate([self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]])
        x_ub = np.concatenate([self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]])
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub))
        x_bounds = crocoddyl.CostModelResidual(self.state, activation_xbounds, crocoddyl.ResidualModelState(self.state, 0 * self.x0, nu))
        costModel.addCost("xBounds", x_bounds, 1.0)

        # Cost for self-collision
        maxfloat = sys.float_info.max
        xlb = np.concatenate([
            -maxfloat * np.ones(6),
            self.rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(self.state.nv),
        ])
        xub = np.concatenate([
            maxfloat * np.ones(6),
            self.rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(self.state.nv),
        ])
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(self.state, self.x0, nu)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        limitCost = crocoddyl.CostModelResidual(self.state, xLimitActivation, xLimitResidual)
        costModel.addCost("limitCost", limitCost, 1e3)

        stateWeights = np.concatenate([
        np.array([0.0, 0.0, 100.0]),
        np.array([500.0, 500.0, 500.0]),
        np.ones(self.state.nv - 6) * 0.01,
        np.array([10.0] * self.state.nv)
    ])
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)

        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, timeStep)

        return [model] * pretakeoffKnots
    
    def createSwingFootModel_pretakeOff(self, timeStep, pretakeoffKnots, supportFootIds, comTask=None):
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)

        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            contact = crocoddyl.ContactModel6D(
                self.state, i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", contact)

        costModel = crocoddyl.CostModelSum(self.state, nu)

        if isinstance(comTask, np.ndarray):
            w_com = np.array([1.0, 1.0, 1.0])
            activation_com = crocoddyl.ActivationModelWeightedQuad(w_com**2)
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, activation_com, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)

        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.06, 0.03]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e4)

        # Torque limit
        u_ub = self.rmodel.effortLimit[6:]
        u_lb = -self.rmodel.effortLimit[6:]
        if not self._fwddyn:
            n_cf = nu - self.actuation.nu
            if n_cf > 0:
                inf_pad = sys.float_info.max * np.ones(n_cf)
                u_ub = np.concatenate([u_ub, inf_pad])
                u_lb = np.concatenate([u_lb, -inf_pad])
        u_bounds = crocoddyl.ActivationBounds(u_lb, u_ub, 1.0)
        uLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(u_bounds)
        uLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
        torqueLimitCost = crocoddyl.CostModelResidual(
            self.state, uLimitActivation, uLimitResidual
        )
        costModel.addCost("torque_limit", torqueLimitCost, 1.0)

        # State weights — penalize Z drop and orientation during crouch
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 100.0]),
            np.array([500.0, 500.0, 500.0]),
            np.ones(self.state.nv - 6) * 0.01,
            np.array([10.0] * self.state.nv)
        ])
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)

        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, timeStep)

        return [model] * pretakeoffKnots
    
    def createSwingFootModel_takeOff(self, timeStep, groundKnots, supportFootIds, comTask=None, swingFootTask=None, handTask=None):

        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)

        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            contact = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", contact)

        costModel = crocoddyl.CostModelSum(self.state, nu)

        # Rising CoM target
        if isinstance(comTask, np.ndarray):
            w_com = np.array([100.0, 100.0, 100.0])
            activation_com = crocoddyl.ActivationModelWeightedQuad(w_com**2)
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, activation_com, comResidual)
            costModel.addCost("comTrack", comTrack, 1e8)

        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                footTrack = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8)

        # Wrench cone 
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.06, 0.03]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(self.state, i, cone, nu, self._fwddyn)
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(self.state, wrenchActivation, wrenchResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e4)

        # Torque limit 
        u_ub = self.rmodel.effortLimit[6:]
        u_lb = -self.rmodel.effortLimit[6:]
        if not self._fwddyn:
            n_cf = nu - self.actuation.nu
            inf_pad = sys.float_info.max * np.ones(n_cf)
            u_ub = np.concatenate([u_ub, inf_pad])
            u_lb = np.concatenate([u_lb, -inf_pad])
        u_bounds = crocoddyl.ActivationBounds(u_lb, u_ub, 1.0)
        uLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(u_bounds)
        uLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
        torqueLimitCost = crocoddyl.CostModelResidual(self.state, uLimitActivation, uLimitResidual)
        costModel.addCost("torque_limit", torqueLimitCost, 1.0)

        # State limits
        x_lb = np.concatenate([self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]])
        x_ub = np.concatenate([self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]])
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub))
        x_bounds = crocoddyl.CostModelResidual(self.state, activation_xbounds, crocoddyl.ResidualModelState(self.state, 0 * self.x0, nu))
        costModel.addCost("xBounds", x_bounds, 1.0)

        # Joint position limits
        maxfloat = sys.float_info.max
        xlb = np.concatenate([
            -maxfloat * np.ones(6),
            self.rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(self.state.nv),
        ])
        xub = np.concatenate([
            maxfloat * np.ones(6),
            self.rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(self.state.nv),
        ])
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(self.state, self.x0, nu)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        limitCost = crocoddyl.CostModelResidual(self.state, xLimitActivation, xLimitResidual)
        costModel.addCost("limitCost", limitCost, 1e3)

        # State regularization
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 100.0]),
            np.array([500.0, 500.0, 500.0]),   
            np.ones(self.state.nv - 6) * 0.01,
            np.array([10.0] * self.state.nv)
        ])
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)

        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e-1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-2)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, timeStep)

        return [model] * groundKnots
    
    def createSwingFootModel_flyingUp(
        self, timeStep, supportFootIds=[], comTask=None, swingFootTask=None
    ):
        # During flight supportFootIds is always none
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)

        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        # no contacts added

        costModel = crocoddyl.CostModelSum(self.state, nu)

        # CoM tracking toward apex
        if isinstance(comTask, np.ndarray):
            w_com = np.array([100.0, 100.0, 10.0])
            activation_com = crocoddyl.ActivationModelWeightedQuad(w_com**2)
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, activation_com, comResidual)
            costModel.addCost("comTrack", comTrack, 1e8)

        # Foot targets during ascent
        if swingFootTask is not None:
            for i in swingFootTask:
                w_foot = np.array([1.0] * 3 + [0.0] * 3)
                activation_foot = crocoddyl.ActivationModelWeightedQuad(w_foot**2)
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, activation_foot, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8
                )

        # Torque limit
        u_ub = self.rmodel.effortLimit[6:]
        u_lb = -self.rmodel.effortLimit[6:]
        if not self._fwddyn:
            n_cf = nu - self.actuation.nu
            inf_pad = sys.float_info.max * np.ones(n_cf)
            u_ub = np.concatenate([u_ub, inf_pad])
            u_lb = np.concatenate([u_lb, -inf_pad])
        u_bounds = crocoddyl.ActivationBounds(u_lb, u_ub, 1.0)
        uLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(u_bounds)
        uLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
        torqueLimitCost = crocoddyl.CostModelResidual(
            self.state, uLimitActivation, uLimitResidual
        )
        costModel.addCost("torque_limit", torqueLimitCost, 1.0)

        # State regularization
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            np.ones(self.state.nv - 6) * 0.01,
            np.array([50.0] * self.state.nv)
        ])
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )

        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e-1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-2)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, timeStep)

        return model
    
    def createSwingFootModel_down(
        self, timeStep, supportFootIds, comTask=None, swingFootTask=None
    ):
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)

        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        costModel = crocoddyl.CostModelSum(self.state, nu)

        # CoM tracking during descent
        if isinstance(comTask, np.ndarray):
            w_com = np.array([100.0, 100.0, 10.0])
            activation_com = crocoddyl.ActivationModelWeightedQuad(w_com**2)
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, activation_com, comResidual)
            # costModel.addCost("comTrack", comTrack, 1e6)

        # Foot targets during descent
        if swingFootTask is not None:
            for i in swingFootTask:
                w_foot = np.array([1] * 3 + [0] * 3)
                activation_foot = crocoddyl.ActivationModelWeightedQuad(w_foot**2)
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                footTrack = crocoddyl.CostModelResidual(self.state, activation_foot, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        # stateWeights must be size ndx=2*nv=70
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            np.ones(self.state.nv - 6) * 0.01,
            np.array([50.0] * self.state.nv)
        ])
        
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)

        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        # Adding the state limits penalization
        x_lb = np.concatenate([self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]])
        x_ub = np.concatenate([self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]])
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub))
        x_bounds = crocoddyl.CostModelResidual(self.state, activation_xbounds, crocoddyl.ResidualModelState(self.state, 0 * self.x0, nu))
        costModel.addCost("xBounds", x_bounds, 1.0)

        # Cost for self-collision
        maxfloat = sys.float_info.max
        xlb = np.concatenate([
            -maxfloat * np.ones(6),
            self.rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(self.state.nv),
        ])
        xub = np.concatenate([
            maxfloat * np.ones(6),
            self.rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(self.state.nv),
        ])
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(self.state, self.x0, nu)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        limitCost = crocoddyl.CostModelResidual(self.state, xLimitActivation, xLimitResidual)
        costModel.addCost("limitCost", limitCost, 1e3)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, timeStep)

        return model
    
    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None, torsoTask=None):

        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)

        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        costModel = crocoddyl.CostModelSum(self.state, nu)

        # Adding the state limits penalization
        x0 = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])
        x_lb = np.concatenate([self.state.lb[1 : self.state.nv + 1], self.state.lb[-self.state.nv :]])
        x_ub = np.concatenate([self.state.ub[1 : self.state.nv + 1], self.state.ub[-self.state.nv :]])
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub))
        x_bounds = crocoddyl.CostModelResidual(self.state, activation_xbounds, crocoddyl.ResidualModelState(self.state, 0 * self.x0, nu))
        costModel.addCost("xBounds", x_bounds, 1.0)

        # Cost for self-collision
        maxfloat = sys.float_info.max
        xlb = np.concatenate([
            -maxfloat * np.ones(6),
            self.rmodel.lowerPositionLimit[7:],
            -maxfloat * np.ones(self.state.nv),
        ])
        xub = np.concatenate([
            maxfloat * np.ones(6),
            self.rmodel.upperPositionLimit[7:],
            maxfloat * np.ones(self.state.nv),
        ])
        bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
        xLimitResidual = crocoddyl.ResidualModelState(self.state, self.x0, nu)
        xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
        limitCost = crocoddyl.CostModelResidual(self.state, xLimitActivation, xLimitResidual)
        costModel.addCost("limitCost", limitCost, 1e3)

        if isinstance(comTask, np.ndarray):
            w_com = np.array([100.0, 100.0, 10.0])
            activation_com = crocoddyl.ActivationModelWeightedQuad(w_com**2)
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, activation_com, comResidual)
            # costModel.addCost("comTrack", comTrack, 1e8)

        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.06, 0.03]))  # G1 foot half-extents
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(self.state, i, cone, nu, self._fwddyn)
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            wrenchCone = crocoddyl.CostModelResidual(self.state, wrenchActivation, wrenchResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1)

        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                footTrack = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e9)

        if torsoTask is not None:
            for i in torsoTask:
                w_torso = np.array([0] * 3 + [1] * 3)
                activation_torso = crocoddyl.ActivationModelWeightedQuad(w_torso**2)
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                torsoTrack = crocoddyl.CostModelResidual(self.state, activation_torso, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_torsoTrack", torsoTrack, 1e8)

        # Cost for torque limit — pad bounds to nu for inv-dynamics mode
        u_ub = self.rmodel.effortLimit[6:]
        u_lb = -self.rmodel.effortLimit[6:]
        if not self._fwddyn:
            n_cf = nu - self.actuation.nu
            inf_pad = sys.float_info.max * np.ones(n_cf)
            u_ub = np.concatenate([u_ub, inf_pad])
            u_lb = np.concatenate([u_lb, -inf_pad])
        u_bounds = crocoddyl.ActivationBounds(u_lb, u_ub, 1.0)
        uLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(u_bounds)
        uLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
        torqueLimitCost = crocoddyl.CostModelResidual(self.state, uLimitActivation, uLimitResidual)
        costModel.addCost("torque_limit", torqueLimitCost, 1.0)

        # stateWeights must be size ndx=2*nv=70, not the old hardcoded Digit per-joint array
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            np.ones(self.state.nv - 6) * 0.01,
            np.array([50.0] * self.state.nv)
        ])
        
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)

        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.four)
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(nu, crocoddyl.RKType.three)
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK4(dmodel, control, timeStep)

        return model

    def createFootSwitchModel(self, supportFootIds, swingFootTask, torsoTask, pseudoImpulse=True):
        """Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask, torsoTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask, torsoTask):

        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)

        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        costModel = crocoddyl.CostModelSum(self.state, nu)

        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.06, 0.03]))  # G1 foot half-extents
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(self.state, i, cone, nu, self._fwddyn)
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            wrenchCone = crocoddyl.CostModelResidual(self.state, wrenchActivation, wrenchResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1)

        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                footTrack = crocoddyl.CostModelResidual(self.state, framePlacementResidual)
                impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_impulseVel", impulseFootVelCost, 1e8)

        if torsoTask is not None:
            for i in torsoTask:
                w_torso = np.array([0] * 3 + [1] * 3)
                activation_torso = crocoddyl.ActivationModelWeightedQuad(w_torso**2)
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(self.state, i[0], i[1], nu)
                torsoTrack = crocoddyl.CostModelResidual(self.state, activation_torso, framePlacementResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_torsoTrack", torsoTrack, 1e8)

        # stateWeights must be size ndx=2*nv=70
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            np.ones(self.state.nv - 6) * 0.01,
            np.array([50.0] * self.state.nv)
        ])
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)

        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(self.state, self.actuation, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)

        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Cost for torque limit — pad bounds to nu for inv-dynamics mode
        u_ub = self.rmodel.effortLimit[6:]
        u_lb = -self.rmodel.effortLimit[6:]
        if not self._fwddyn:
            n_cf = nu - self.actuation.nu
            inf_pad = sys.float_info.max * np.ones(n_cf)
            u_ub = np.concatenate([u_ub, inf_pad])
            u_lb = np.concatenate([u_lb, -inf_pad])
        u_bounds = crocoddyl.ActivationBounds(u_lb, u_ub, 1.0)
        uLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(u_bounds)
        uLimitResidual = crocoddyl.ResidualModelControl(self.state, nu)
        torqueLimitCost = crocoddyl.CostModelResidual(self.state, uLimitActivation, uLimitResidual)
        costModel.addCost("torque_limit", torqueLimitCost, 1.0)

        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.four, 0.0)
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.three, 0.0)
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)

        return model

    def createImpulseModel(self, supportFootIds, swingFootTask):

        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(
                self.state, i, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
                    self.state, i[0], i[1].translation, 0
                )
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8)

        # stateWeights must be size ndx=2*nv=70
        stateWeights = np.concatenate([
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            np.ones(self.state.nv - 6) * 0.01,
            np.array([50.0] * self.state.nv)
        ])
        
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, 0)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1e1)

        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        return model


def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt

    xs, us, cs = [], [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []

    def updateTrajectories(solver):
        xs.extend(solver.xs[:-1])
        for m, d in zip(solver.problem.runningModels, solver.problem.runningDatas):
            if hasattr(m, "differential"):
                cs.append(d.differential.multibody.pinocchio.com[0])
                us.append(d.differential.multibody.joint.tau)
                if bounds and isinstance(
                    m.differential, crocoddyl.DifferentialActionModelContactFwdDynamics
                ):
                    us_lb.extend([m.u_lb])
                    us_ub.extend([m.u_ub])
            else:
                cs.append(d.multibody.pinocchio.com[0])
                us.append(np.zeros(nu))
                if bounds:
                    us_lb.append(np.nan * np.ones(nu))
                    us_ub.append(np.nan * np.ones(nu))
            if bounds:
                xs_lb.extend([m.state.lb])
                xs_ub.extend([m.state.ub])

    if isinstance(solver, list):
        for s in solver:
            rmodel = solver[0].problem.runningModels[0].state.pinocchio
            nq, nv, nu = (
                rmodel.nq,
                rmodel.nv,
                solver[0].problem.runningModels[0].differential.actuation.nu,
            )
            updateTrajectories(s)
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        nq, nv, nu = (
            rmodel.nq,
            rmodel.nv,
            solver.problem.runningModels[0].differential.actuation.nu,
        )
        updateTrajectories(solver)

    # Getting the state and control trajectories
    nx = nq + nv
    X = [0.0] * nx
    U = [0.0] * nu
    if bounds:
        U_LB = [0.0] * nu
        U_UB = [0.0] * nu
        X_LB = [0.0] * nx
        X_UB = [0.0] * nx
    for i in range(nx):
        X[i] = [x[i] for x in xs]
        if bounds:
            X_LB[i] = [x[i] for x in xs_lb]
            X_UB[i] = [x[i] for x in xs_ub]
    for i in range(nu):
        U[i] = [u[i] for u in us]
        if bounds:
            U_LB[i] = [u[i] for u in us_lb]
            U_UB[i] = [u[i] for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ["1", "2", "3", "4", "5", "6"]
    # left foot
    plt.subplot(2, 3, 1)
    plt.title("joint position [rad]")
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(7, 13))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(7, 13))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title("joint velocity [rad/s]")
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 6, nq + 12))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 12))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title("joint torque [Nm]")
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 6))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(0, 6))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(0, 6))]
    plt.ylabel("LF")
    plt.legend()

    # right foot
    plt.subplot(2, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 19))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(13, 19))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(13, 19))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(2, 3, 5)
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 12, nq + 18))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 18))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(6, 12))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(6, 12))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()

    plt.figure(figIndex + 1)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[: rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(c[0])
        Cy.append(c[1])
    plt.plot(Cx, Cy)
    plt.title("CoM position")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    if show:
        plt.show()