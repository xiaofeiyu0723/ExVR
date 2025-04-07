r"""
    Wrapper for RBDL model.
"""


__all__ = ['RBDLModel']


import rbdl
import numpy as np
from ...math import adjoint_transformation_matrix_np


class RBDLModel:
    def __init__(self, model_file: str, gravity=np.array((0, -9.81, 0)), update_kinematics_by_hand=False):
        r"""
        Init an RBDL model. (numpy, single)

        :param model_file: Robot urdf file path.
        :param gravity: Vector3 for gravity.
        :param update_kinematics_by_hand: If True, user should call update_kinematics() by hand at proper time.
                                          Set True only if you know what you are doing.
        """
        model = rbdl.loadModel(model_file.encode())
        model.gravity = gravity
        self.model = model
        self.q_size = model.q_size
        self.qdot_size = model.qdot_size
        self.uk = not update_kinematics_by_hand

    def forward_dynamics(self, q, qdot, tau):
        r"""
        Compute forward dynamics with the Articulated Body Algorithm.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param tau: Robot joint torque tau in shape [dof].
        :return: Robot acceleration qddot in shape [dof].
        """
        qddot = np.zeros(self.qdot_size)
        rbdl.ForwardDynamics(self.model, q, qdot, tau, qddot)
        return qddot

    def inverse_dynamics(self, q, qdot, qddot):
        r"""
        Compute inverse dynamics with the Newton-Euler Algorithm.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param qddot: Robot acceleration qddot in shape [dof].
        :return: Robot joint torque tau in shape [dof].
        """
        tau = np.zeros(self.qdot_size)
        rbdl.InverseDynamics(self.model, q, qdot, qddot, tau)
        return tau

    def calc_M(self, q):
        r"""
        Calculate the inertia matrix M(q) of the robot.

        :param q: Robot configuration q in shape [dof].
        :return: Inertial matrix M in shape [dof, dof].
        """
        M = np.zeros((self.qdot_size, self.qdot_size))
        rbdl.CompositeRigidBodyAlgorithm(self.model, q, M, update_kinematics=self.uk)
        return M

    def calc_h(self, q, qdot):
        r"""
        Calculate the h(q, qdot) of the robot.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :return: h in shape [dof].
        """
        h = np.zeros(self.qdot_size)
        rbdl.NonlinearEffects(self.model, q, qdot, h)
        return h

    def calc_body_to_base_coordinates(self, q, body, coordinates_in_body_frame=np.zeros(3)):
        r"""
        Transform a point expressed in the body frame to the base frame.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :param coordinates_in_body_frame: Vector3 for the coordinates of the point expressed in the body frame.
        :return: Ndarray in shape [3] for the point coordinates expressed in the base frame.
        """
        p = rbdl.CalcBodyToBaseCoordinates(self.model, q, body.value, coordinates_in_body_frame,
                                           update_kinematics=self.uk)
        return p

    def calc_base_to_body_coordinates(self, q, body, coordinates_in_base_frame=np.zeros(3)):
        r"""
        Transform a point expressed in the base frame to the body frame.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :param coordinates_in_base_frame: Vector3 for the coordinates of the point expressed in the base frame.
        :return: Ndarray in shape [3] for the point coordinates expressed in the body frame.
        """
        p = rbdl.CalcBaseToBodyCoordinates(self.model, q, body.value, coordinates_in_base_frame,
                                           update_kinematics=self.uk)
        return p

    def calc_body_position(self, q, body):
        r"""
        Calculate the global position of a body.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :return: Ndarray in shape [3] for the body position.
        """
        return self.calc_body_to_base_coordinates(q, body)

    def calc_body_orientation(self, q, body):
        r"""
        Calculate the global orientation of a body.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :return: Ndarray in shape [3, 3] for the body orientation.
        """
        return rbdl.CalcBodyWorldOrientation(self.model, q, body.value, update_kinematics=self.uk).T

    def calc_body_Jacobian(self, q, body):
        r"""
        Calculate the 6D Jacobian of a body expressed in its own frame.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :return: Ndarray in shape [6, dof] for body Jacobian.
        """
        J = np.zeros([6, self.qdot_size])
        rbdl.CalcBodySpatialJacobian(self.model, q, body.value, np.zeros(3), J, update_kinematics=self.uk)
        return J

    def calc_space_Jacobian(self, q, body):
        r"""
        Calculate the 6D Jacobian of a body expressed in the base frame.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :return: Ndarray in shape [6, dof] for space Jacobian.
        """
        Jb = self.calc_body_Jacobian(q, body)
        Rsb = self.calc_body_orientation(q, body)
        ps = self.calc_body_position(q, body)
        ADTsb = adjoint_transformation_matrix_np(Rsb, ps)
        Js = np.dot(ADTsb, Jb)
        return Js

    def calc_point_Jacobian(self, q, body, coordinates_in_body_frame=np.zeros(3)):
        r"""
        Calculate the 3D Jacobian of a point on a body expressed in the base frame.

        :math:`J_s \dot q = \dot r_s`, i.e. the global velocity of the point in the base frame.

        :param q: Robot configuration q in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :param coordinates_in_body_frame: Vector3 for the coordinates of the point expressed in the body frame.
        :return: Ndarray in shape [3, dof] for the point Jacobian.
        """
        J = np.zeros([3, self.qdot_size])
        rbdl.CalcPointJacobian(self.model, q, body.value, coordinates_in_body_frame, J, update_kinematics=self.uk)
        return J

    def calc_point_acceleration(self, q, qdot, qddot, body, coordinates_in_body_frame=np.zeros(3)):
        r"""
        Calculate the linear acceleration of a point on a body expressed in the base frame.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param qddot: Robot acceleration qddot in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :param coordinates_in_body_frame: Vector3 for the coordinates of the point expressed in the body frame.
        :return: Ndarray in shape [3] for the point acceleration.
        """
        acc = rbdl.CalcPointAcceleration(self.model, q, qdot, qddot, body.value, coordinates_in_body_frame,
                                         update_kinematics=self.uk)
        return acc

    def calc_point_velocity(self, q, qdot, body, coordinates_in_body_frame=np.zeros(3)):
        r"""
        Calculate the linear velocity of a point on a body expressed in the base frame.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param body: An enum obj where body.value should be the desired body id.
        :param coordinates_in_body_frame: Vector3 for the coordinates of the point expressed in the body frame.
        :return: Ndarray in shape [3] for the point velocity.
        """
        vel = rbdl.CalcPointVelocity(self.model, q, qdot, body.value, coordinates_in_body_frame,
                                     update_kinematics=self.uk)
        return vel

    def calc_center_of_mass_position(self, q, qdot):
        r"""
        Calculate the total mass and the location of center of mass of the robot.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :return: Total mass (float) and the location of center of mass (ndarray in shape [3]).
        """
        com = np.zeros(3)
        mass = rbdl.CalcCenterOfMass(self.model, q, qdot, None, com, update_kinematics=self.uk)
        return mass, com

    def calc_center_of_mass_position_velocity(self, q, qdot):
        r"""
        Calculate the total mass, the location, velocity, and angular momentum of center of mass of the robot.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :return: Total mass (float),
                 location of center of mass (com) in base frame (ndarray in shape [3]),
                 linear velocity of com in base frame (ndarray in shape [3]), and
                 angular momentum of the robot at com in base frame (ndarray in shape [3]).
        """
        com = np.zeros(3)
        com_velocity = np.zeros(3)
        angular_momentum = np.zeros(3)
        mass = rbdl.CalcCenterOfMass(self.model, q, qdot, None, com, com_velocity, None, angular_momentum,
                                     update_kinematics=self.uk)
        return mass, com, com_velocity, angular_momentum

    def calc_center_of_mass_position_velocity_acceleration(self, q, qdot, qddot):
        r"""
        Calculate the total mass, the location, velocity, angular momentum, acceleration, and change of
        angular momentum of center of mass of the robot.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param qddot: Robot acceleration qddot in shape [dof].
        :return: Total mass (float),
                 location of center of mass (com) in base frame (ndarray in shape [3]),
                 linear velocity of com in base frame (ndarray in shape [3]),
                 angular momentum of the robot at com in base frame (ndarray in shape [3]),
                 linear acceleration of com in base frame (ndarray in shape [3]), and
                 change of angular momentum of the robot at com in base frame (ndarray in shape [3]).
        """
        com = np.zeros(3)
        com_velocity = np.zeros(3)
        angular_momentum = np.zeros(3)
        com_acceleration = np.zeros(3)
        change_of_angular_momentum = np.zeros(3)
        mass = rbdl.CalcCenterOfMass(self.model, q, qdot, qddot, com, com_velocity, com_acceleration,
                                     angular_momentum, change_of_angular_momentum, update_kinematics=self.uk)
        return mass, com, com_velocity, angular_momentum, com_acceleration, change_of_angular_momentum

    def calc_zero_moment_point(self, q, qdot, qddot, plane_normal=np.array([0, 1., 0])):
        r"""
        Computes the Zero-Moment-Point (ZMP) on a given contact surface that passes the origin (determined by normal).

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param qddot: Robot acceleration qddot in shape [dof].
        :param plane_normal: The ground normal in shape [3].
        :return: The zero moment point in shape [3].
        """
        zmp = np.zeros(3)
        rbdl.CalcZeroMomentPoint(self.model, q, qdot, qddot, zmp, plane_normal, np.zeros(3), update_kinematics=self.uk)
        return zmp

    def update_kinematics(self, q, qdot, qddot):
        r"""
        Update the kinematic states.

        :param q: Robot configuration q in shape [dof].
        :param qdot: Robot velocity qdot in shape [dof].
        :param qddot: Robot acceleration qddot in shape [dof].
        """
        rbdl.UpdateKinematics(self.model, q, qdot, qddot)
