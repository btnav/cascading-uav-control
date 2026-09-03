"""
Filename: controller.py
Author: btnav
Date: May 2026

Description:
    AERO60492 Coursework 3: Feedback Control

    A control algorithm for a quadcopter UAV. Implements four controllers:
    - 3x PID-PD cascading controllers for the X/Y/Z-axes.
        - PID outer loops for position control.
        - PD inner loops for velocity control.
    - 1x PI controller for yaw.

    Cascading control has been chosen as the advanced method, as it offers
    excellent disturbance rejection, so works well when wind is enabled. 
    While the outer loop acts as a regular PID, controlling the drone's 
    position, the inner loop can respond quickly to sudden changes in the
    drone's velocity (like gusts of wind).

    For the cascade to be effective, the inner loop must be faster than the
    outer loop [1], PD was selected as this controller is most suitable when
    a fast response is required [2]. Conversely, the yaw controller is a PI
    controller. Yaw dynamics are slow, so the derivative term isn't as 
    important. The integral term is neccessary to counterract steady-state
    errs. Initally, a PID controller was used for yaw, but the derivative
    term amplifed noise, and excluding it improved performance.

    Gains have been tuned by finding the ultimate gain and period for each
    axis, then applying Ziegler-Nichols and Pessen Integral tuning rules [3].
    These gains have then been manually adjusted to improve performance in 
    wind.

    References:
    [1] G. Ellis, `Chapter 3 - Tuning a Control System`, in Control System
    Design Guide (Fourth Edition), Fourth Edition., Boston: Butterworth-
    Heinemann, 2012, pp. 31-60. doi: 
    https://doi.org/10.1016/B978-0-12-385920-4.00003-5.
    [2] G. Ellis, 'Chapter 6 - Four Types of Controllers', in Control System
    Design Guide (Fourth Edition), Fourth Edition., Boston: Butterworth-
    Heinemann, 2012, pp. 97-119. doi: 
    https://doi.org/10.1016/B978-0-12-385920-4.00006-0.
    [3] A. S. McCormack and K. R. Godfrey, 'Rule-based autotuning based on
    frequency domain identification', IEEE Transactions on Control Systems
    Technology, vol. 6, no. 1, pp. 43-61, 1998, doi:
    https://doi.org/10.1109/87.654876.

License: BSD 3-Clause
Contact: 174347826+btnav@users.noreply.github.com
"""

import numpy as np

# Configuration

VEL_LIM = 8.0  # m/s (DJI Tello max speed is 8 m/s)
YAW_RATE_LIM = np.deg2rad(100)  # rad/s

# PID gains for X/Y outer loop
KP_XY = 12.220308
KI_XY = 0.269427
KD_XY = 3.080499
KI_XY_SAT = 0.05
# PD gains for X/Y inner loop
KP_XY_VEL = 3.878187
KD_XY_VEL = 3.141330

# PID gains for Z outer loop
KP_Z = 4.532092
KI_Z = 1.686047
KD_Z = 0.057137
KI_Z_SAT = 0.05
# PD gains for Z inner loop
KP_Z_VEL = 3.224464
KD_Z_VEL = 0.370813

# PI gains for yaw
KP_YAW = 2.543493
KI_YAW = 1.986478
KI_YAW_SAT = 5


class PID:
    """PID controller with anti-windup.

    Args:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        Ki_sat (float): Integral saturation limit.
    """

    def __init__(self, Kp, Ki, Kd, Ki_sat):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ki_sat = Ki_sat
        self.i_err = 0.0
        self.prev_err = 0.0

    def reset(self):
        """Reset errs to zero."""
        self.i_err = 0.0
        self.prev_err = 0.0

    def update(self, err, dt):
        """Update the controller.

        Args:
            err (float): Current err value.
            dt (float): Time step (seconds).
        Returns:
            float: Control setpoint.
        """
        # Proportional error
        p_err = err

        # Integral err
        self.i_err += err * dt
        # Anti windup clips Ki values that exceed Ki_sat
        if abs(self.i_err) > self.Ki_sat:
            self.i_err = (self.i_err / abs(self.i_err)) * self.Ki_sat
        self.i_err = np.clip(self.i_err, -self.Ki_sat, self.Ki_sat)

        # Derivative error
        d_err = (err - self.prev_err) / dt
        self.prev_err = err

        # PID output
        output = self.Kp * p_err + self.Ki * self.i_err + self.Kd * d_err
        return output


class PD(PID):
    """PD controller, Ki = 0.

    Args:
        Kp (float): Proportional gain.
        Kd (float): Derivative gain.
    """

    def __init__(self, Kp, Kd):
        super().__init__(Kp=Kp, Ki=0.0, Kd=Kd, Ki_sat=0.0)


class PI(PID):
    """PI controller, Kd = 0.

    Args:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Ki_sat (float): Integral saturation limit.
    """

    def __init__(self, Kp, Ki, Ki_sat):
        super().__init__(Kp=Kp, Ki=Ki, Kd=0.0, Ki_sat=Ki_sat)


class Cascade:
    """Cascading PID-PD controller.

    Args:
        outer (PID): Outer loop position controller.
        inner (PD): Inner loop velocity controller.
    """

    def __init__(self, outer: PID, inner: PD):
        self.outer = outer
        self.inner = inner
        self.prev_pos = 0.0
        self.prev_vel = 0.0

    def reset(self):
        """Reset controller state."""
        self.outer.reset()
        self.inner.reset()
        self.prev_pos = 0.0
        self.prev_vel = 0.0

    def update(self, current, target, dt):
        """Update the PID-PD controller.

        Args:
            current (float): Current position.
            target (float): Target position.
            dt (float): Time step (seconds).

        Returns:
            float: Cascade control setpoint.
        """
        dt = _safe_dt(dt)
        pos_err = target - current

        # Outer loop PID position control, returns a velocity setpoint
        vel_output_prime = self.outer.update(pos_err, dt)

        # Differentiate position to get measured velocity and compute error
        vel_measured = (current - self.prev_pos) / dt
        vel_err = vel_output_prime - vel_measured

        # Inner loop velocity PD control, returns an acceleration setpoint
        acc_output = self.inner.update(vel_err, dt)

        # Integrate acceleration to get the new velocity setpoint, and clamp
        vel_output = vel_output_prime + acc_output * dt
        vel_output = np.clip(vel_output, -VEL_LIM, VEL_LIM)

        # Update state for the next iteration
        self.prev_pos = current
        self.prev_vel = vel_output

        # Output velocity setpoint
        return vel_output


# Initialise controllers

x_control = Cascade(
    outer=PID(
        Kp=KP_XY,
        Ki=KI_XY,
        Kd=KD_XY,
        Ki_sat=KI_XY_SAT,
    ),
    inner=PD(
        Kp=KP_XY_VEL,
        Kd=KD_XY_VEL,
    ),
)

y_control = Cascade(
    outer=PID(
        Kp=KP_XY,
        Ki=KI_XY,
        Kd=KD_XY,
        Ki_sat=KI_XY_SAT,
    ),
    inner=PD(
        Kp=KP_XY_VEL,
        Kd=KD_XY_VEL,
    ),
)

z_control = Cascade(
    outer=PID(
        Kp=KP_Z,
        Ki=KI_Z,
        Kd=KD_Z,
        Ki_sat=KI_Z_SAT,
    ),
    inner=PD(
        Kp=KP_Z_VEL,
        Kd=KD_Z_VEL,
    ),
)

yaw_control = PI(
    Kp=KP_YAW,
    Ki=KI_YAW,
    Ki_sat=KI_YAW_SAT,
)


def controller(state, target_pos, dt, wind_enabled=False):
    """Entry point for the controller.

    Args:
        state (tuple): Current drone state in the format
            (position_x (m), position_y (m), position_z (m),
             roll (rad), pitch (rad), yaw (rad)).
        target_pos (tuple): Target position in the format
            (x (m), y (m), z (m), yaw (radians)).
        dt (float): Time step (s).
        wind_enabled (bool, optional): Flag to enable wind disturbance,
            defaults to False.

    Returns:
        tuple: Velocity commands in the format
            (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s),
             velocity_z_setpoint (m/s), yaw_rate_setpoint (rad/s)).
    """
    # Create an attribute to track the previous target
    if not hasattr(controller, "_prev_target"):
        controller._prev_target = None

    # If the target has changed, reset errors and setpoints
    if controller._prev_target is None or not np.allclose(
        target_pos, controller._prev_target
    ):
        x_control.reset()
        y_control.reset()
        z_control.reset()
        yaw_control.reset()
        controller._prev_target = target_pos
        return (0.0, 0.0, 0.0, 0.0)

    # Extract values from state and target_pos
    x_curr, y_curr, z_curr, roll_curr, pitch_curr, yaw_curr = state
    x_target, y_target, z_target, yaw_target = target_pos

    # Cascade PID-PD control for X/Y/Z axes
    x_output = x_control.update(x_curr, x_target, dt)
    y_output = y_control.update(y_curr, y_target, dt)
    z_output = z_control.update(z_curr, z_target, dt)
    # PI control for yaw, normalising yaw error to [-π, π]
    yaw_err = _normalise_angle(yaw_target - yaw_curr)
    yaw_output = yaw_control.update(yaw_err, dt)

    # Convert resulting velocities from the world frame to the drone's frame
    vel_world = [x_output, y_output, z_output]
    x_vel, y_vel, z_vel = _world_to_body(vel_world, yaw_curr)
    # Clamp yaw rate to limits
    yaw_rate = np.clip(yaw_output, -YAW_RATE_LIM, YAW_RATE_LIM)

    # Return the transformed and clamped setpoints
    setpoints = (x_vel, y_vel, z_vel, yaw_rate)
    return setpoints


# Helper functions


def _safe_dt(timestep, fallback=0.1):
    """Sanitise a timestep value.
    Fallback to 0.1 s (the dt used in the lab experiments)"""
    try:
        dt = float(timestep)
    except Exception:
        return fallback
    # Ignore Unix timestamps (this was a problem in the first lab)
    if dt > 1e6:
        return fallback
    # Ignore non-finite or non-positive timesteps (just in case)
    if not np.isfinite(dt) or dt <= 0.0:
        return fallback
    return dt


def _normalise_angle(angle):
    """Normalise angle to [-π, π] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _world_to_body(vel_world, yaw):
    """Convert velocity from world frame to body frame."""
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x_vel_body = vel_world[0] * cos_yaw + vel_world[1] * sin_yaw
    y_vel_body = -vel_world[0] * sin_yaw + vel_world[1] * cos_yaw
    z_vel_body = vel_world[2]
    return (x_vel_body, y_vel_body, z_vel_body)
