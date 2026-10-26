"""Full nonlinear 6-DOF rigid-body aircraft model.

The model uses standard NED navigation axes (x forward, y right, z down) and
body axes with z down. It includes translational and rotational dynamics,
quaternion attitude propagation, stability-derivative aerodynamics, engine
thrust, actuator limits, and envelope measurements.

It is deliberately generic and educational: no real aircraft data, weapon
system, classified performance, or flight-certified control law is represented.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dynamics import G, air_density, clamp


Vector3 = np.ndarray
Quaternion = np.ndarray


def quat_normalize(q: Quaternion) -> Quaternion:
    arr = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(arr))
    if arr.shape != (4,) or norm <= 0:
        raise ValueError("quaternion must contain four non-zero values")
    return arr / norm


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_to_matrix(q: Quaternion) -> np.ndarray:
    """Rotation matrix mapping body vectors to NED navigation axes."""

    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Aerospace roll/pitch/yaw to body-to-NED quaternion."""

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


def quat_to_euler(q: Quaternion) -> tuple[float, float, float]:
    r = quat_to_matrix(q)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.asin(clamp(-r[2, 0], -1.0, 1.0))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


@dataclass(frozen=True)
class Aircraft6DOFParams:
    mass: float = 12_000.0
    inertia: tuple[float, float, float] = (18_000.0, 92_000.0, 98_000.0)
    wing_area: float = 28.0
    wing_span: float = 11.0
    mean_chord: float = 3.2
    max_thrust: float = 145_000.0
    cl0: float = 0.20
    cl_alpha: float = 5.2
    cl_q: float = 4.0
    cl_de: float = 0.35
    cd0: float = 0.035
    cd_alpha2: float = 0.75
    cd_beta2: float = 0.25
    cy_beta: float = -0.55
    cy_dr: float = 0.12
    cl_beta: float = -0.08
    cl_p: float = -0.35
    cl_r: float = 0.08
    cl_da: float = 0.16
    cl_dr: float = 0.015
    cm0: float = 0.0
    cm_alpha: float = -0.90
    cm_q: float = -8.0
    cm_de: float = -1.20
    cn_beta: float = 0.12
    cn_p: float = -0.03
    cn_r: float = -0.18
    cn_da: float = -0.015
    cn_dr: float = -0.09
    alpha_max: float = math.radians(13.0)
    beta_max: float = math.radians(10.0)
    # Post-stall extension (teaching approximation, off by default).
    post_stall: bool = False
    stall_alpha: float = math.radians(16.0)
    stall_blend: float = math.radians(10.0)
    post_stall_cm_factor: float = 0.30
    max_load_factor: float = 7.0
    min_load_factor: float = -2.5
    max_bank: float = math.radians(80.0)
    max_surface: float = math.radians(25.0)
    max_rudder: float = math.radians(20.0)
    surface_rate: float = math.radians(90.0)
    throttle_rate: float = 1.2

    def validate(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if min(self.inertia) <= 0:
            raise ValueError("all inertia terms must be positive")
        if min(self.wing_area, self.wing_span, self.mean_chord) <= 0:
            raise ValueError("aircraft geometry must be positive")
        if self.max_thrust <= 0:
            raise ValueError("max_thrust must be positive")
        if self.cl_alpha <= 0 or self.cd0 < 0 or self.cd_alpha2 < 0:
            raise ValueError("invalid aerodynamic lift/drag coefficients")
        if self.cm_de >= 0:
            raise ValueError("elevator effectiveness must be negative in this convention")
        if self.cl_da <= 0 or self.cn_dr >= 0:
            raise ValueError("invalid aileron/rudder signs")
        if not 0 < self.alpha_max < math.radians(30):
            raise ValueError("alpha_max outside teaching range")
        if self.post_stall:
            if not self.alpha_max < self.stall_alpha < math.radians(45):
                raise ValueError("stall_alpha must sit between alpha_max and 45 degrees")
            if self.stall_blend <= 0:
                raise ValueError("stall_blend must be positive")
            if not 0 < self.post_stall_cm_factor <= 1.0:
                raise ValueError("post_stall_cm_factor must be in (0, 1]")
        if self.min_load_factor >= self.max_load_factor:
            raise ValueError("load-factor limits are inconsistent")
        if min(self.max_surface, self.max_rudder, self.surface_rate, self.throttle_rate) <= 0:
            raise ValueError("actuator limits must be positive")


@dataclass(frozen=True)
class RigidBodyState:
    position: Vector3  # NED metres
    velocity_body: Vector3  # u, v, w m/s
    quaternion: Quaternion  # body to NED, [w, x, y, z]
    rates_body: Vector3  # p, q, r rad/s

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=float))
        object.__setattr__(self, "velocity_body", np.asarray(self.velocity_body, dtype=float))
        object.__setattr__(self, "quaternion", quat_normalize(self.quaternion))
        object.__setattr__(self, "rates_body", np.asarray(self.rates_body, dtype=float))
        if self.position.shape != (3,) or self.velocity_body.shape != (3,) or self.rates_body.shape != (3,):
            raise ValueError("position, velocity and rates must be three-component vectors")

    @property
    def altitude(self) -> float:
        return -float(self.position[2])

    @property
    def airspeed(self) -> float:
        return float(np.linalg.norm(self.velocity_body))

    @property
    def alpha(self) -> float:
        # True aerodynamic angle of attack: exceeds +/-90 deg when the
        # longitudinal flow reverses (post-stall demonstrations like the cobra).
        u, _, w = self.velocity_body
        if abs(float(u)) < 1e-9 and abs(float(w)) < 1e-9:
            return 0.0
        return math.atan2(float(w), float(u))

    @property
    def beta(self) -> float:
        speed = max(self.airspeed, 1.0)
        return math.asin(clamp(float(self.velocity_body[1]) / speed, -1.0, 1.0))

    @property
    def euler(self) -> tuple[float, float, float]:
        return quat_to_euler(self.quaternion)

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            [self.position, self.velocity_body, self.quaternion, self.rates_body]
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "RigidBodyState":
        arr = np.asarray(values, dtype=float)
        if arr.shape != (13,):
            raise ValueError("rigid-body state must have 13 values")
        return cls(arr[0:3], arr[3:6], arr[6:10], arr[10:13])


@dataclass(frozen=True)
class SurfaceCommand:
    throttle: float
    elevator: float
    aileron: float
    rudder: float

    def clamped(self, params: Aircraft6DOFParams) -> "SurfaceCommand":
        return SurfaceCommand(
            clamp(self.throttle, 0.0, 1.0),
            clamp(self.elevator, -params.max_surface, params.max_surface),
            clamp(self.aileron, -params.max_surface, params.max_surface),
            clamp(self.rudder, -params.max_rudder, params.max_rudder),
        )


@dataclass(frozen=True)
class AeroSnapshot:
    dynamic_pressure: float
    lift: float
    drag: float
    side_force: float
    coefficients: tuple[float, float, float, float, float, float]
    load_factor: float


@dataclass(frozen=True)
class RigidBodyDerivatives:
    position_dot: Vector3
    velocity_dot: Vector3
    quaternion_dot: Quaternion
    rates_dot: Vector3
    aero: AeroSnapshot

    def as_array(self) -> np.ndarray:
        return np.concatenate(
            [self.position_dot, self.velocity_dot, self.quaternion_dot, self.rates_dot]
        )


def aerodynamics(
    state: RigidBodyState,
    surfaces: SurfaceCommand,
    params: Aircraft6DOFParams,
    wind_body: Vector3 | None = None,
) -> tuple[Vector3, Vector3, AeroSnapshot]:
    """Return body-axis aerodynamic force, moment and diagnostic data.

    ``wind_body`` is the wind velocity expressed in body axes. The aircraft's
    relative airflow is its inertial body velocity minus that wind, which lets
    scenarios add carrier burble or other local disturbances without changing
    the rigid-body equations.
    """

    params.validate()
    s = surfaces.clamped(params)
    wind = np.zeros(3) if wind_body is None else np.asarray(wind_body, dtype=float)
    if wind.shape != (3,):
        raise ValueError("wind_body must be a three-component vector")
    relative_flow = state.velocity_body - wind
    speed = max(float(np.linalg.norm(relative_flow)), 1.0)
    alpha = math.atan2(float(relative_flow[2]), float(relative_flow[0]))
    beta = math.asin(clamp(float(relative_flow[1]) / speed, -1.0, 1.0))
    p, q, r = (float(v) for v in state.rates_body)
    qbar = 0.5 * air_density(state.altitude) * speed * speed
    area = params.wing_area
    span_rate = params.wing_span / (2.0 * speed)
    chord_rate = params.mean_chord / (2.0 * speed)

    cl = params.cl0 + params.cl_alpha * alpha + params.cl_q * q * chord_rate + params.cl_de * s.elevator
    cd = params.cd0 + params.cd_alpha2 * alpha * alpha + params.cd_beta2 * beta * beta
    cy = params.cy_beta * beta + params.cy_dr * s.rudder
    cm_alpha_effective = params.cm_alpha
    if params.post_stall and alpha > params.stall_alpha:
        # Teaching flat-plate continuation beyond stall: lift follows a
        # sin(2a) decay matched at stall, drag grows toward a flat-plate
        # maximum, and static pitch stability partially breaks down.
        # This is NOT a validated high-alpha model; it exists so scripted
        # post-stall demonstrations (e.g. the cobra) stay plausible.
        w = clamp((alpha - params.stall_alpha) / params.stall_blend, 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)  # smoothstep
        a_s = params.stall_alpha
        cl_at_stall = params.cl0 + params.cl_alpha * a_s
        cd_at_stall = params.cd0 + params.cd_alpha2 * a_s * a_s
        cl_plate = cl_at_stall * math.sin(2.0 * alpha) / math.sin(2.0 * a_s)
        cd_flat_max = 1.8
        ratio = (math.sin(alpha) - math.sin(a_s)) / max(1e-6, 1.0 - math.sin(a_s))
        cd_plate = cd_at_stall + (cd_flat_max - cd_at_stall) * ratio * ratio
        cl = (1.0 - w) * cl + w * (cl_plate + params.cl_q * q * chord_rate + params.cl_de * s.elevator)
        cd = (1.0 - w) * cd + w * (cd_plate + params.cd_beta2 * beta * beta)
        cm_alpha_effective = params.cm_alpha * ((1.0 - w) + w * params.post_stall_cm_factor)
    c_roll = (
        params.cl_beta * beta
        + params.cl_p * p * span_rate
        + params.cl_r * r * span_rate
        + params.cl_da * s.aileron
        + params.cl_dr * s.rudder
    )
    c_pitch = (
        params.cm0
        + cm_alpha_effective * alpha
        + params.cm_q * q * chord_rate
        + params.cm_de * s.elevator
    )
    c_yaw = (
        params.cn_beta * beta
        + params.cn_p * p * span_rate
        + params.cn_r * r * span_rate
        + params.cn_da * s.aileron
        + params.cn_dr * s.rudder
    )

    lift = qbar * area * cl
    drag = qbar * area * cd
    side = qbar * area * cy
    force = np.array(
        [
            lift * math.sin(alpha) - drag * math.cos(alpha),
            side,
            -lift * math.cos(alpha) - drag * math.sin(alpha),
        ],
        dtype=float,
    )
    moment = qbar * area * np.array(
        [params.wing_span * c_roll, params.mean_chord * c_pitch, params.wing_span * c_yaw],
        dtype=float,
    )
    load_factor = -force[2] / (params.mass * G)
    snapshot = AeroSnapshot(
        qbar, lift, drag, side, (cl, cd, cy, c_roll, c_pitch, c_yaw), load_factor
    )
    return force, moment, snapshot


def rigid_body_derivatives(
    state: RigidBodyState,
    surfaces: SurfaceCommand,
    params: Aircraft6DOFParams,
    wind_body: Vector3 | None = None,
) -> RigidBodyDerivatives:
    """Full nonlinear 6-DOF equations of motion."""

    params.validate()
    if state.airspeed < 1.0:
        raise ValueError("airspeed must be positive in flight")
    s = surfaces.clamped(params)
    rotation = quat_to_matrix(state.quaternion)
    aero_force, aero_moment, snapshot = aerodynamics(state, s, params, wind_body)
    thrust = np.array([s.throttle * params.max_thrust, 0.0, 0.0], dtype=float)
    gravity_body = rotation.T @ np.array([0.0, 0.0, params.mass * G], dtype=float)
    omega = state.rates_body

    velocity_dot = (
        (aero_force + thrust + gravity_body) / params.mass
        - np.cross(omega, state.velocity_body)
    )
    inertia = np.asarray(params.inertia, dtype=float)
    rates_dot = (aero_moment - np.cross(omega, inertia * omega)) / inertia
    quaternion_dot = 0.5 * quat_multiply(
        state.quaternion,
        np.array([0.0, omega[0], omega[1], omega[2]], dtype=float),
    )
    position_dot = rotation @ state.velocity_body
    return RigidBodyDerivatives(
        position_dot, velocity_dot, quaternion_dot, rates_dot, snapshot
    )


def rk4_step_6dof(
    state: RigidBodyState,
    surfaces: SurfaceCommand,
    params: Aircraft6DOFParams,
    dt: float,
    wind_body: Vector3 | None = None,
) -> RigidBodyState:
    if dt <= 0:
        raise ValueError("dt must be positive")
    y0 = state.as_array()
    k1 = rigid_body_derivatives(state, surfaces, params, wind_body).as_array()
    k2 = rigid_body_derivatives(RigidBodyState.from_array(y0 + 0.5 * dt * k1), surfaces, params, wind_body).as_array()
    k3 = rigid_body_derivatives(RigidBodyState.from_array(y0 + 0.5 * dt * k2), surfaces, params, wind_body).as_array()
    k4 = rigid_body_derivatives(RigidBodyState.from_array(y0 + dt * k3), surfaces, params, wind_body).as_array()
    return RigidBodyState.from_array(y0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)


def trim_level_flight(
    params: Aircraft6DOFParams | None = None,
    speed: float = 110.0,
    altitude: float = 1_500.0,
    heading: float = 0.0,
) -> tuple[RigidBodyState, SurfaceCommand]:
    """Compute a practical level-flight trim for the generic model."""

    p = params or Aircraft6DOFParams()
    p.validate()
    qbar = 0.5 * air_density(altitude) * speed * speed
    alpha = math.radians(4.0)
    elevator = 0.0
    throttle = 0.1
    for _ in range(8):
        elevator = -(p.cm0 + p.cm_alpha * alpha) / p.cm_de
        cd = p.cd0 + p.cd_alpha2 * alpha * alpha
        drag = qbar * p.wing_area * cd
        thrust = drag / max(0.2, math.cos(alpha))
        throttle = clamp(thrust / p.max_thrust, 0.0, 1.0)
        lift_required = p.mass * G - thrust * math.sin(alpha)
        cl_required = lift_required / (qbar * p.wing_area)
        alpha = (cl_required - p.cl0 - p.cl_de * elevator) / p.cl_alpha
        alpha = clamp(alpha, math.radians(-2.0), p.alpha_max * 0.9)
    elevator = -(p.cm0 + p.cm_alpha * alpha) / p.cm_de
    cd = p.cd0 + p.cd_alpha2 * alpha * alpha
    drag = qbar * p.wing_area * cd
    throttle = clamp(drag / (p.max_thrust * max(0.2, math.cos(alpha))), 0.0, 1.0)
    state = RigidBodyState(
        position=np.array([0.0, 0.0, -altitude]),
        velocity_body=np.array([speed * math.cos(alpha), 0.0, speed * math.sin(alpha)]),
        quaternion=euler_to_quat(0.0, alpha, heading),
        rates_body=np.zeros(3),
    )
    return state, SurfaceCommand(throttle, elevator, 0.0, 0.0).clamped(p)
