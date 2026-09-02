"""Atmosphere, forces, conservation and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.airframe.aero import Aerodynamics
from interceptor.airframe.propulsion import Motor
from interceptor.core.state import STATE_SIZE, EntityState
from interceptor.core.world import (
    SEA_LEVEL_DENSITY,
    STANDARD_GRAVITY,
    World,
    WorldConfig,
    air_density,
)
from interceptor.entities.missile import Missile
from interceptor.sim.engagement import run


# --------------------------------------------------------------------------
# Atmosphere
# --------------------------------------------------------------------------
def test_density_at_sea_level() -> None:
    assert air_density(0.0) == pytest.approx(SEA_LEVEL_DENSITY)


def test_density_falls_monotonically_with_altitude() -> None:
    altitudes = np.linspace(0.0, 20000.0, 200)
    densities = np.array([air_density(float(h)) for h in altitudes])
    assert np.all(np.diff(densities) < 0.0)


def test_density_is_clamped_below_sea_level() -> None:
    assert air_density(-500.0) == pytest.approx(SEA_LEVEL_DENSITY)


def test_density_halves_at_roughly_six_kilometres() -> None:
    """A sanity anchor against the real atmosphere: rho halves near 5.9 km."""
    assert air_density(5900.0) / SEA_LEVEL_DENSITY == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
def test_state_packs_and_unpacks_losslessly() -> None:
    state = EntityState(
        pos=np.array([1.5, -2.5, 300.0]), vel=np.array([10.0, 0.0, -3.25]), mass=85.0
    )
    packed = state.to_array()
    assert packed.shape == (STATE_SIZE,)

    recovered = EntityState.from_array(packed)
    assert np.array_equal(recovered.pos, state.pos)
    assert np.array_equal(recovered.vel, state.vel)
    assert recovered.mass == state.mass


def test_state_rejects_non_positive_mass() -> None:
    with pytest.raises(ValueError, match="mass must be positive"):
        EntityState(pos=np.zeros(3), vel=np.zeros(3), mass=0.0)


def test_state_rejects_the_wrong_number_of_components() -> None:
    with pytest.raises(ValueError, match="cannot reshape"):
        EntityState(pos=np.zeros(2), vel=np.zeros(3))


# --------------------------------------------------------------------------
# Aerodynamics
# --------------------------------------------------------------------------
def test_drag_opposes_the_velocity() -> None:
    aero = Aerodynamics()
    velocity = np.array([120.0, -60.0, 25.0])
    drag = aero.drag_acceleration(velocity, mass=85.0, density=1.0)
    assert float(np.dot(drag, velocity)) < 0.0


def test_drag_scales_with_the_square_of_speed() -> None:
    aero = Aerodynamics()
    slow = aero.drag_acceleration(np.array([100.0, 0.0, 0.0]), 85.0, 1.0)
    fast = aero.drag_acceleration(np.array([200.0, 0.0, 0.0]), 85.0, 1.0)
    ratio = float(np.linalg.norm(fast) / np.linalg.norm(slow))
    assert ratio == pytest.approx(4.0)


def test_a_body_at_rest_has_no_drag() -> None:
    aero = Aerodynamics()
    assert np.allclose(aero.drag_acceleration(np.zeros(3), 85.0, 1.225), 0.0)


def test_available_g_is_capped_by_the_structural_limit() -> None:
    """Fast and low, the airframe rather than the air is the binding constraint."""
    aero = Aerodynamics(max_lateral_g=30.0)
    available = aero.available_lateral_acceleration(speed=1000.0, mass=85.0, density=1.225)
    assert available == pytest.approx(30.0 * STANDARD_GRAVITY)


def test_available_g_is_capped_by_the_air_when_slow() -> None:
    """Slow and high, the air cannot supply the rated g however strong the airframe."""
    aero = Aerodynamics(max_lateral_g=30.0)
    available = aero.available_lateral_acceleration(speed=250.0, mass=85.0, density=0.30)
    assert available < 30.0 * STANDARD_GRAVITY
    assert available == pytest.approx(2.5 * 0.5 * 0.30 * 250.0**2 * 0.02 / 85.0)


def test_available_g_collapses_as_the_missile_slows() -> None:
    """The reason a late break-turn is hard to answer at the end of a flight."""
    aero = Aerodynamics()
    fast = aero.available_lateral_acceleration(600.0, 85.0, 0.4)
    slow = aero.available_lateral_acceleration(150.0, 85.0, 0.4)
    assert slow < fast


# --------------------------------------------------------------------------
# Propulsion
# --------------------------------------------------------------------------
def test_motor_thrust_schedule() -> None:
    motor = Motor(
        boost_thrust=20000.0, boost_duration=2.5, sustain_thrust=3000.0, sustain_duration=8.0
    )
    assert motor.thrust(-1.0) == 0.0
    assert motor.thrust(1.0) == 20000.0
    assert motor.thrust(3.0) == 3000.0
    assert motor.thrust(11.0) == 0.0
    assert motor.burn_time == pytest.approx(10.5)


def test_motor_burns_propellant_only_while_thrusting() -> None:
    motor = Motor(boost_thrust=20000.0, boost_duration=2.5)
    assert motor.mass_flow(1.0) < 0.0
    assert motor.mass_flow(5.0) == 0.0


def test_a_burning_missile_loses_its_propellant_mass() -> None:
    """Integrated mass flow must match the analytic propellant load.

    Not to machine precision, though, and the reason is worth knowing. Burnout
    is a step discontinuity in the thrust curve, and RK4 evaluates the
    derivative at four points across each step — so on the one step that
    straddles burnout, some of those points see thrust and some do not. The
    resulting error is of order ``dt / burn_time``, about 4e-4 here, and it is
    the honest price of modelling a solid motor as switching off instantly.
    """
    motor = Motor(boost_thrust=20000.0, boost_duration=2.5, specific_impulse=240.0)
    initial = EntityState(pos=np.zeros(3), vel=np.array([0.0, 0.0, 60.0]), mass=85.0)
    world = World(WorldConfig(enable_drag=False))
    world.add(Missile("missile", initial, motor=motor))

    result = run(world, duration=4.0, dt=1e-3, record_hz=1000.0)
    burned = initial.mass - float(result.recorder.mass("missile")[-1])
    assert burned == pytest.approx(motor.propellant_mass, rel=1e-3)


def test_an_inert_motor_produces_nothing() -> None:
    motor = Motor()
    assert motor.thrust(1.0) == 0.0
    assert motor.mass_flow(1.0) == 0.0
    assert motor.propellant_mass == 0.0


# --------------------------------------------------------------------------
# Conservation and determinism
# --------------------------------------------------------------------------
def test_mechanical_energy_is_conserved_without_drag() -> None:
    """Gravity is conservative, so 0.5 m v^2 + m g h must not drift.

    A slow drift here is the classic symptom of an integrator or a timestep
    problem, and it would be invisible in the trajectory plot.
    """
    initial = EntityState(
        pos=np.array([0.0, 0.0, 2000.0]), vel=np.array([180.0, 0.0, 120.0]), mass=85.0
    )
    world = World(WorldConfig(enable_drag=False))
    world.add(Missile("missile", initial))

    result = run(world, duration=20.0, dt=1e-3, record_hz=1000.0)
    speed = result.recorder.speed("missile")
    height = result.recorder.position("missile")[:, 2]
    energy = 0.5 * initial.mass * speed**2 + initial.mass * STANDARD_GRAVITY * height

    drift = float(np.abs(energy - energy[0]).max() / energy[0])
    assert drift < 1e-9, f"energy drifted by {drift:.3e} relative"


def test_drag_removes_energy() -> None:
    initial = EntityState(
        pos=np.array([0.0, 0.0, 2000.0]), vel=np.array([600.0, 0.0, 0.0]), mass=85.0
    )
    world = World(WorldConfig(enable_drag=True))
    world.add(Missile("missile", initial))

    result = run(world, duration=5.0, dt=1e-3)
    speed = result.recorder.speed("missile")
    assert speed[-1] < speed[0]
    assert np.all(np.diff(speed[: len(speed) // 2]) < 0.0)


def test_two_identical_runs_agree_bit_for_bit() -> None:
    """Determinism is what makes every other assertion in this suite meaningful."""

    def once() -> np.ndarray:
        world = World()
        world.add(
            Missile(
                "missile",
                EntityState(pos=np.zeros(3), vel=np.array([200.0, 50.0, 210.0]), mass=85.0),
                motor=Motor(boost_thrust=15000.0, boost_duration=2.0),
            )
        )
        return run(world, duration=8.0, dt=1e-3).recorder.position("missile")

    assert np.array_equal(once(), once())


def test_a_world_rejects_duplicate_names() -> None:
    world = World()
    state = EntityState(pos=np.zeros(3), vel=np.array([0.0, 0.0, 100.0]))
    world.add(Missile("missile", state))
    with pytest.raises(ValueError, match="already in this world"):
        world.add(Missile("missile", state))
