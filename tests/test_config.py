"""Scenario files, and the errors they should produce when wrong.

Most of this file is about failure rather than success. A config format's value
is almost entirely in what it does with a mistake: a loader that accepts
``anglesigma = 0.002`` and quietly runs with the default has cost its user more
than one that never existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from interceptor.config import (
    ConfigError,
    bundled_names,
    load,
    load_bundled,
    loads,
)
from interceptor.guidance.pronav import ProportionalNavigation
from interceptor.guidance.pursuit import PurePursuit

MINIMAL = """
name = "test"

[missile]
position = [0.0, 0.0, 1000.0]

[target]
position = [0.0, 5000.0, 1000.0]
velocity = [200.0, 0.0, 0.0]
"""


# --------------------------------------------------------------------------
# The bundled scenarios
# --------------------------------------------------------------------------
def test_the_package_ships_scenarios() -> None:
    names = bundled_names()
    assert "crossing" in names
    assert len(names) >= 3


@pytest.mark.parametrize("name", bundled_names())
def test_every_bundled_scenario_loads_and_builds(name: str) -> None:
    """Each shipped file must survive the whole path from text to a world.

    Parametrised over the directory listing rather than a hard-coded list, so
    a new scenario file is covered the moment it is added and a broken one
    cannot be shipped quietly.
    """
    spec = load_bundled(name)
    assert spec.name
    assert spec.description
    world, detector = spec.build(seed=0)
    assert world is not None
    assert detector is not None


def test_an_unknown_bundled_name_suggests_the_closest() -> None:
    with pytest.raises(ConfigError, match="did you mean 'crossing'"):
        load_bundled("crssing")


def test_the_perfect_information_preset_has_no_seeker() -> None:
    """The Phase 3 baseline, reachable without editing Python."""
    assert load_bundled("perfect-information").seeker is None


# --------------------------------------------------------------------------
# Defaults and shorthand
# --------------------------------------------------------------------------
def test_a_minimal_file_fills_in_sensible_defaults() -> None:
    spec = loads(MINIMAL)
    assert isinstance(spec.law, ProportionalNavigation)
    assert spec.seeker is not None
    assert spec.estimator is not None
    assert spec.scenario.missile_mass == pytest.approx(85.0)


def test_launch_speed_aims_the_missile_at_the_target() -> None:
    """The shorthand that keeps the two positions the only source of geometry."""
    spec = loads(MINIMAL.replace("[missile]\n", "[missile]\nspeed = 100.0\n"))
    velocity = spec.scenario.missile_velocity
    assert float(np.linalg.norm(velocity)) == pytest.approx(100.0)
    direction = spec.scenario.target_position - spec.scenario.missile_position
    cosine = float(
        np.dot(velocity, direction) / (np.linalg.norm(velocity) * np.linalg.norm(direction))
    )
    assert cosine == pytest.approx(1.0)


def test_an_explicit_velocity_overrides_the_shorthand() -> None:
    spec = loads(MINIMAL.replace("[missile]\n", "[missile]\nvelocity = [10.0, 20.0, 30.0]\n"))
    assert spec.scenario.missile_velocity == pytest.approx([10.0, 20.0, 30.0])


def test_the_gimbal_limit_is_written_in_degrees() -> None:
    """Radians in the code, degrees in the file — nobody writes 0.698."""
    spec = loads(MINIMAL + "\n[seeker]\ngimbal_limit_deg = 30.0\n")
    assert spec.seeker is not None
    assert spec.seeker.gimbal_limit == pytest.approx(np.deg2rad(30.0))


@pytest.mark.parametrize(
    ("law", "expected"),
    [("pronav", ProportionalNavigation), ("pursuit", PurePursuit)],
)
def test_guidance_laws_can_be_selected(law: str, expected: type) -> None:
    spec = loads(MINIMAL + f'\n[guidance]\nlaw = "{law}"\n')
    assert isinstance(spec.law, expected)


def test_guidance_can_be_switched_off_entirely() -> None:
    """An unguided round — how the tests prove an intercept came from guidance."""
    assert loads(MINIMAL + '\n[guidance]\nlaw = "none"\n').law is None


@pytest.mark.parametrize("kind", ["none", "alpha_beta", "ekf"])
def test_estimators_can_be_selected(kind: str) -> None:
    spec = loads(MINIMAL + f'\n[estimator]\nkind = "{kind}"\n')
    if kind == "none":
        assert spec.estimator is None
    else:
        assert spec.estimator is not None


def test_the_estimator_is_a_factory_not_an_instance() -> None:
    """A filter carries state, so every run needs its own.

    Sharing one across a sweep would let the second run start already convinced
    of where a different target was.
    """
    spec = loads(MINIMAL + '\n[estimator]\nkind = "ekf"\n')
    assert spec.estimator is not None
    assert spec.estimator() is not spec.estimator()


@pytest.mark.parametrize(
    "manoeuvre",
    [
        '[target.manoeuvre]\nkind = "straight"',
        '[target.manoeuvre]\nkind = "weave"\namplitude_g = 5.0\nperiod = 3.0',
        '[target.manoeuvre]\nkind = "break_turn"\namplitude_g = 8.0\nstart_time = 4.0',
    ],
)
def test_manoeuvres_can_be_selected(manoeuvre: str) -> None:
    spec = loads(MINIMAL + "\n" + manoeuvre + "\n")
    assert callable(spec.scenario.manoeuvre)


# --------------------------------------------------------------------------
# Rejection
# --------------------------------------------------------------------------
def test_a_misspelled_key_is_an_error_with_a_suggestion() -> None:
    """The single most valuable behaviour in this module.

    Silently ignoring an unrecognised key is how someone spends an afternoon
    wondering why their noise setting does nothing.
    """
    with pytest.raises(
        ConfigError, match=r"seeker\.glint: unknown key — did you mean 'glint_sigma'"
    ):
        loads(MINIMAL + "\n[seeker]\nglint = 1.5\n")


def test_an_unrecognisable_key_lists_the_alternatives() -> None:
    with pytest.raises(ConfigError, match="expected one of"):
        loads(MINIMAL + "\n[seeker]\nzzzzzz = 1.5\n")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[guidance]\nlaw = "homing"', "must be one of"),
        ('[estimator]\nkind = "particle"', "must be one of"),
        ('[target.manoeuvre]\nkind = "barrel_roll"', "must be one of"),
    ],
)
def test_an_unknown_choice_names_the_valid_ones(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        loads(MINIMAL + "\n" + text + "\n")


def test_a_negative_navigation_constant_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be at least"):
        loads(MINIMAL + "\n[guidance]\nnavigation_constant = -1.0\n")


def test_a_string_where_a_number_belongs_is_rejected() -> None:
    with pytest.raises(ConfigError, match="expected a number, got str"):
        loads(MINIMAL + '\n[seeker]\nglint_sigma = "large"\n')


def test_a_boolean_is_not_accepted_as_a_number() -> None:
    """``True`` is an ``int`` in Python, and that must not leak into the schema."""
    with pytest.raises(ConfigError, match="expected a number, got bool"):
        loads(MINIMAL + "\n[seeker]\nglint_sigma = true\n")


def test_a_vector_must_have_three_components() -> None:
    with pytest.raises(ConfigError, match=r"expected three numbers"):
        loads(MINIMAL.replace("position = [0.0, 5000.0, 1000.0]", "position = [0.0, 5000.0]"))


def test_a_missing_required_vector_is_reported_by_name() -> None:
    with pytest.raises(ConfigError, match=r"target\.velocity: is required"):
        loads(MINIMAL.replace("velocity = [200.0, 0.0, 0.0]", ""))


def test_a_target_on_top_of_the_missile_is_rejected() -> None:
    """Aim-at-target shorthand has nothing to aim at."""
    text = MINIMAL.replace("position = [0.0, 5000.0, 1000.0]", "position = [0.0, 0.0, 1000.0]")
    with pytest.raises(ConfigError, match="nothing to aim at"):
        loads(text)


def test_malformed_toml_is_reported_as_such() -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        loads("this is not = = toml")


def test_a_missing_file_is_an_error_not_a_traceback() -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load("/nonexistent/scenario.toml")


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------
def test_a_file_on_disk_loads_the_same_as_its_text(tmp_path: object) -> None:
    path = tmp_path / "scenario.toml"  # type: ignore[operator]
    path.write_text(MINIMAL, encoding="utf-8")
    from_disk = load(path)
    from_text = loads(MINIMAL)
    assert from_disk.scenario.target_position == pytest.approx(from_text.scenario.target_position)
