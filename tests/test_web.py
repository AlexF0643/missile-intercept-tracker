"""The browser app, minus the browser.

Everything here runs on a build server: the field table, the payload shape, and
the routes over a real socket on a real ephemeral port. What genuinely needs a
browser — that the page draws anything — lives in ``test_browser.py`` behind the
``gui`` marker, the same way the VPython viewer does.

The tests that matter most are the two drift checks. The form and the validator
are two descriptions of the same schema, and the failure mode of two such
descriptions is not that one breaks but that they quietly disagree: a control
that writes a key nothing reads, or a default that says one thing while an
omitted key means another. Neither shows up as an error anywhere. Both show up
here.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from interceptor.config import ConfigError, bundled_text, loads, resolve
from interceptor.sim.engagement import run
from interceptor.web.app import MAX_DURATION, FlightRequest, address_of, fly, make_server
from interceptor.web.fields import FIELDS, GROUPS, OMITTED, as_json, default_scenario

MINIMAL = """
[missile]
position = [0.0, 0.0, 1000.0]
[target]
position = [0.0, 6000.0, 1000.0]
velocity = [250.0, 0.0, 0.0]
"""


# --------------------------------------------------------------------------
# The form and the validator describe the same schema
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("manoeuvre", "law", "estimator"),
    list(
        itertools.product(
            ("straight", "weave", "break_turn", "barrel_roll", "jink"),
            ("pronav", "apn", "pursuit", "none"),
            ("ekf", "alpha_beta", "none"),
        )
    ),
)
def test_every_form_combination_is_a_scenario_the_validator_accepts(
    manoeuvre: str, law: str, estimator: str
) -> None:
    """Sixty combinations, and each has to survive the real reader.

    An unknown key is an error rather than a shrug, which is what gives this
    test its teeth: a control that names a key wrong, or that stays visible
    under a manoeuvre that rejects it, fails here rather than the first time
    somebody picks that combination in the window.
    """
    data = default_scenario(
        **{
            "target.manoeuvre.kind": manoeuvre,
            "guidance.law": law,
            "estimator.kind": estimator,
        }
    )
    spec = resolve(data, "form")
    assert spec.scenario.duration > 0.0


def _fly_briefly(text: str) -> np.ndarray:
    """Half a second of flight, as positions. Enough to tell two configs apart."""
    spec = loads(text, "probe")
    world, detector = spec.build(seed=0)
    result = run(world, duration=0.5, dt=1e-3, stop=detector)
    return np.asarray(result.recorder.position("missile"))


@pytest.mark.parametrize(
    "field",
    # Required keys have no "omitted" behaviour to match — a scenario without
    # them does not load — so their default is only a seed for the form. A
    # `blank_is_default` key is written only when set, which is the case this
    # test would be checking, so there is nothing left to check.
    [f for f in FIELDS if not f.required and not f.blank_is_default],
    ids=lambda f: f.path,
)
def test_each_form_default_is_what_omitting_the_key_would_mean(field: Any) -> None:
    """The defaults in the form must be the validator's own.

    Otherwise a scenario that leaves a key out shows one value in the form and
    behaves as another — the form quietly lying about a file it did not write.
    Rather than compare two specs (which hold numpy arrays and closures and do
    not usefully compare), this flies both for half a second and insists the
    missile went to the same places.

    Caught two: the manoeuvre defaulted to a weave in the form and to straight
    in the validator, and a break turn defaulted to breaking at eight seconds
    here and at zero there.
    """
    values: dict[str, Any] = {field.path: field.default}
    # A field gated behind a choice needs that choice made, or it is not written
    # at all and the comparison is vacuous. `when` states its values as text,
    # because that is how a form control reports them; a toggle wants the
    # boolean back before it reaches the validator.
    if field.when is not None:
        gate, wanted = field.when[0], field.when[1][0]
        controller = next(f for f in FIELDS if f.path == gate)
        values[gate] = wanted == "true" if controller.kind == "toggle" else wanted

    stated = default_scenario(**values)
    omitted = default_scenario(**values)
    parts = field.path.split(".")
    table = omitted
    for part in parts[:-1]:
        table = table[part]
    del table[parts[-1]]

    assert np.allclose(_fly_briefly(_as_toml(stated)), _fly_briefly(_as_toml(omitted))), (
        f"{field.path}: the form offers {field.default!r} as the default, but leaving the key "
        f"out of a scenario file means something else"
    )


def _as_toml(data: dict[str, Any], indent: str = "") -> str:
    """A nested dict as TOML. Only the shapes a scenario uses.

    Written here rather than pulled in as a dependency: the package installs
    with numpy alone and a test is not a reason to change that.
    """
    del indent
    lines: list[str] = []
    tables: list[tuple[str, dict[str, Any]]] = [("", data)]
    while tables:
        prefix, table = tables.pop(0)
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if prefix and scalars:
            lines.append(f"[{prefix}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_scalar(value)}")
        if scalars:
            lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                tables.append((f"{prefix}.{key}" if prefix else key, value))
    return "\n".join(lines) + "\n"


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


# --------------------------------------------------------------------------
# The field table itself
# --------------------------------------------------------------------------
def test_every_field_belongs_to_a_declared_group() -> None:
    for field in FIELDS:
        assert field.group in GROUPS, f"{field.path} is in an undeclared group"


def test_turning_the_seeker_off_puts_the_whole_estimator_away() -> None:
    """Visibility is transitive, and this is the case that needs it.

    With no seeker the guidance law is handed the truth and ``build`` ignores
    whatever filter is configured. A rule that looked only one level up would
    hide the estimator's name and go on offering its jerk sigma — a control
    doing nothing, which is exactly what the command line refuses to imply when
    it says "perfect information" instead of naming a filter.
    """
    perfect = default_scenario(**{"seeker.enabled": False})
    assert "estimator" not in perfect
    assert perfect["seeker"] == {"enabled": False}

    with_seeker = default_scenario()
    assert "jerk_sigma" in with_seeker["estimator"]


def test_a_blank_default_writes_no_key_at_all() -> None:
    """Beta's default is computed from alpha, so the form must not state one.

    Writing ``beta = 0`` would be rejected outright, and writing any other
    number would silently replace the critically damped value the validator
    would otherwise choose.
    """
    data = default_scenario(**{"estimator.kind": "alpha_beta"})
    assert "beta" not in data["estimator"]
    assert "alpha" in data["estimator"]


def test_no_two_fields_write_the_same_key() -> None:
    paths = [field.path for field in FIELDS]
    assert len(paths) == len(set(paths))


def test_a_conditional_field_points_at_a_field_that_exists() -> None:
    """A `when` naming a path that is not a control can never become true."""
    paths = {field.path for field in FIELDS}
    for field in FIELDS:
        if field.when is not None:
            assert field.when[0] in paths, f"{field.path} is gated on unknown {field.when[0]}"


def test_the_form_covers_every_setting_the_reference_scenario_documents() -> None:
    """The drift check in the other direction.

    ``reference.toml`` is the annotated file that lists every setting there is,
    and it is already tested to load. If a setting appears there and has no
    control here, the form has fallen behind the simulation — so either add one
    or say in `OMITTED` why not.
    """
    documented = set()
    table = ""
    for raw in bundled_text("reference").splitlines():
        line = raw.strip().lstrip("#").strip()
        header = re.fullmatch(r"\[([\w.]+)\]", line)
        if header:
            table = header.group(1)
            continue
        key = re.match(r"([a-z_]\w*)\s*=", line)
        if key:
            documented.add(f"{table}.{key.group(1)}" if table else key.group(1))

    covered = {field.path for field in FIELDS}
    missing = {
        path
        for path in documented
        if path not in covered and path.rsplit(".", 1)[-1] not in OMITTED and path not in OMITTED
    }
    assert not missing, f"documented settings with no control and no reason: {sorted(missing)}"


def test_the_json_the_page_receives_carries_what_it_needs_to_draw_a_control() -> None:
    for entry in as_json():
        assert entry["path"], "a control with no key writes nowhere"
        assert entry["label"], f"{entry['path']} has nothing to call itself"
        assert entry["kind"], f"{entry['path']} has no way to be drawn"
        if entry["kind"] == "choice":
            assert entry["choices"], f"{entry['path']} is a choice with nothing to choose"
        if entry["kind"] == "number":
            assert entry["min"] is not None, f"{entry['path']} has no lower bound to drag to"
            assert entry["max"] is not None, f"{entry['path']} has no upper bound to drag to"


# --------------------------------------------------------------------------
# Flying one
# --------------------------------------------------------------------------
def test_a_run_produces_frames_that_index_into_the_paths() -> None:
    """The payload's whole shape rests on this.

    Frames carry an index rather than their own copy of the trail — a cumulative
    trail per frame is quadratic — so an index that does not address the path is
    a renderer drawing the wrong thing or crashing.
    """
    payload = fly(FlightRequest({"toml": bundled_text("crossing")}))
    path = payload["paths"]["missile"]
    assert payload["frames"]
    for frame in payload["frames"]:
        assert 0 <= frame["i"] < len(path)
    assert [f["t"] for f in payload["frames"]] == sorted(f["t"] for f in payload["frames"])


def test_the_payload_says_what_was_flown() -> None:
    payload = fly(FlightRequest({"toml": bundled_text("augmented")}))
    assert "APN (N=3)" in payload["subtitle"]
    assert payload["intercept"] is not None
    assert payload["intercept"]["hit"] is True


def test_the_same_seed_gives_the_same_flight() -> None:
    first = fly(FlightRequest({"toml": bundled_text("crossing"), "seed": 4}))
    second = fly(FlightRequest({"toml": bundled_text("crossing"), "seed": 4}))
    assert first["intercept"] == second["intercept"]


def test_a_different_seed_gives_a_different_one() -> None:
    first = fly(FlightRequest({"toml": bundled_text("crossing"), "seed": 1}))
    second = fly(FlightRequest({"toml": bundled_text("crossing"), "seed": 2}))
    assert first["intercept"]["missDistance"] != second["intercept"]["missDistance"]


def test_a_run_longer_than_the_server_will_fly_is_refused() -> None:
    """A page can ask for anything; a worker thread is not free."""
    text = f"duration = {MAX_DURATION + 1}\n{MINIMAL}"
    with pytest.raises(ConfigError, match="more than this server will fly"):
        fly(FlightRequest({"toml": text}))


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ({}, "expected some TOML text"),
        ({"toml": "   "}, "expected some TOML text"),
        ({"toml": MINIMAL, "seed": -1}, "between 0 and"),
        ({"toml": MINIMAL, "seed": 1.5}, "whole number"),
        ({"toml": MINIMAL, "fps": 0}, "between 1 and 120"),
        ({"toml": MINIMAL, "speed": "fast"}, "expected a number"),
    ],
)
def test_a_malformed_request_is_rejected_before_anything_is_flown(
    body: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ConfigError, match=problem):
        FlightRequest(body)


# --------------------------------------------------------------------------
# Over a socket
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def address() -> Iterator[str]:
    """A real server on a real ephemeral port, for the length of the module."""
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield address_of(server).rstrip("/")
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _post(url: str, body: dict[str, Any]) -> tuple[int, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_is_served(address: str) -> None:
    with urllib.request.urlopen(f"{address}/", timeout=30) as response:
        body = response.read().decode()
    assert response.status == 200
    assert '<canvas id="view">' in body
    assert "/app.js" in body


@pytest.mark.parametrize("asset", ["/app.js", "/style.css"])
def test_the_assets_are_served(address: str, asset: str) -> None:
    with urllib.request.urlopen(f"{address}{asset}", timeout=30) as response:
        assert response.status == 200
        assert len(response.read()) > 500


def test_there_is_no_path_to_traverse(address: str) -> None:
    """Assets are a fixed table, so this is a 404 rather than a file read."""
    for attempt in ("/../pyproject.toml", "/static/../../etc/passwd", "/etc/passwd"):
        status, _ = _get(f"{address}{attempt}")
        assert status == 404


def test_the_scenario_list_is_the_bundled_one(address: str) -> None:
    status, listing = _get(f"{address}/api/scenarios")
    assert status == 200
    names = {entry["name"] for entry in listing}
    assert {"crossing", "augmented", "reference"} <= names
    assert all(entry["description"] for entry in listing)


def test_a_scenario_arrives_as_both_text_and_values(address: str) -> None:
    status, data = _get(f"{address}/api/scenario/crossing")
    assert status == 200
    assert data["values"] == tomllib.loads(data["toml"])
    assert "# " in data["toml"], "the comments are most of what a shipped scenario is for"


def test_an_unknown_scenario_suggests_the_closest(address: str) -> None:
    status, data = _get(f"{address}/api/scenario/crssing")
    assert status == 404
    assert "did you mean 'crossing'" in data["error"]


def test_the_field_table_is_served(address: str) -> None:
    status, fields = _get(f"{address}/api/fields")
    assert status == 200
    assert len(fields) == len(FIELDS)


def test_running_over_http_returns_a_drawable_payload(address: str) -> None:
    status, payload = _post(f"{address}/api/run", {"toml": bundled_text("crossing"), "seed": 0})
    assert status == 200
    assert payload["frames"], "nothing to play"
    assert payload["paths"]["missile"], "nothing to draw"


def test_a_scenario_the_validator_rejects_comes_back_as_advice(address: str) -> None:
    """The whole reason the browser sends TOML rather than its own dictionary."""
    status, payload = _post(f"{address}/api/run", {"toml": f"{MINIMAL}\n[seeker]\nglint = 1.5\n"})
    assert status == 400
    assert "did you mean 'glint_sigma'" in payload["error"]


def test_an_unknown_route_is_a_404_not_a_traceback(address: str) -> None:
    status, payload = _get(f"{address}/api/nothing")
    assert status == 404
    assert "no such path" in payload["error"]
