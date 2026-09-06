"""The command-line interface.

Exercised through :func:`interceptor.cli.main` with an explicit argument list
rather than by launching a subprocess. Same code path, no interpreter start-up,
and a failure points at a line rather than at a shell.

Exit statuses are asserted alongside the output because they are the part a
script consuming this tool actually reads: a run that misses is not a crash, a
scenario file with a typo in it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interceptor.cli import main

CROSSING = """
name = "quick"
duration = 12.0

[missile]
position = [0.0, 0.0, 1000.0]
speed = 60.0

[target]
position = [0.0, 3000.0, 1000.0]
velocity = [200.0, 0.0, 0.0]

[guidance]
law = "pronav"

[seeker]
enabled = false
"""


@pytest.fixture
def scenario(tmp_path: Path) -> Path:
    """A short engagement, so the CLI tests do not each fly for 30 seconds."""
    path = tmp_path / "quick.toml"
    path.write_text(CROSSING, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# list and show
# --------------------------------------------------------------------------
def test_list_names_every_bundled_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "crossing" in out
    assert "perfect-information" in out


def test_show_describes_a_scenario_without_flying_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["show", "crossing"]) == 0
    out = capsys.readouterr().out
    assert "initial separation" in out
    assert "6000 m" in out
    assert "miss distance" not in out


def test_show_raw_prints_a_file_that_can_be_edited_and_rerun(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The documented way to start your own scenario, so it has to round-trip."""
    assert main(["show", "crossing", "--raw"]) == 0
    copied = tmp_path / "mine.toml"
    copied.write_text(capsys.readouterr().out, encoding="utf-8")
    assert main(["show", str(copied)]) == 0


def test_show_names_the_law_with_its_navigation_constant(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Class names would read `ProportionalNavigation` against
    `AugmentedProportionalNavigation` — a prefix apart, with the constant that
    changes the behaviour invisible in both."""
    assert main(["show", "augmented"]) == 0
    assert "APN (N=3)" in capsys.readouterr().out
    assert main(["show", "crossing"]) == 0
    assert "ProNav (N=3)" in capsys.readouterr().out


def test_show_reports_perfect_information_without_naming_an_estimator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A filter behind a flawless track is not running; saying otherwise misleads."""
    assert main(["show", "perfect-information"]) == 0
    out = capsys.readouterr().out
    assert "perfect information" in out
    assert "EKF" not in out


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def test_run_flies_a_scenario_from_a_path(
    scenario: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(scenario)]) == 0
    out = capsys.readouterr().out
    assert "miss distance" in out
    assert "HIT" in out


def test_a_seed_changes_the_outcome(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Proof the seed reaches the seeker rather than being accepted and dropped."""
    path = tmp_path / "noisy.toml"
    path.write_text(CROSSING.replace("enabled = false", "enabled = true"), encoding="utf-8")

    main(["run", str(path), "--seed", "1"])
    first = capsys.readouterr().out
    main(["run", str(path), "--seed", "2"])
    second = capsys.readouterr().out
    assert first != second


def test_the_same_seed_gives_the_same_answer(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Determinism from a seed, asserted where a user would notice it break."""
    path = tmp_path / "noisy.toml"
    path.write_text(CROSSING.replace("enabled = false", "enabled = true"), encoding="utf-8")

    main(["run", str(path), "--seed", "7"])
    first = capsys.readouterr().out
    main(["run", str(path), "--seed", "7"])
    assert capsys.readouterr().out == first


def test_run_can_write_a_figure(scenario: Path, tmp_path: Path) -> None:
    """Not marked ``gui``: figures are built on an Agg canvas and need no display,
    only the optional matplotlib extra."""
    pytest.importorskip("matplotlib")
    figure = tmp_path / "engagement.png"
    assert main(["run", str(scenario), "--figure", str(figure)]) == 0
    assert figure.exists()
    assert figure.stat().st_size > 5_000


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def test_sweep_reports_a_distribution_rather_than_one_number(
    scenario: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One run of a stochastic engagement is an anecdote."""
    assert main(["sweep", str(scenario), "--seeds", "3"]) == 0
    out = capsys.readouterr().out
    for heading in ("median", "best", "worst", "hits"):
        assert heading in out


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
def test_a_bad_scenario_exits_with_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "broken.toml"
    path.write_text(CROSSING.replace("enabled = false", "glint = 1.5"), encoding="utf-8")
    assert main(["run", str(path)]) == 2
    assert "glint_sigma" in capsys.readouterr().err


def test_an_unknown_scenario_name_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "crssing"]) == 2
    assert "did you mean 'crossing'" in capsys.readouterr().err


def test_no_command_is_a_usage_error() -> None:
    """argparse exits rather than returning, and that is the correct behaviour."""
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


# --------------------------------------------------------------------------
# monte-carlo
# --------------------------------------------------------------------------
def test_monte_carlo_reports_both_kinds_of_uncertainty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tiny, because the machinery is tested in test_montecarlo.py.

    What this checks is the thing only the command can get wrong: that it prints
    both answers rather than the flattering one. A study that reported only the
    interval from the noise would look more confident and say less.
    """
    assert main(["monte-carlo", "head-on", "--draws", "2", "--seeds", "2", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "With the constants this project ships" in out
    assert "Across constants that are equally plausible" in out
    assert "probability of kill" in out


def test_monte_carlo_with_one_draw_says_what_it_is_leaving_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--draws 1` is the conventional study, and it should admit as much."""
    assert main(["monte-carlo", "head-on", "--draws", "1", "--seeds", "2", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "noise only" in out
    assert "Across constants" not in out
