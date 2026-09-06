"""The half of the browser app that needs a browser.

Marked ``gui`` and deselected in CI, exactly like the VPython viewer, because a
build server has no browser and installing one to run four assertions is not a
trade worth making. Everything that can be checked without one already is, in
``test_web.py``; what is left genuinely cannot be.

What is left is worth checking, though, and it is the part that most easily rots
without anyone noticing: whether the page actually *draws*. A canvas that throws
inside its render loop still serves a 200 and still shows a form. Nothing on the
Python side can tell the difference between that and a working viewer — only
counting the pixels can.

Run it here with::

    pytest -m gui
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from interceptor.web.app import address_of, make_server

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="the browser tests need playwright"
)

pytestmark = pytest.mark.gui

#: A crossing engagement is a few real seconds of integration, and the page
#: flies one on load. Everything here waits on that rather than guessing.
PATIENCE = 60_000


def _note(failures: list[str], message: Any) -> None:
    """Record a console error, unless the test suite asked for it.

    One test posts a scenario the validator must reject, and the browser logs
    every non-2xx response as a console error. That one is the point of the
    test, so counting it as a fault would make a passing feature look broken.
    """
    if message.type != "error":
        return
    if "Failed to load resource" in message.text:
        return
    failures.append(message.text)


@pytest.fixture(scope="module")
def address() -> Iterator[str]:
    server = make_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield address_of(server).rstrip("/")
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def page(address: str) -> Iterator[Any]:
    """One browser, one page, flown once. Every test below reads that state."""
    with playwright_api.sync_playwright() as driver:
        browser = driver.chromium.launch()
        tab = browser.new_page(viewport={"width": 1280, "height": 800})
        failures: list[str] = []
        tab.on("pageerror", lambda error: failures.append(str(error)))
        tab.on("console", lambda message: _note(failures, message))
        tab.goto(address)
        # The status line stops saying "flying" when the first run lands.
        tab.wait_for_function(
            "() => /hit|miss|no closest/.test(document.querySelector('#status').textContent)",
            timeout=PATIENCE,
        )
        tab.failures = failures
        yield tab
        browser.close()


def test_the_page_flies_a_scenario_on_its_own(page: Any) -> None:
    """It arrives with something on screen rather than an empty stage."""
    assert "hit by" in page.text_content("#status")
    assert "ProNav" in page.text_content("#subtitle")


def test_the_canvas_actually_draws(page: Any) -> None:
    """The assertion no amount of server-side testing can make.

    Counts pixels the renderer put down. A projection that throws, a camera
    behind the scene, or a payload the drawing code cannot read all leave this
    at zero while every HTTP response stays a cheerful 200.
    """
    painted = page.evaluate(
        """() => {
            const canvas = document.getElementById('view');
            const pixels = canvas.getContext('2d')
                .getImageData(0, 0, canvas.width, canvas.height).data;
            let count = 0;
            for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 8) count += 1;
            return count;
        }"""
    )
    assert painted > 2000, f"the view is essentially blank: {painted} painted pixels"


def test_the_readout_reports_the_engagement(page: Any) -> None:
    readout = page.text_content("#readout")
    for label in ("range", "closing", "LOS", "used", "speed"):
        assert label in readout


def test_scrubbing_moves_the_clock(page: Any) -> None:
    page.fill("#scrub", "0")
    page.dispatch_event("#scrub", "input")
    start = page.text_content("#clock")

    page.fill("#scrub", page.get_attribute("#scrub", "max"))
    page.dispatch_event("#scrub", "input")
    end = page.text_content("#clock")

    assert start != end
    assert start.startswith("0.00")


def test_choosing_a_manoeuvre_changes_which_controls_exist(page: Any) -> None:
    """The `when` rules, as the person using them experiences them.

    A break turn has a start time and no period; a weave is the other way
    round. Getting this wrong does not merely look untidy — the validator
    rejects a period under a break turn, so a form that showed both would build
    scenarios that cannot be read.
    """
    page.select_option("#f\\:target\\.manoeuvre\\.kind", "break_turn")
    labels = page.eval_on_selector_all(
        "#controls .field label span:first-child", "els => els.map(e => e.textContent)"
    )
    assert "Breaks at" in labels
    assert "Period" not in labels

    page.select_option("#f\\:target\\.manoeuvre\\.kind", "weave")
    labels = page.eval_on_selector_all(
        "#controls .field label span:first-child", "els => els.map(e => e.textContent)"
    )
    assert "Period" in labels
    assert "Breaks at" not in labels


def test_the_form_writes_the_scenario_the_server_will_read(page: Any) -> None:
    """Touching a control rebuilds the TOML pane, because that is what is sent."""
    page.select_option("#f\\:guidance\\.law", "apn")
    text = page.input_value("#toml")
    assert 'law = "apn"' in text
    assert "[target.manoeuvre]" in text


def test_a_scenario_the_validator_rejects_is_explained_in_the_page(page: Any) -> None:
    """The validator's own words, in front of the person who caused them."""
    page.evaluate(
        """() => {
            const box = document.getElementById('toml');
            box.value = '[missile]\\nposition=[0,0,1000]\\n'
                + '[target]\\nposition=[0,6000,1000]\\nvelocity=[250,0,0]\\n'
                + '[seeker]\\nglint = 1.5\\n';
            box.dispatchEvent(new Event('input'));
        }"""
    )
    page.click("#fly")
    page.wait_for_selector("#error.shown", timeout=PATIENCE)
    assert "did you mean 'glint_sigma'" in page.text_content("#error")


def test_nothing_threw_along_the_way(page: Any) -> None:
    """Every test above ran against this page; none of them may have thrown.

    Last on purpose, so it reports errors raised by any of them.
    """
    assert page.failures == []
