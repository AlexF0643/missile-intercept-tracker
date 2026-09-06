"""The engagement, in a browser window, from the standard library.

``interceptor view`` opens a real 3D window through VPython and ``interceptor
record`` writes a GIF, and between them they cover watching an engagement — but
neither lets you *change* one. Changing the target's speed still meant editing a
file and running the command again. This is the window that closes that loop:
the 3D view and every parameter, side by side, re-flown without leaving the page.

**No web framework and no CDN.** ``http.server`` is in the standard library and
this is a single-user tool on ``localhost``, which is precisely the case it is
adequate for. The browser side draws with a canvas and a hand-rolled projection
rather than three.js, so the whole app works with no network at all — which is
the difference between a tool that runs on a laptop on a train and one that does
not. The package still installs with nothing but numpy.

**One validation path.** The browser never decides what a valid scenario is. It
sends TOML — the same text ``interceptor show --raw`` prints — and this module
hands it to :func:`interceptor.config.loads`. Every bound, every default and
every "did you mean 'glint_sigma'?" is the one the command line already uses, so
the two cannot drift apart and a mistake made in the browser is explained as
well as a mistake made in a file.

**What it will not do.** Static files are served from a fixed table of names,
not by joining a path, so there is nothing to traverse; the only file content
this server will ever read from disk is its own three assets and the bundled
scenarios. It binds to the loopback address unless told otherwise, and it caps
the simulated duration of a run, because a browser tab that can ask for a
thirty-thousand-second engagement can hold a worker thread for an hour.
"""

from __future__ import annotations

import json
import threading
import tomllib
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Final

from interceptor.config import ConfigError, bundled_names, bundled_text, loads
from interceptor.sim.engagement import run
from interceptor.viz.scene import storyboard_from
from interceptor.web.fields import as_json
from interceptor.web.payload import storyboard_payload

__all__ = ["FlightRequest", "address_of", "fly", "make_server", "serve"]

#: Where the simulation itself is integrated, seconds. The same 1 kHz the CLI
#: uses; a browser is not a reason to integrate differently.
PHYSICS_STEP: Final = 1e-3

#: Simulated seconds a single request may ask for. Generous against the 30 s
#: every shipped scenario uses, and finite, which is the point: this is a server
#: taking instructions from a page, and the page can be wrong.
MAX_DURATION: Final = 120.0

#: Bytes. A scenario is a couple of kilobytes of TOML; anything approaching this
#: is not one.
MAX_BODY = 256 * 1024

_STATIC = resources.files("interceptor.web") / "static"

#: Served files, by URL path. A fixed table rather than a directory join: there
#: is then no such thing as a path this server can be talked into reading.
_ASSETS: Final[dict[str, tuple[str, str]]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class FlightRequest:
    """One validated ``/api/run`` request.

    Parsing is separated from flying so that a bad request is rejected before a
    worker thread spends two seconds integrating one.
    """

    def __init__(self, body: dict[str, Any]) -> None:
        self.toml = _string(body, "toml")
        self.seed = _whole(body, "seed", default=0, minimum=0, maximum=1_000_000)
        self.fps = _real(body, "fps", default=30.0, minimum=1.0, maximum=120.0)
        self.speed = _real(body, "speed", default=1.0, minimum=0.05, maximum=20.0)
        self.slow_motion = bool(body.get("slowMotion", True))


def _string(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{key}: expected some TOML text"
        raise ConfigError(msg)
    return value


def _real(
    body: dict[str, Any], key: str, *, default: float, minimum: float, maximum: float
) -> float:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key}: expected a number, got {type(value).__name__}"
        raise ConfigError(msg)
    if not minimum <= float(value) <= maximum:
        msg = f"{key}: must be between {minimum:g} and {maximum:g}, got {value:g}"
        raise ConfigError(msg)
    return float(value)


def _whole(body: dict[str, Any], key: str, *, default: int, minimum: int, maximum: int) -> int:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key}: expected a whole number, got {type(value).__name__}"
        raise ConfigError(msg)
    if not minimum <= value <= maximum:
        msg = f"{key}: must be between {minimum} and {maximum}, got {value}"
        raise ConfigError(msg)
    return value


def fly(request: FlightRequest) -> dict[str, Any]:
    """Validate, fly, and return what the browser needs to draw it.

    Raises:
        ConfigError: The scenario could not be read, or asks for more simulated
            time than one request is allowed. Both are the caller's fault and
            both become a 400 with the message shown in the page.
    """
    spec = loads(request.toml, "scenario")
    if spec.scenario.duration > MAX_DURATION:
        msg = (
            f"duration: {spec.scenario.duration:g} s is more than this server will fly "
            f"in one request ({MAX_DURATION:g} s). Run it from the command line instead."
        )
        raise ConfigError(msg)

    world, detector = spec.build(seed=request.seed)
    result = run(world, duration=spec.scenario.duration, dt=PHYSICS_STEP, stop=detector)
    intercept = detector.result

    slow_from = None
    if request.slow_motion and intercept is not None:
        # The endgame is what anybody watches for and it is over in moments.
        slow_from = max(intercept.time - 1.5, 0.0)

    storyboard = storyboard_from(
        result,
        fps=request.fps,
        speed=request.speed,
        intercept=intercept,
        title=spec.name,
        slow_motion_from=slow_from,
    )
    return storyboard_payload(
        storyboard,
        missile_path=result.recorder.position("missile"),
        target_path=result.recorder.position("target"),
        subtitle=spec.describe(),
    )


def _scenario_list() -> list[dict[str, str]]:
    from interceptor.config import load_bundled

    return [
        {"name": name, "description": load_bundled(name).description} for name in bundled_names()
    ]


class Handler(BaseHTTPRequestHandler):
    """Routing, and nothing else.

    Every branch here is a lookup or a status code. Anything that computes
    belongs in :func:`fly` or :mod:`interceptor.web.payload`, where a test can
    reach it without opening a socket.
    """

    server_version = "interceptor"
    protocol_version = "HTTP/1.1"

    #: Set by :func:`make_server`. Request logging is off by default because a
    #: line per asset per reload buries the one message worth reading, which is
    #: the URL to open.
    verbose = False

    def log_message(self, format: str, *args: Any) -> None:
        if self.verbose:
            super().log_message(format, *args)

    # -- responses ---------------------------------------------------------
    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page is generated from the code it ships with; a cached asset
            # against a re-run server is a confusing way to debug a change.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The tab was closed or reloaded while this was being written. That
            # is a completely ordinary thing for a browser to do, and it is not
            # worth a traceback in a window whose only other output is the URL
            # to open — which is exactly what the traceback would bury.
            self.close_connection = True

    def _json(self, status: HTTPStatus, payload: dict[str, Any] | list[Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in _ASSETS:
            name, content_type = _ASSETS[path]
            self._send(HTTPStatus.OK, (_STATIC / name).read_bytes(), content_type)
            return

        if path == "/api/fields":
            self._json(HTTPStatus.OK, as_json())
            return

        if path == "/api/scenarios":
            self._json(HTTPStatus.OK, _scenario_list())
            return

        if path.startswith("/api/scenario/"):
            name = path.removeprefix("/api/scenario/")
            try:
                text = bundled_text(name)
            except ConfigError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            # Both the text and the parsed table. The page fills its form from
            # the table — a key the file omits is a key at its default, which is
            # what the validator will make of it too — and shows the text in the
            # editor, comments and all, because the shipped scenarios explain
            # themselves and that is most of their value.
            self._json(
                HTTPStatus.OK,
                {"name": name, "toml": text, "values": tomllib.loads(text)},
            )
            return

        self._error(HTTPStatus.NOT_FOUND, f"no such path: {path}")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/run":
            self._error(HTTPStatus.NOT_FOUND, f"no such path: {self.path}")
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "that is not a scenario")
            return

        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            self._error(HTTPStatus.BAD_REQUEST, f"not valid JSON — {error}")
            return
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "expected a JSON object")
            return

        try:
            payload = fly(FlightRequest(body))
        except ConfigError as error:
            # The scenario is wrong, and the message says how. This is the
            # normal path for a person editing settings, not an exception.
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return

        self._json(HTTPStatus.OK, payload)


def make_server(
    host: str = "127.0.0.1", port: int = 8765, *, verbose: bool = False
) -> ThreadingHTTPServer:
    """Build the server without starting it.

    Separate from :func:`serve` so a test can bind port 0, run it on a thread
    and ask it real questions over a real socket.
    """
    handler = type("BoundHandler", (Handler,), {"verbose": verbose})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def address_of(server: ThreadingHTTPServer) -> str:
    """The URL to open for a running server.

    Worth a function because port 0 means "any free port", so the port a server
    is actually on is only knowable after it is bound — which is how the tests
    run one without picking a number and hoping.
    """
    host, port = server.server_address[0], server.server_address[1]
    name = host.decode() if isinstance(host, bytes) else str(host)
    # 0.0.0.0 is an instruction about what to listen on, not somewhere to visit.
    if name in {"127.0.0.1", "0.0.0.0", "::", "::1"}:
        name = "localhost"
    return f"http://{name}:{port}/"


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    open_browser: bool = True,
    verbose: bool = False,
) -> None:
    """Run the app until interrupted."""
    server = make_server(host, port, verbose=verbose)
    url = address_of(server)
    print(f"interceptor is running at {url}")
    print("Ctrl-C to stop.")

    if open_browser:
        # On a timer, because the browser has to have something to connect to:
        # `serve_forever` below is what starts answering.
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
