/* The browser half of the app: a form, a projection and a transport.
 *
 * Three things happen here and nothing else. The form is generated from the
 * field table the server sends and writes a TOML scenario. The renderer
 * projects the two trajectories the server returns onto a canvas. The transport
 * walks through the frames. There is no simulation in this file — every number
 * drawn was computed in Python, which is where the physics is allowed to live.
 *
 * Why a canvas and a hand-rolled projection rather than three.js: a perspective
 * transform of two polylines and a ground grid is about forty lines, and doing
 * it here means the whole app runs with no network beyond localhost. A tool
 * that needs a CDN to draw is a tool that stops working on a train.
 */

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------
const state = {
  fields: [],
  values: {}, // dotted path -> value
  meta: { name: "scenario", description: "" },
  payload: null,
  frame: 0,
  playing: false,
  tomlDirty: false, // the pane was hand-edited; do not overwrite it
};

//: Orbit around the framed centre. `distance` is in multiples of the scene's
//: own radius, so the same numbers frame a six-kilometre engagement and a
//: six-hundred-metre one. 1.9 fits a sphere of that radius inside a 45° vertical
//: field of view with room at the edges for the trails to breathe.
const HOME = { azimuth: -0.75, elevation: 0.34, distance: 2.2 };
const camera = { ...HOME };

const el = (id) => document.getElementById(id);
const canvas = el("view");
const context = canvas.getContext("2d");

// --------------------------------------------------------------------------
// Building a scenario file
// --------------------------------------------------------------------------
/**
 * Whether a field's `when` condition is satisfied by the current values.
 *
 * Walks the chain, not one link of it: the estimator's tuning is gated on which
 * estimator, which is gated on there being a seeker at all. Checking only the
 * immediate gate would hide the estimator and keep offering its jerk sigma.
 * The same rule is implemented in `fields.py`, and the two must agree, because
 * one decides what you see and the other decides what gets written.
 */
function shown(field) {
  let current = field;
  while (current && current.when) {
    const gate = current.when.path;
    const value = state.values[gate];
    const text = typeof value === "boolean" ? String(value) : String(value ?? "");
    if (!current.when.values.includes(text)) return false;
    current = state.fields.find((f) => f.path === gate);
  }
  return true;
}

/** A number as TOML: floats keep a decimal point, integers do not. */
function asToml(value, kind) {
  if (Array.isArray(value)) return `[${value.map((v) => asToml(v, "number")).join(", ")}]`;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return JSON.stringify(value);
  if (kind === "integer") return String(Math.round(value));
  // Keep it short but unambiguous: 0.002 rather than 0.0020000000000000005, and
  // 30.0 rather than 30, because a bare 30 is an integer in TOML.
  const rounded = Number(value.toPrecision(12));
  return Number.isInteger(rounded) ? rounded.toFixed(1) : String(rounded);
}

/**
 * The form as a scenario file.
 *
 * Tables appear in the order the fields declare them, which puts the bare keys
 * first and every parent table before its children — both of which TOML cares
 * about and neither of which is worth a sorting pass.
 */
function buildToml() {
  const tables = new Map([["", []]]);
  for (const field of state.fields) {
    if (!shown(field)) continue;
    const value = state.values[field.path];
    if (field.blankIsDefault && !value) continue;
    const parts = field.path.split(".");
    const key = parts.pop();
    const table = parts.join(".");
    if (!tables.has(table)) tables.set(table, []);
    tables.get(table).push(`${key} = ${asToml(value, field.kind)}`);
  }

  const lines = [
    `name = ${JSON.stringify(state.meta.name)}`,
    `description = ${JSON.stringify(state.meta.description)}`,
    "",
  ];
  for (const [table, entries] of tables) {
    if (!entries.length) continue;
    if (table) lines.push(`[${table}]`);
    lines.push(...entries, "");
  }
  return lines.join("\n").trimEnd() + "\n";
}

/** Pull the values the form knows about out of a parsed scenario. */
function adoptScenario(name, parsed) {
  state.meta = {
    name: parsed.name ?? name,
    description: parsed.description ?? "",
  };
  for (const field of state.fields) {
    let node = parsed;
    for (const part of field.path.split(".")) {
      node = node && typeof node === "object" ? node[part] : undefined;
    }
    // A key the file leaves out is a key at its default, which is exactly what
    // the validator will do with it too.
    state.values[field.path] = node === undefined ? field.default : node;
  }
}

// --------------------------------------------------------------------------
// The form
// --------------------------------------------------------------------------
/** Speed, heading and climb from a velocity vector, and back again. */
function toBearing([east, north, up]) {
  const speed = Math.hypot(east, north, up);
  const flat = Math.hypot(east, north);
  return {
    speed,
    heading: ((Math.atan2(east, north) * 180) / Math.PI + 360) % 360,
    climb: flat < 1e-9 && Math.abs(up) < 1e-9 ? 0 : (Math.atan2(up, flat) * 180) / Math.PI,
  };
}

function fromBearing(speed, heading, climb) {
  const h = (heading * Math.PI) / 180;
  const c = (climb * Math.PI) / 180;
  return [
    Number((speed * Math.cos(c) * Math.sin(h)).toFixed(4)),
    Number((speed * Math.cos(c) * Math.cos(h)).toFixed(4)),
    Number((speed * Math.sin(c)).toFixed(4)),
  ];
}

function labelled(field, valueText) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const label = document.createElement("label");
  label.htmlFor = `f:${field.path}`;
  label.innerHTML = `<span>${field.label}</span><span class="value" id="v:${field.path}">${valueText}</span>`;
  wrap.append(label);
  return wrap;
}

function helpFor(field) {
  if (!field.help) return null;
  const note = document.createElement("p");
  note.className = "help";
  note.textContent = field.help;
  return note;
}

/** One control. Changing it updates the value and rebuilds anything gated on it. */
function control(field) {
  const set = (value) => {
    state.values[field.path] = value;
    state.tomlDirty = false;
    // Cheap enough to rebuild the lot, and it is the only way a `when` that
    // depends on this field can appear or disappear at the right moment.
    if (field.choices.length || field.kind === "toggle") renderForm();
    else {
      const readout = el(`v:${field.path}`);
      if (readout) readout.textContent = displayValue(field, value);
    }
    syncToml();
  };

  if (field.kind === "choice") {
    const wrap = labelled(field, "");
    const select = document.createElement("select");
    select.id = `f:${field.path}`;
    for (const choice of field.choices) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice.replace(/_/g, " ");
      option.selected = choice === state.values[field.path];
      select.append(option);
    }
    select.addEventListener("change", () => set(select.value));
    wrap.append(select);
    const note = helpFor(field);
    if (note) wrap.append(note);
    return wrap;
  }

  if (field.kind === "toggle") {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const row = document.createElement("label");
    row.className = "row";
    row.style.justifyContent = "flex-start";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.id = `f:${field.path}`;
    box.checked = Boolean(state.values[field.path]);
    box.addEventListener("change", () => set(box.checked));
    row.append(box, document.createTextNode(` ${field.label}`));
    wrap.append(row);
    const note = helpFor(field);
    if (note) wrap.append(note);
    return wrap;
  }

  if (field.kind === "position") {
    const wrap = labelled(field, "m");
    const grid = document.createElement("div");
    grid.className = "triple";
    ["east", "north", "up"].forEach((axis, index) => {
      const cell = document.createElement("div");
      const tag = document.createElement("div");
      tag.className = "axis";
      tag.textContent = axis;
      const input = document.createElement("input");
      input.type = "number";
      input.step = "10";
      input.value = state.values[field.path][index];
      input.addEventListener("change", () => {
        const next = [...state.values[field.path]];
        next[index] = Number(input.value);
        set(next);
      });
      cell.append(tag, input);
      grid.append(cell);
    });
    wrap.append(grid);
    const note = helpFor(field);
    if (note) wrap.append(note);
    return wrap;
  }

  if (field.kind === "velocity") {
    // Speed and a bearing rather than three components. "Fly the target at 250
    // m/s heading 090" is how anybody actually thinks about it; the vector the
    // scenario file wants is arithmetic, so let the page do the arithmetic.
    const current = toBearing(state.values[field.path]);
    const wrap = labelled(field, "");
    const grid = document.createElement("div");
    grid.className = "triple";
    const parts = [
      { tag: "m/s", value: current.speed, step: 10, min: 0, max: 1200 },
      { tag: "heading°", value: current.heading, step: 5, min: 0, max: 360 },
      { tag: "climb°", value: current.climb, step: 1, min: -89, max: 89 },
    ];
    const inputs = [];
    parts.forEach((part) => {
      const cell = document.createElement("div");
      const tag = document.createElement("div");
      tag.className = "axis";
      tag.textContent = part.tag;
      const input = document.createElement("input");
      input.type = "number";
      input.step = String(part.step);
      input.min = String(part.min);
      input.max = String(part.max);
      input.value = Number(part.value.toFixed(2));
      input.addEventListener("change", () => {
        set(fromBearing(Number(inputs[0].value), Number(inputs[1].value), Number(inputs[2].value)));
      });
      inputs.push(input);
      cell.append(tag, input);
      grid.append(cell);
    });
    wrap.append(grid);
    const vector = document.createElement("p");
    vector.className = "help";
    vector.textContent = `[${state.values[field.path].join(", ")}] m/s`;
    wrap.append(vector);
    const note = helpFor(field);
    if (note) wrap.append(note);
    return wrap;
  }

  // A slider and a number, kept in step. The slider is for exploring, the box
  // for saying exactly what you meant.
  const wrap = labelled(field, displayValue(field, state.values[field.path]));
  const row = document.createElement("div");
  row.className = "row";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.id = `f:${field.path}`;
  slider.min = String(field.min ?? 0);
  slider.max = String(field.max ?? 100);
  slider.step = String(field.step ?? 0.01);
  slider.value = String(state.values[field.path]);
  const box = document.createElement("input");
  box.type = "number";
  box.style.width = "88px";
  box.step = String(field.step ?? 0.01);
  box.value = String(state.values[field.path]);

  slider.addEventListener("input", () => {
    box.value = slider.value;
    set(Number(slider.value));
  });
  box.addEventListener("change", () => {
    // Typed values are not clamped to the slider's range: those bounds are for
    // dragging, and the validator has the final say on what is allowed.
    slider.value = box.value;
    set(Number(box.value));
  });

  row.append(slider, box);
  wrap.append(row);
  const note = helpFor(field);
  if (note) wrap.append(note);
  return wrap;
}

function displayValue(field, value) {
  if (typeof value !== "number") return field.unit || "";
  const text = Math.abs(value) < 0.01 && value !== 0 ? value.toExponential(1) : String(value);
  return field.unit ? `${text} ${field.unit}` : text;
}

/** Rebuild the whole sidebar. Cheap, and the only way `when` stays honest. */
function renderForm() {
  const host = el("controls");
  const open = new Set(
    [...host.querySelectorAll("details.group")].filter((d) => d.open).map((d) => d.dataset.group)
  );
  const first = !host.children.length;
  host.textContent = "";

  const groups = new Map();
  for (const field of state.fields) {
    if (!shown(field)) continue;
    if (!groups.has(field.group)) groups.set(field.group, []);
    groups.get(field.group).push(field);
  }

  for (const [group, fields] of groups) {
    const box = document.createElement("details");
    box.className = "group";
    box.dataset.group = group;
    // Open the four panels somebody actually reaches for; the rest stay folded
    // so that forty controls do not arrive as a wall on first sight.
    box.open = first ? ["Target", "Manoeuvre", "Guidance", "Seeker"].includes(group) : open.has(group);
    const summary = document.createElement("summary");
    summary.textContent = group;
    box.append(summary);
    for (const field of fields) box.append(control(field));
    host.append(box);
  }
}

function syncToml() {
  if (!state.tomlDirty) el("toml").value = buildToml();
}

// --------------------------------------------------------------------------
// Projection
// --------------------------------------------------------------------------
const NEAR = 1.0; // metres in front of the eye

const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function normalise(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

/**
 * The camera as a basis plus a focal length.
 *
 * Orbits a fixed centre: azimuth around the vertical, elevation above the
 * horizontal, distance in multiples of the scene's own size — so the same
 * gesture frames a six-kilometre engagement and a six-hundred-metre one alike.
 */
function view() {
  const { lower, upper } = state.payload.bounds;

  // Frame the ground with the flight, not just the flight. The trajectories of
  // a level engagement occupy a few tens of metres of altitude out of several
  // kilometres of ground track, so centring on them alone puts the drawn ground
  // plane entirely in the bottom half of the window and fills the top with sky.
  //
  // How far down to reach is capped at half the horizontal span, so an
  // engagement at ten kilometres frames itself rather than zooming out until
  // the interesting part is a line.
  const span = Math.max(upper[0] - lower[0], upper[1] - lower[1], 1);
  const floor = Math.max(0, lower[2] - 0.5 * span);

  const centre = [
    (lower[0] + upper[0]) / 2,
    (lower[1] + upper[1]) / 2,
    (floor + upper[2]) / 2,
  ];
  const radius =
    Math.max(
      Math.hypot(upper[0] - lower[0], upper[1] - lower[1], upper[2] - floor) / 2,
      100
    ) || 100;

  const offset = [
    Math.cos(camera.elevation) * Math.sin(camera.azimuth),
    Math.cos(camera.elevation) * Math.cos(camera.azimuth),
    Math.sin(camera.elevation),
  ];
  const eye = [
    centre[0] + offset[0] * radius * camera.distance,
    centre[1] + offset[1] * radius * camera.distance,
    centre[2] + offset[2] * radius * camera.distance,
  ];

  const forward = normalise(subtract(centre, eye));
  const right = normalise(cross(forward, [0, 0, 1]));
  const up = cross(right, forward);
  const height = canvas.height / (window.devicePixelRatio || 1);
  return {
    eye,
    forward,
    right,
    up,
    centre,
    radius,
    // 45° vertical field of view.
    focal: height / 2 / Math.tan(Math.PI / 8),
    cx: canvas.width / (window.devicePixelRatio || 1) / 2,
    cy: height / 2,
  };
}

/** World point to camera space. `z` is depth along the view direction. */
function toCamera(v, point) {
  const rel = subtract(point, v.eye);
  return [dot(rel, v.right), dot(rel, v.up), dot(rel, v.forward)];
}

function toScreen(v, camPoint) {
  return [
    v.cx + (v.focal * camPoint[0]) / camPoint[2],
    v.cy - (v.focal * camPoint[1]) / camPoint[2],
  ];
}

/**
 * A segment, clipped to what is in front of the eye.
 *
 * Without this, a point behind the camera projects to a mirrored position in
 * front of it and the trail grows a long wrong line across the screen — which
 * is exactly what happens the moment you zoom inside the engagement, i.e. the
 * moment the view becomes interesting.
 */
function clipped(v, a, b) {
  let p = toCamera(v, a);
  let q = toCamera(v, b);
  if (p[2] < NEAR && q[2] < NEAR) return null;
  if (p[2] < NEAR) p = mix(p, q, (NEAR - p[2]) / (q[2] - p[2]));
  else if (q[2] < NEAR) q = mix(q, p, (NEAR - q[2]) / (p[2] - q[2]));
  return [toScreen(v, p), toScreen(v, q)];
}

const mix = (a, b, t) => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

// --------------------------------------------------------------------------
// Drawing
// --------------------------------------------------------------------------
function palette() {
  const style = getComputedStyle(document.documentElement);
  const read = (name) => style.getPropertyValue(name).trim();
  return {
    missile: read("--missile"),
    target: read("--target"),
    // The figure palette's grid colour is tuned for thin lines on a flat white
    // panel. Laid out in perspective the same value all but disappears, so the
    // ground borrows the slightly stronger one and leans on transparency for
    // its recessiveness instead.
    grid: read("--line"),
    limit: read("--limit"),
    muted: read("--muted"),
    text: read("--text"),
  };
}

function line(v, a, b, colour, width, dash) {
  const segment = clipped(v, a, b);
  if (!segment) return;
  context.save();
  context.strokeStyle = colour;
  context.lineWidth = width;
  if (dash) context.setLineDash(dash);
  context.beginPath();
  context.moveTo(segment[0][0], segment[0][1]);
  context.lineTo(segment[1][0], segment[1][1]);
  context.stroke();
  context.restore();
}

function polyline(v, points, upto, colour, width) {
  context.save();
  context.strokeStyle = colour;
  context.lineWidth = width;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();
  let drawing = false;
  for (let i = 1; i <= upto; i += 1) {
    const segment = clipped(v, points[i - 1], points[i]);
    if (!segment) {
      drawing = false;
      continue;
    }
    if (!drawing) {
      context.moveTo(segment[0][0], segment[0][1]);
      drawing = true;
    }
    context.lineTo(segment[1][0], segment[1][1]);
  }
  context.stroke();
  context.restore();
}

/** A ground plane, so that height means something. */
function ground(v, colours) {
  const { lower, upper } = state.payload.bounds;
  const pad = v.radius * 0.35;
  const x0 = lower[0] - pad;
  const x1 = upper[0] + pad;
  const y0 = lower[1] - pad;
  const y1 = upper[1] + pad;
  // A round number of metres per square, about a dozen squares across.
  const span = Math.max(x1 - x0, y1 - y0);
  const raw = span / 12;
  const step = Math.pow(10, Math.floor(Math.log10(raw))) * [1, 2, 5, 10].find(
    (m) => Math.pow(10, Math.floor(Math.log10(raw))) * m >= raw
  );
  const z = 0;

  for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
    line(v, [x, y0, z], [x, y1, z], colours.grid, 1);
  }
  for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
    line(v, [x0, y, z], [x1, y, z], colours.grid, 1);
  }
  return step;
}

function marker(v, point, colour, radius) {
  const cam = toCamera(v, point);
  if (cam[2] < NEAR) return null;
  const [sx, sy] = toScreen(v, cam);
  context.save();
  context.fillStyle = colour;
  context.beginPath();
  context.arc(sx, sy, radius, 0, Math.PI * 2);
  context.fill();
  context.restore();
  return [sx, sy, cam[2]];
}

function draw() {
  const ratio = window.devicePixelRatio || 1;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, canvas.width / ratio, canvas.height / ratio);
  if (!state.payload) return;

  const colours = palette();
  const v = view();
  const frame = state.payload.frames[state.frame];
  const missilePath = state.payload.paths.missile;
  const targetPath = state.payload.paths.target;
  const missile = missilePath[frame.i];
  const target = targetPath[frame.i];

  const step = ground(v, colours);

  // Height lines. Two dots in a perspective view are ambiguous about altitude
  // until something ties them to the ground.
  line(v, missile, [missile[0], missile[1], 0], colours.limit, 1, [3, 4]);
  line(v, target, [target[0], target[1], 0], colours.limit, 1, [3, 4]);

  // The sightline: the thing the whole guidance chain is about.
  line(v, missile, target, colours.muted, 1, [5, 5]);

  polyline(v, targetPath, frame.i, colours.target, 1.8);
  polyline(v, missilePath, frame.i, colours.missile, 1.8);

  const intercept = state.payload.intercept;
  if (intercept && frame.t >= intercept.time - 1e-9) {
    const cam = toCamera(v, intercept.position);
    if (cam[2] >= NEAR) {
      const [sx, sy] = toScreen(v, cam);
      context.save();
      context.strokeStyle = intercept.hit ? colours.missile : colours.limit;
      context.lineWidth = 2;
      context.beginPath();
      context.arc(sx, sy, 13, 0, Math.PI * 2);
      context.stroke();
      context.restore();
    }
  }

  // Nearer marker last, so it sits on top of the farther one.
  const pair = [
    { point: target, colour: colours.target, radius: 5 },
    { point: missile, colour: colours.missile, radius: 5 },
  ];
  const depths = pair.map((entry) => toCamera(v, entry.point)[2]);
  const order = depths[0] > depths[1] ? [0, 1] : [1, 0];
  for (const index of order) marker(v, pair[index].point, pair[index].colour, pair[index].radius);

  context.save();
  context.fillStyle = colours.muted;
  context.font = "11px ui-monospace, monospace";
  context.fillText(
    `grid ${step >= 1000 ? `${step / 1000} km` : `${step} m`}`,
    14,
    canvas.height / ratio - 12
  );
  context.restore();
}

// --------------------------------------------------------------------------
// The readout
// --------------------------------------------------------------------------
function pad(value, digits, width) {
  return value.toFixed(digits).padStart(width);
}

function showFrame() {
  const payload = state.payload;
  if (!payload) return;
  const frame = payload.frames[state.frame];

  el("readout").innerHTML =
    `<span class="k">t      </span> ${pad(frame.t, 2, 7)} s\n` +
    `<span class="k">range  </span> ${pad(frame.range, 0, 7)} m\n` +
    `<span class="k">closing</span> ${pad(frame.closing, 0, 7)} m/s\n` +
    `<span class="k">LOS    </span> ${pad(frame.los, 3, 7)} rad/s\n` +
    `<span class="k">used   </span> ${pad(frame.used, 1, 7)} g\n` +
    `<span class="k">speed  </span> ${pad(frame.speed, 0, 7)} m/s` +
    outcomeMarkup(frame);

  el("scrub").value = String(state.frame);
  el("clock").textContent = `${frame.t.toFixed(2)} / ${payload.duration.toFixed(2)} s`;
}

function outcomeMarkup(frame) {
  const intercept = state.payload.intercept;
  if (!intercept) return "";
  const last = state.frame === state.payload.frames.length - 1;
  // Closest approach is interpolated between two integration steps, so it
  // usually falls just after the final recorded sample: without the `last`
  // test no frame would ever satisfy the time comparison.
  if (!last && frame.t < intercept.time) return "";
  const verdict = intercept.hit ? "HIT" : "miss";
  return `<div id="outcome" class="${intercept.hit ? "hit" : "miss"}">${verdict} by ${intercept.missDistance.toFixed(2)} m</div>`;
}

// --------------------------------------------------------------------------
// Transport
// --------------------------------------------------------------------------
let lastTick = 0;
let carry = 0;

function tick(now) {
  requestAnimationFrame(tick);
  if (!state.payload) return;

  if (state.playing) {
    const elapsed = Math.min((now - lastTick) / 1000, 0.25);
    carry += elapsed * state.payload.fps * Number(el("rate").value);
    const advance = Math.floor(carry);
    if (advance > 0) {
      carry -= advance;
      state.frame += advance;
      if (state.frame >= state.payload.frames.length) {
        state.frame = state.payload.frames.length - 1;
        setPlaying(false);
      }
      showFrame();
    }
  }
  lastTick = now;
  draw();
}

function setPlaying(playing) {
  state.playing = playing;
  el("play").textContent = playing ? "Pause" : "Play";
  carry = 0;
}

function restart() {
  state.frame = 0;
  setPlaying(true);
  showFrame();
}

// --------------------------------------------------------------------------
// Talking to the server
// --------------------------------------------------------------------------
function setStatus(text, working) {
  const node = el("status");
  node.textContent = text;
  node.classList.toggle("working", Boolean(working));
}

function showError(message) {
  const node = el("error");
  node.textContent = message;
  node.classList.toggle("shown", Boolean(message));
}

async function fly() {
  const body = {
    toml: state.tomlDirty ? el("toml").value : buildToml(),
    seed: Number(el("seed").value) || 0,
    fps: 30,
    slowMotion: true,
  };
  // A crossing engagement is thirteen simulated seconds at a thousand steps a
  // second, through a seeker and a Kalman filter, and that takes a few real
  // ones. Counting them up is the difference between "working" and "hung".
  const began = performance.now();
  const ticking = setInterval(
    () => setStatus(`flying… ${((performance.now() - began) / 1000).toFixed(1)} s`, true),
    100
  );
  setStatus("flying…", true);
  el("fly").disabled = true;
  showError("");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      setStatus("not flown");
      showError(data.error || `server said ${response.status}`);
      return;
    }
    state.payload = data;
    el("subtitle").textContent = data.subtitle;
    el("scrub").max = String(data.frames.length - 1);
    const outcome = data.intercept
      ? `${data.intercept.hit ? "hit" : "miss"} by ${data.intercept.missDistance.toFixed(2)} m`
      : "no closest approach — it never got there";
    setStatus(outcome);
    restart();
  } catch (error) {
    setStatus("not flown");
    showError(String(error));
  } finally {
    clearInterval(ticking);
    el("fly").disabled = false;
  }
}

async function loadScenario(name) {
  const response = await fetch(`/api/scenario/${encodeURIComponent(name)}`);
  const data = await response.json();
  if (!response.ok) {
    showError(data.error || "could not load that scenario");
    return;
  }
  adoptScenario(name, data.values);
  el("scenario-help").textContent = state.meta.description;
  state.tomlDirty = false;
  renderForm();
  syncToml();
  await fly();
}

async function start() {
  const [fields, scenarios] = await Promise.all([
    fetch("/api/fields").then((r) => r.json()),
    fetch("/api/scenarios").then((r) => r.json()),
  ]);
  state.fields = fields;

  const picker = el("scenario");
  for (const scenario of scenarios) {
    const option = document.createElement("option");
    option.value = scenario.name;
    option.textContent = scenario.name;
    picker.append(option);
  }
  picker.value = scenarios.some((s) => s.name === "crossing") ? "crossing" : scenarios[0].name;
  picker.addEventListener("change", () => loadScenario(picker.value));

  await loadScenario(picker.value);
  requestAnimationFrame(tick);
}

// --------------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------------
function resize() {
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(box.width * ratio));
  canvas.height = Math.max(1, Math.round(box.height * ratio));
  draw();
}

new ResizeObserver(resize).observe(canvas);
window.addEventListener("resize", resize);

let dragging = null;
canvas.addEventListener("pointerdown", (event) => {
  dragging = { x: event.clientX, y: event.clientY };
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  camera.azimuth -= (event.clientX - dragging.x) * 0.006;
  camera.elevation = Math.max(
    -1.45,
    Math.min(1.45, camera.elevation + (event.clientY - dragging.y) * 0.005)
  );
  dragging = { x: event.clientX, y: event.clientY };
  draw();
});
const release = (event) => {
  dragging = null;
  canvas.classList.remove("dragging");
  if (event.pointerId !== undefined && canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
};
canvas.addEventListener("pointerup", release);
canvas.addEventListener("pointercancel", release);

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    camera.distance = Math.max(0.15, Math.min(8, camera.distance * Math.exp(event.deltaY * 0.001)));
    draw();
  },
  { passive: false }
);

canvas.addEventListener("dblclick", () => {
  Object.assign(camera, HOME);
  draw();
});

el("play").addEventListener("click", () => {
  if (!state.payload) return;
  if (state.frame >= state.payload.frames.length - 1) restart();
  else setPlaying(!state.playing);
});

el("scrub").addEventListener("input", () => {
  setPlaying(false);
  state.frame = Number(el("scrub").value);
  showFrame();
});

el("fly").addEventListener("click", fly);

el("toggle-toml").addEventListener("click", () => {
  el("toml-pane").classList.toggle("shown");
});

el("toml").addEventListener("input", () => {
  state.tomlDirty = true;
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key === " ") {
    event.preventDefault();
    el("play").click();
  } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
    event.preventDefault();
    setPlaying(false);
    const step = event.key === "ArrowRight" ? 1 : -1;
    state.frame = Math.max(
      0,
      Math.min(state.payload.frames.length - 1, state.frame + step * (event.shiftKey ? 10 : 1))
    );
    showFrame();
  }
});

start().catch((error) => {
  setStatus("failed to start");
  showError(String(error));
});
