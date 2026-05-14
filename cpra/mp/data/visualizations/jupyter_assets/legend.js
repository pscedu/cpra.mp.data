/**
 * CPRA map legend plugin for py-maplibregl.
 * Contract: export createControl() -> IControl
 * Reads legend data from map.getGlobalState().legend on load and styledata events.
 */

function addDiscreteSteps(entry, steps) {
  const seen = new Set();
  for (const step of steps) {
    if (seen.has(step.label)) continue;
    seen.add(step.label);
    const row = document.createElement("div");
    row.className = "discrete-step";
    const swatch = document.createElement("div");
    swatch.className = "discrete-symbol";
    swatch.style.backgroundColor = step.color;
    const label = document.createElement("span");
    label.className = "discrete-label";
    label.innerText = step.label;
    row.append(swatch, label);
    entry.append(row);
  }
}

function addContinuousSteps(entry, steps) {
  if (steps.length < 2) return;
  const start = steps[0].value;
  const end = steps[steps.length - 1].value;
  const range = end - start || 1;
  const stops = steps.map((s) => ({
    ...s,
    pct: ((s.value - start) * 100) / range,
  }));

  const wrap = document.createElement("div");
  wrap.className = "continuous-steps";

  const bar = document.createElement("div");
  bar.className = "continuous-symbol";
  bar.style.backgroundImage = `linear-gradient(to right, ${stops
    .map((s) => `${s.color} ${s.pct}%`)
    .join(", ")})`;
  wrap.append(bar);

  const ticks = document.createElement("div");
  ticks.className = "continuous-ticks";
  for (const stop of stops) {
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.style.left = `${stop.pct}%`;
    const lbl = document.createElement("span");
    lbl.className = "label";
    lbl.style.left = `-${stop.pct}%`;
    lbl.innerText = String(stop.value);
    tick.append(lbl);
    ticks.append(tick);
  }
  wrap.append(ticks);
  entry.append(wrap);
}

function populateContainer(container, map) {
  container.innerHTML = "";
  const legend = (map.getGlobalState() || {}).legend || {};
  for (const spec of legend.entries || []) {
    const entry = document.createElement("div");
    entry.className = "entry";

    if (spec.title) {
      const h2 = document.createElement("h2");
      h2.className = "title";
      h2.innerText = spec.title;
      if (spec.subtitle) {
        const sub = document.createElement("span");
        sub.className = "subtitle";
        sub.innerText = `(${spec.subtitle})`;
        h2.innerText += " ";
        h2.append(sub);
      }
      entry.append(h2);
    }

    if (spec.type === "discrete") addDiscreteSteps(entry, spec.steps || []);
    else if (spec.type === "continuous")
      addContinuousSteps(entry, spec.steps || []);

    container.append(entry);
  }
}

export function createControl() {
  let _map = null;
  let _container = null;
  const update = () => {
    if (_map && _container) populateContainer(_container, _map);
  };


  return {
    getDefaultPosition: () => "bottom-left",
    onAdd(map) {
      _map = map;
      _container = document.createElement("div");
      _container.className =
        "maplibregl-ctrl maplibregl-ctrl-group cpra-map-legend";
      map.on("styledata", update);
      update();
      return _container;
    },
    onRemove() {
      if (_map) {
        _map.off("load", update);
        _map.off("styledata", update);
      }
      if (_container?.parentNode)
        _container.parentNode.removeChild(_container);
      _map = null;
      _container = null;
    },
  };
}
