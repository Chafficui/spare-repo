/* CourtVision Web-UI: gemeinsame Helfer (fetch, WebSocket, Rollenwahl, SVG). Kein Build, keine externen Abhängigkeiten.
 *
 * WICHTIG: Die Rolle (X-Role) ist ein Platzhalter für echtes Login -- siehe courtvision/api/auth.py. */
(function () {
  'use strict';

  var ROLES = [
    ['fan', 'Fan (öffentlich)'], ['spieler', 'Spieler / Eltern'], ['trainer', 'Trainer'],
    ['kampfgericht', 'Kampfgericht'], ['hallenwart', 'Hallenwart'], ['verband', 'Verband'], ['admin', 'Admin']
  ];
  var NAV = [
    ['index.html', 'Start'], ['setup.html', 'Setup'], ['console.html', 'Konsole'], ['dashboard.html', 'Trainer'],
    ['game.html', 'Spielseite'], ['player.html', 'Spieler'], ['verband.html', 'Verband']
  ];

  // ---- Speicher (localStorage kann fehlen/werfen: dann nur im Arbeitsspeicher)
  var mem = {};
  function lsGet(k, d) {
    try { var v = window.localStorage.getItem(k); if (v !== null) return v; } catch (e) { /* ignorieren */ }
    return Object.prototype.hasOwnProperty.call(mem, k) ? mem[k] : d;
  }
  function lsSet(k, v) {
    mem[k] = v;
    try { window.localStorage.setItem(k, v); } catch (e) { /* ignorieren */ }
  }

  function role() { return lsGet('cv.role', 'fan'); }
  function teamId() { return lsGet('cv.team', ''); }
  function playerId() { return lsGet('cv.player', ''); }

  // ---- DOM
  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else if (k === 'value') el.value = v;
        else el.setAttribute(k, v === true ? '' : v);
      });
    }
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function append(el, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { append(el, x); }); return; }
    el.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
  function $(sel, root) { return (root || document).querySelector(sel); }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function s(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined) el.setAttribute(k, attrs[k]); });
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }

  // ---- Meldungen
  function toast(msg, bad) {
    var box = $('#toast');
    if (!box) { box = h('div', { id: 'toast', role: 'status', 'aria-live': 'polite' }); document.body.appendChild(box); }
    var t = h('div', { class: bad ? 'bad' : '', text: msg });
    box.appendChild(t);
    setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, bad ? 6000 : 3000);
  }

  // ---- API
  function ApiError(status, message, data) { this.status = status; this.message = message; this.data = data; }
  ApiError.prototype = Object.create(Error.prototype);

  function detailText(d) {
    if (!d) return '';
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map(function (x) { return (x.loc ? x.loc.slice(1).join('.') + ': ' : '') + (x.msg || JSON.stringify(x)); }).join('; ');
    if (d.message) return d.message + (d.errors ? ' (' + d.errors.map(function (e) { return 'Zeile ' + e.line + ': ' + e.message; }).join('; ') + ')' : '');
    return JSON.stringify(d);
  }

  function headers(extra) {
    var hd = { 'X-Role': role() };
    if (teamId()) hd['X-Team-Id'] = teamId();
    if (playerId()) hd['X-Player-Id'] = playerId();
    return Object.assign(hd, extra || {});
  }

  /** api(path, {method, json, body, contentType, raw}) -> JSON | Text; wirft ApiError mit deutscher Meldung. */
  function api(path, opts) {
    opts = opts || {};
    var hd = headers(opts.headers);
    var body = opts.body;
    if (opts.json !== undefined) { body = JSON.stringify(opts.json); hd['Content-Type'] = 'application/json'; }
    if (opts.contentType) hd['Content-Type'] = opts.contentType;
    return fetch(path, { method: opts.method || (body !== undefined ? 'POST' : 'GET'), headers: hd, body: body }).then(function (r) {
      var ct = r.headers.get('content-type') || '';
      var p = ct.indexOf('json') >= 0 ? r.json() : r.text();
      return p.then(function (data) {
        if (!r.ok) {
          var msg = detailText(data && data.detail !== undefined ? data.detail : data) || ('Fehler ' + r.status);
          throw new ApiError(r.status, msg, data);
        }
        return opts.raw ? { data: data, response: r } : data;
      });
    });
  }
  /** Wie api(), zeigt Fehler als Toast und liefert dann undefined. */
  function tryApi(path, opts) {
    return api(path, opts).catch(function (e) { toast(e.message || String(e), true); });
  }

  // ---- WebSocket mit Wiederverbindung; Fallback-Polling in live()
  function connect(gameId, o) {
    o = o || {};
    var ws, closed = false, last = o.after || 0, timer = null, isOpen = false;
    function open() {
      var q = new URLSearchParams({ role: role(), after_seq: String(last) });
      if (teamId()) q.set('team_id', teamId());
      if (playerId()) q.set('player_id', playerId());
      var proto = location.protocol === 'https:' ? 'wss' : 'ws';
      try { ws = new WebSocket(proto + '://' + location.host + '/ws/games/' + encodeURIComponent(gameId) + '?' + q); }
      catch (e) { schedule(); return; }
      ws.onopen = function () { isOpen = true; if (o.onStatus) o.onStatus('open'); };
      ws.onmessage = function (m) {
        var d; try { d = JSON.parse(m.data); } catch (e) { return; }
        if (d.kind === 'event') { last = Math.max(last, d.event.seq || 0); if (o.onEvent) o.onEvent(d); }
        else if (d.kind === 'synced' && o.onSynced) o.onSynced(d);
      };
      ws.onclose = function () { isOpen = false; if (o.onStatus) o.onStatus('closed'); if (!closed) schedule(); };
      ws.onerror = function () { try { ws.close(); } catch (e) { /* ignorieren */ } };
    }
    function schedule() { clearTimeout(timer); timer = setTimeout(open, 3000); }
    open();
    return { close: function () { closed = true; clearTimeout(timer); try { ws.close(); } catch (e) { /* ignorieren */ } }, isOpen: function () { return isOpen; } };
  }

  /** live(gameId, refresh): ruft refresh() bei jedem WebSocket-Event (entprellt) und als Polling-Fallback auf. */
  function live(gameId, refresh, o) {
    var pending = null, ws;
    function soon() { clearTimeout(pending); pending = setTimeout(refresh, 150); }
    ws = connect(gameId, Object.assign({
      onEvent: function (d) { if (o && o.onEvent) o.onEvent(d); soon(); },
      onSynced: function () { soon(); },
      onStatus: function (st) { if (o && o.onStatus) o.onStatus(st); }
    }, {}));
    var ticks = 0;
    var iv = setInterval(function () { ticks++; if (!ws.isOpen() || ticks % 6 === 0) refresh(); }, 5000);
    refresh();
    return { close: function () { ws.close(); clearInterval(iv); }, ws: ws };
  }

  // ---- Format
  function clock(sec) {
    if (sec === null || sec === undefined) return '--:--';
    var t = Math.max(0, Math.floor(sec));
    return String(Math.floor(t / 60)).padStart(2, '0') + ':' + String(t % 60).padStart(2, '0');
  }
  function num(x, d) { return (x === null || x === undefined || isNaN(x)) ? '–' : Number(x).toFixed(d === undefined ? 0 : d).replace('.', ','); }
  function pct(x) { return (x === null || x === undefined) ? '–' : (100 * x).toFixed(0) + ' %'; }
  function fmtTs(sec) {
    var m = Math.floor(sec / 60), r = sec - m * 60;
    return String(m).padStart(2, '0') + ':' + r.toFixed(1).padStart(4, '0');
  }
  var STATUS_DE = { scheduled: 'geplant', live: 'LIVE', processing: 'Verarbeitung', review: 'Korrektur', final: 'final' };

  // ---- Kopfzeile + Rolle
  function init(opts) {
    opts = opts || {};
    var top = $('#cv-top') || document.body.insertBefore(h('div', { id: 'cv-top' }), document.body.firstChild);
    clear(top);
    var page = location.pathname.split('/').pop() || 'index.html';
    var sel = h('select', { id: 'cv-role', 'aria-label': 'Rolle', onchange: function () {
      lsSet('cv.role', sel.value); location.reload();
    } }, ROLES.map(function (r) { return h('option', { value: r[0], text: r[1] }); }));
    sel.value = role();
    var extra = null;
    if (role() === 'trainer') {
      extra = h('select', { id: 'cv-team', 'aria-label': 'Team', onchange: function () { lsSet('cv.team', extra.value); location.reload(); } },
        h('option', { value: '', text: 'Team wählen' }));
      api('/teams').then(function (ts) {
        ts.forEach(function (t) { extra.appendChild(h('option', { value: t.id, text: t.name })); });
        extra.value = teamId();
      }).catch(function () { /* ignorieren */ });
    } else if (role() === 'spieler') {
      extra = h('input', { id: 'cv-player', placeholder: 'Spieler-ID', 'aria-label': 'Spieler-ID', value: playerId(),
        onchange: function () { lsSet('cv.player', extra.value.trim()); location.reload(); } });
    }
    top.className = 'topbar';
    top.appendChild(h('a', { class: 'brand', href: 'index.html' }, 'Court', h('b', { text: 'Vision' })));
    top.appendChild(h('nav', { 'aria-label': 'Hauptnavigation' }, NAV.map(function (n) {
      return h('a', { href: n[0], 'aria-current': n[0] === page ? 'page' : null, text: n[1] });
    })));
    top.appendChild(h('div', { class: 'rolebox', title: 'Platzhalter für Login (X-Role-Header)' }, 'Rolle', sel, extra));
    if (opts.roles && opts.roles.indexOf(role()) < 0) {
      var main = $('main');
      if (main) main.insertBefore(h('div', { class: 'banner' },
        'Diese Seite ist für ', opts.roles.join(' / '), ' gedacht (aktuell: ', role(), '). ',
        h('button', { class: 'chip', onclick: function () { lsSet('cv.role', opts.roles[0]); location.reload(); }, text: 'Zu „' + opts.roles[0] + '“ wechseln' })), main.firstChild);
    }
  }

  // ---- Spielauswahl
  function qs(name) { return new URLSearchParams(location.search).get(name); }
  function gamePicker(selectEl, opts) {
    opts = opts || {};
    return api('/games' + (opts.query || '')).then(function (games) {
      clear(selectEl);
      if (!games.length) selectEl.appendChild(h('option', { value: '', text: 'Keine Spiele vorhanden' }));
      games.forEach(function (g) {
        selectEl.appendChild(h('option', { value: g.id,
          text: g.home_name + ' – ' + g.away_name + ' (' + (STATUS_DE[g.status] || g.status) + (g.start_time ? ', ' + g.start_time.slice(0, 10) : '') + ')' }));
      });
      var want = qs('game') || lsGet('cv.game', '');
      if (want && games.some(function (g) { return g.id === want; })) selectEl.value = want;
      selectEl.addEventListener('change', function () {
        lsSet('cv.game', selectEl.value);
        var u = new URL(location.href); u.searchParams.set('game', selectEl.value); history.replaceState(null, '', u);
        if (opts.onChange) opts.onChange(selectEl.value);
      });
      if (selectEl.value && opts.onChange) opts.onChange(selectEl.value);
      return games;
    }).catch(function (e) { toast(e.message, true); return []; });
  }

  // ---- Spielfeld (SVG)
  function courtSvg(container, geo) {
    var svg = s('svg', { class: 'court', viewBox: '-14.8 -8.3 29.6 16.6', role: 'img', 'aria-label': 'Spielfeld' });
    function pl(pts, cls) {
      return s('polyline', { class: 'ln' + (cls ? ' ' + cls : ''), points: pts.map(function (p) { return p[0] + ',' + (-p[1]); }).join(' ') });
    }
    Object.keys(geo.lines).forEach(function (k) { svg.appendChild(pl(geo.lines[k], k.indexOf('dashed') >= 0 ? 'dash' : '')); });
    ['left', 'right'].forEach(function (k) { svg.appendChild(pl(geo.three_point[k])); });
    var layer = s('g');
    svg.appendChild(layer);
    clear(container).appendChild(svg);
    return { svg: svg, layer: layer };
  }

  // ---- Diagramme
  function sparkline(values, o) {
    o = o || {};
    var w = o.w || 260, hgt = o.h || 64, pad = 8, lo = o.min === undefined ? 1 : o.min, hi = o.max === undefined ? 10 : o.max;
    var svg = s('svg', { class: 'chart', viewBox: '0 0 ' + w + ' ' + hgt, width: '100%', role: 'img', 'aria-label': 'Verlauf der Spielnoten' });
    if (!values || !values.length) { svg.appendChild(s('text', { x: 4, y: hgt / 2 }, 'noch keine Noten')); return svg; }
    var step = values.length > 1 ? (w - 2 * pad) / (values.length - 1) : 0;
    var pts = values.map(function (v, i) { return [values.length > 1 ? pad + i * step : w / 2, hgt - pad - (v - lo) / (hi - lo) * (hgt - 2 * pad)]; });
    svg.appendChild(s('line', { x1: pad, x2: w - pad, y1: hgt - pad - (6 - lo) / (hi - lo) * (hgt - 2 * pad), y2: hgt - pad - (6 - lo) / (hi - lo) * (hgt - 2 * pad), class: 'grid', 'stroke-dasharray': '3 3' }));
    svg.appendChild(s('polyline', { points: pts.map(function (p) { return p.join(','); }).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }));
    pts.forEach(function (p, i) {
      svg.appendChild(s('circle', { cx: p[0], cy: p[1], r: i === pts.length - 1 ? 4.5 : 3, fill: 'var(--accent)' }, s('title', null, 'Note ' + values[i].toFixed(1))));
    });
    return svg;
  }

  var RADAR_LABELS = { scoring: 'Scoring', efficiency: 'Effizienz', rebounding: 'Rebounds', playmaking: 'Playmaking', defense: 'Defense', ball_security: 'Ballsicherheit' };
  function radar(values, o) {
    o = o || {};
    var keys = Object.keys(RADAR_LABELS), size = o.size || 300, c = size / 2, r = size * 0.33, n = keys.length;
    var svg = s('svg', { class: 'chart', viewBox: '0 0 ' + size + ' ' + size, width: '100%', style: 'max-width:' + size + 'px', role: 'img', 'aria-label': 'Radar-Profil' });
    function pt(i, f) { var a = -Math.PI / 2 + i * 2 * Math.PI / n; return [c + Math.cos(a) * r * f, c + Math.sin(a) * r * f]; }
    [0.25, 0.5, 0.75, 1].forEach(function (f) {
      svg.appendChild(s('polygon', { class: 'grid', fill: 'none', points: keys.map(function (_, i) { return pt(i, f).join(','); }).join(' ') }));
    });
    keys.forEach(function (k, i) {
      var p = pt(i, 1), q = pt(i, 1.2);
      svg.appendChild(s('line', { class: 'grid', x1: c, y1: c, x2: p[0], y2: p[1] }));
      svg.appendChild(s('text', { x: q[0], y: q[1], 'text-anchor': q[0] < c - 4 ? 'end' : (q[0] > c + 4 ? 'start' : 'middle'), 'dominant-baseline': 'middle' }, RADAR_LABELS[k]));
    });
    svg.appendChild(s('polygon', { class: 'area', points: keys.map(function (k, i) { return pt(i, Math.max(0.02, (values[k] || 0) / 100)).join(','); }).join(' ') }, s('title', null, keys.map(function (k) { return RADAR_LABELS[k] + ' ' + Math.round(values[k] || 0); }).join(', '))));
    return svg;
  }

  // ---- Tabellen
  /** Sortierbare Tabelle. cols: [{key, label, fmt(v,row), cls, title}] */
  function table(cols, rows, o) {
    o = o || {};
    var state = { key: o.sortKey || null, dir: o.sortDir || -1 };
    var thead = h('thead'), tbody = h('tbody');
    var t = h('table', null, thead, tbody);
    function draw() {
      var pinned = rows.filter(function (r) { return r._cls === 'total'; });
      var data = rows.filter(function (r) { return r._cls !== 'total'; });
      if (state.key) data.sort(function (a, b) {
        var x = a[state.key], y = b[state.key];
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (typeof x === 'string' ? x.localeCompare(y) : x - y) * state.dir;
      });
      clear(thead).appendChild(h('tr', null, cols.map(function (c) {
        return h('th', { class: (c.sort === false ? '' : 'sortable ') + (c.l ? 'l' : ''), title: c.title, 'aria-sort': state.key === c.key ? (state.dir > 0 ? 'ascending' : 'descending') : null,
          onclick: c.sort === false ? null : function () { state.dir = state.key === c.key ? -state.dir : -1; state.key = c.key; draw(); } },
          c.label + (state.key === c.key ? (state.dir > 0 ? ' ▲' : ' ▼') : ''));
      })));
      clear(tbody);
      var all = data.concat(pinned);
      all.forEach(function (r) {
        tbody.appendChild(h('tr', { class: r._cls || '' }, cols.map(function (c) {
          var v = r[c.key];
          return h('td', { class: c.l ? 'l' : '' }, c.fmt ? c.fmt(v, r) : (v === null || v === undefined ? '–' : v));
        })));
      });
      if (!all.length) tbody.appendChild(h('tr', null, h('td', { colspan: cols.length, class: 'empty', text: o.empty || 'Keine Daten' })));
    }
    draw();
    var wrap = h('div', { class: 'tablewrap' }, t);
    wrap.redraw = function (newRows) { rows = newRows; draw(); };
    return wrap;
  }

  window.CV = {
    ROLES: ROLES, h: h, s: s, $: $, clear: clear, toast: toast, api: api, tryApi: tryApi, connect: connect, live: live,
    role: role, teamId: teamId, playerId: playerId, lsGet: lsGet, lsSet: lsSet, init: init, qs: qs,
    gamePicker: gamePicker, courtSvg: courtSvg, sparkline: sparkline, radar: radar, table: table,
    clock: clock, num: num, pct: pct, fmtTs: fmtTs, STATUS_DE: STATUS_DE, headers: headers, ApiError: ApiError
  };
})();
