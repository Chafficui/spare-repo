/* Benutzerverwaltung (nur Admin): Rollen/Mitgliedschaften vergeben, per E-Mail einladen. Kein Inline-Skript (CSP). */
(function () {
  'use strict';
  var h = CV.h, $ = CV.$;
  CV.init({ roles: ['admin'] });

  var ROLE_DE = { admin: 'Admin', hallenwart: 'Hallenwart', kampfgericht: 'Kampfgericht', trainer: 'Trainer', spieler: 'Spieler / Eltern', verband: 'Verband' };
  var ROLES = ['trainer', 'kampfgericht', 'hallenwart', 'spieler', 'verband', 'admin'];
  var teams = [], playersByTeam = {};

  function teamName(id) { var t = teams.filter(function (x) { return x.id === id; })[0]; return t ? t.name : id; }
  function mlabel(m) {
    var s = ROLE_DE[m.role] || m.role;
    if (m.team_id && m.role === 'trainer') s += ' · ' + teamName(m.team_id);
    if (m.player_id) s += ' · ' + (m.player_label || m.player_id);
    return s;
  }
  function fmtTime(ts) {
    if (!ts) return '–';
    var d = new Date(ts * 1000);
    return d.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' });
  }

  /** Editor für eine Liste von Mitgliedschaften: Chips + "Rolle hinzufügen"-Zeile. get() liefert die Liste. */
  function membershipEditor(initial) {
    var list = (initial || []).map(function (m) { return { role: m.role, team_id: m.team_id || null, player_id: m.player_id || null, player_label: m.player_label }; });
    var chips = h('div', { class: 'mlist', 'aria-label': 'Vergebene Rollen' });
    var role = h('select', { 'aria-label': 'Rolle' }, ROLES.map(function (r) { return h('option', { value: r, text: ROLE_DE[r] }); }));
    var team = h('select', { 'aria-label': 'Team' });
    var player = h('select', { 'aria-label': 'Spieler' });
    var addBtn = h('button', { type: 'button', class: 'chip', text: 'Hinzufügen' });
    var rowTeam = h('div', null, h('label', { text: 'Team' }), team);
    var rowPlayer = h('div', null, h('label', { text: 'Spieler' }), player);

    function paint() {
      CV.clear(chips);
      if (!list.length) chips.appendChild(h('span', { class: 'muted small', text: 'Keine Rolle – nur Fan-Sicht' }));
      list.forEach(function (m, i) {
        chips.appendChild(h('span', { class: 'mchip' }, mlabel(m),
          h('button', { type: 'button', 'aria-label': mlabel(m) + ' entfernen', title: 'Entfernen', text: '×', onclick: function () { list.splice(i, 1); paint(); } })));
      });
    }
    function fillTeams() {
      CV.clear(team);
      teams.forEach(function (t) { team.appendChild(h('option', { value: t.id, text: t.name })); });
    }
    function fillPlayers() {
      CV.clear(player);
      var tid = team.value;
      (playersByTeam[tid] || []).forEach(function (p) { player.appendChild(h('option', { value: p.id, text: '#' + p.number + ' ' + (p.name || p.id) })); });
    }
    function refresh() {
      var r = role.value;
      rowTeam.classList.toggle('hidden', r !== 'trainer' && r !== 'spieler');
      rowPlayer.classList.toggle('hidden', r !== 'spieler');
      if (r === 'spieler') ensurePlayers(team.value).then(fillPlayers);
    }
    role.addEventListener('change', refresh);
    team.addEventListener('change', function () { if (role.value === 'spieler') ensurePlayers(team.value).then(fillPlayers); });
    addBtn.addEventListener('click', function () {
      var m = { role: role.value, team_id: null, player_id: null };
      if (m.role === 'trainer') { m.team_id = team.value; if (!m.team_id) return CV.toast('Bitte ein Team wählen', true); }
      if (m.role === 'spieler') {
        m.player_id = player.value; m.team_id = team.value;
        if (!m.player_id) return CV.toast('Bitte einen Spieler wählen', true);
        var p = (playersByTeam[team.value] || []).filter(function (x) { return x.id === m.player_id; })[0];
        m.player_label = p ? '#' + p.number + ' ' + p.name : m.player_id;
      }
      if (list.some(function (x) { return x.role === m.role && x.team_id === m.team_id && x.player_id === m.player_id; })) return;
      list.push(m); paint();
    });
    fillTeams(); paint(); refresh();
    var el = h('div', { class: 'meditor' }, chips,
      h('div', { class: 'row' }, h('div', null, h('label', { text: 'Rolle hinzufügen' }), role), rowTeam, rowPlayer, h('div', { class: 'fit' }, addBtn)));
    return { el: el, get: function () { return list.map(function (m) { return { role: m.role, team_id: m.team_id, player_id: m.player_id }; }); } };
  }

  function ensurePlayers(tid) {
    if (!tid || playersByTeam[tid]) return Promise.resolve();
    return CV.api('/teams/' + encodeURIComponent(tid) + '/players').then(function (ps) { playersByTeam[tid] = ps; }).catch(function () { playersByTeam[tid] = []; });
  }

  // ---- Benutzerliste
  var editing = null;
  function memberChips(ms, implicit) {
    return h('div', { class: 'mlist' }, ms.map(function (m) { return h('span', { class: 'mchip' + (implicit ? ' implicit' : '') }, m.label || mlabel(m)); }));
  }
  function loadUsers() {
    return CV.api('/admin/users').then(function (d) {
      var tb = $('#users tbody');
      CV.clear(tb);
      if (!d.users.length) tb.appendChild(h('tr', null, h('td', { colspan: 5, class: 'empty', text: 'Noch hat sich niemand angemeldet.' })));
      d.users.forEach(function (u) {
        var assigned = u.memberships || [];
        tb.appendChild(h('tr', null,
          h('td', { class: 'l' }, h('b', { text: u.username || '–' }), h('div', { class: 'muted small', title: u.subject, text: u.subject.slice(0, 8) + '…' })),
          h('td', { class: 'l', text: u.email || '–' }),
          h('td', { class: 'nowrap', text: fmtTime(u.last_login) }),
          h('td', { class: 'l' }, u.effective.length ? memberChips(u.effective.map(function (m) { return { label: m.label }; }), !assigned.length && !u.bootstrap_admin) : null,
            u.bootstrap_admin ? h('span', { class: 'badge', title: 'über COURTVISION_ADMINS', text: 'Admin per Konfiguration' }) : null),
          h('td', { class: 'l' }, h('button', { type: 'button', class: 'chip', text: 'Rollen bearbeiten', onclick: function () { edit(u); } }))));
      });
      var ti = $('#invites tbody');
      CV.clear(ti);
      if (!d.invites.length) ti.appendChild(h('tr', null, h('td', { colspan: 3, class: 'empty', text: 'Keine offenen Einladungen.' })));
      d.invites.forEach(function (i) {
        ti.appendChild(h('tr', null, h('td', { class: 'l', text: i.email }), h('td', { class: 'l' }, memberChips(i.memberships.map(function (m) { return { label: mlabel(m) }; }))),
          h('td', { class: 'l' }, h('button', { type: 'button', class: 'chip danger', text: 'Zurücknehmen', onclick: function () {
            CV.tryApi('/admin/invites/' + encodeURIComponent(i.email), { method: 'DELETE' }).then(function () { loadUsers(); });
          } }))));
      });
      if (!d.email_trust) $('#invite-note').textContent += ' Achtung: Dieser Server vertraut der beim Login getippten Adresse nicht (COURTVISION_TRUST_TYPED_EMAIL=0) – Einladungen greifen nur, wenn Codai die Adresse mitliefert.';
    }).catch(function (e) { CV.toast(e.message, true); });
  }

  function edit(u) {
    var host = $('#editor-host');
    CV.clear(host);
    var ed = membershipEditor((u.memberships || []).map(function (m) {
      var out = { role: m.role, team_id: m.team_id, player_id: m.player_id };
      var eff = (u.effective || []).filter(function (x) { return x.role === m.role && x.team_id === m.team_id && x.player_id === m.player_id; })[0];
      if (m.player_id && eff) out.player_label = eff.label.split(' · ').slice(1).join(' · ');
      return out;
    }));
    host.appendChild(h('div', { class: 'meditor' }, h('h3', { text: 'Rollen von ' + (u.username || u.email || u.subject) }), ed.el,
      h('p', null,
        h('button', { type: 'button', class: 'primary', text: 'Speichern', onclick: function () {
          CV.api('/admin/users/' + encodeURIComponent(u.subject) + '/memberships', { method: 'PUT', json: { memberships: ed.get() } })
            .then(function () { CV.toast('Rollen gespeichert'); CV.clear(host); loadUsers(); })
            .catch(function (e) { CV.toast(e.message, true); });
        } }), ' ', h('button', { type: 'button', text: 'Abbrechen', onclick: function () { CV.clear(host); } }))));
    host.scrollIntoView({ block: 'nearest' });
  }

  // ---- Einladung
  var invEd = null;
  function resetInvite() { CV.clear($('#inv-editor')); invEd = membershipEditor([]); $('#inv-editor').appendChild(invEd.el); $('#inv-email').value = ''; }
  $('#inv-go').addEventListener('click', function () {
    var email = $('#inv-email').value.trim();
    if (!email) return CV.toast('Bitte eine E-Mail-Adresse eingeben', true);
    CV.api('/admin/invites', { json: { email: email, memberships: invEd.get() } }).then(function (r) {
      CV.toast(r.applied_to ? 'Benutzer kennt die Adresse schon – Rollen sofort vergeben' : 'Einladung gespeichert');
      resetInvite(); loadUsers();
    }).catch(function (e) { CV.toast(e.message, true); });
  });

  CV.api('/teams').then(function (ts) { teams = ts; return Promise.all(ts.map(function (t) { return ensurePlayers(t.id); })); })
    .then(function () { resetInvite(); return loadUsers(); })
    .catch(function (e) { CV.toast(e.message, true); });
})();
