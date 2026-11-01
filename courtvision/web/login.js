/* CourtVision-Anmeldung (Codai ID, eigene Oberfläche). Nur über die eigene API (/auth/*): der Browser spricht nie mit
 * Codai direkt und sieht weder Service-Token noch Flow-ID. Kein Inline-Skript (CSP). */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  var cfg = { code_length: 6, resend_wait_s: 30 };
  var email = '';
  var busy = false;
  var timer = null;

  // ---- Weiterleitung nach dem Login: nur relative *.html-Seiten dieser Anwendung
  function nextUrl() {
    var n = new URLSearchParams(location.search).get('next') || '';
    if (/^[A-Za-z0-9_\-]+\.html(\?[A-Za-z0-9_\-=&%.]*)?$/.test(n) && n.indexOf('login.html') !== 0) return n;
    return 'index.html';
  }

  // ---- Ansichten
  var STEPS = ['step-done', 'step-email', 'step-code', 'step-password'];
  function show(step, focusId) {
    STEPS.forEach(function (s) { $(s).classList.toggle('hidden', s !== step); });
    if (focusId) setTimeout(function () { var el = $(focusId); if (el) el.focus(); }, 30);
  }
  function say(kind, text) {
    var err = $('msg'), info = $('info');
    err.classList.add('hidden'); info.classList.add('hidden');
    if (!text) return;
    var box = kind === 'bad' ? err : info;
    box.textContent = text;
    box.classList.remove('hidden');
  }
  function setBusy(on, btnId) {
    busy = on;
    ['email-go', 'code-go', 'pw-go'].forEach(function (id) { $(id).disabled = on; });
    if (btnId) $(btnId).setAttribute('aria-busy', on ? 'true' : 'false');
  }

  // ---- Fehlertexte (Server liefert deutsche Meldungen; hier die Feinheiten)
  function errorText(e) {
    var d = (e && e.data && e.data.detail) || {};
    var code = d.code || '';
    var msg = d.message || (e && e.message) || 'Das hat nicht geklappt. Bitte versuche es noch einmal.';
    if (code === 'bad_code' && typeof d.attempts_left === 'number') {
      return msg + ' Noch ' + d.attempts_left + (d.attempts_left === 1 ? ' Versuch' : ' Versuche') + ' übrig.';
    }
    if ((code === 'rate_limited' || code === 'resend_wait') && d.retry_after) {
      var mins = Math.ceil(d.retry_after / 60);
      return code === 'resend_wait' ? msg : msg + ' (ca. ' + (mins <= 1 ? 'eine Minute' : mins + ' Minuten') + ')';
    }
    if (e && e.status === 403 && code === 'csrf') return msg;
    if (e && e.status === 0) return 'Keine Verbindung zum Server.';
    return msg;
  }
  function codeOf(e) { return (e && e.data && e.data.detail && e.data.detail.code) || ''; }
  function post(path, json) { return CV.api(path, { json: json || {}, noRedirect: true }); }

  // ---- Countdown für "Code erneut senden"
  function countdown(seconds) {
    clearInterval(timer);
    var left = Math.max(0, Math.ceil(seconds));
    var btn = $('resend');
    function paint() {
      btn.disabled = left > 0;
      btn.textContent = left > 0 ? 'Code erneut senden (' + left + ' s)' : 'Code erneut senden';
    }
    paint();
    if (left <= 0) return;
    timer = setInterval(function () { left -= 1; paint(); if (left <= 0) clearInterval(timer); }, 1000);
  }

  function toCodeStep(res, note) {
    email = res.email || email;
    $('sent-to').textContent = email;
    $('code').value = '';
    say(note ? 'info' : null, note || '');
    show('step-code', 'code');
    countdown(res.resend_after || cfg.resend_wait_s);
  }

  function done(me) {
    clearInterval(timer);
    location.replace(nextUrl());
  }

  // ---- Schritt 1: E-Mail -> Code anfordern
  function validEmail(v) { return /^[^@\s]+@[^@\s]+\.[^@\s.]{2,}$/.test(v); }
  function requestCode(addr) {
    say(null);
    if (!validEmail(addr)) { say('bad', 'Das sieht nicht wie eine E-Mail-Adresse aus.'); $('email').focus(); return Promise.resolve(); }
    setBusy(true, 'email-go');
    email = addr;
    return post('/auth/code/start', { email: addr, locale: 'de' }).then(function (res) {
      setBusy(false, 'email-go');
      toCodeStep(res);
    }).catch(function (e) {
      setBusy(false, 'email-go');
      say('bad', errorText(e));
      if (codeOf(e) === 'resend_wait' && e.data.detail.retry_after) { toCodeStep({ email: addr, resend_after: e.data.detail.retry_after }); say('bad', errorText(e)); }
    });
  }
  $('email-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    if (!busy) requestCode($('email').value.trim());
  });

  // ---- Schritt 2: Code prüfen (automatisch bei der 6. Ziffer)
  function submitCode() {
    if (busy) return;
    var digits = $('code').value.replace(/\D/g, '');
    if (digits.length !== cfg.code_length) { say('bad', 'Bitte gib den ' + cfg.code_length + '-stelligen Code ein.'); return; }
    say(null);
    setBusy(true, 'code-go');
    post('/auth/code/verify', { code: digits }).then(function (me) {
      setBusy(false, 'code-go');
      done(me);
    }).catch(function (e) {
      setBusy(false, 'code-go');
      var c = codeOf(e);
      say('bad', errorText(e));
      $('code').value = '';
      if (c === 'expired' || c === 'no_flow') {
        // Flow ist serverseitig beendet: neuen Code anfordern lassen
        $('resend').disabled = false; $('resend').textContent = 'Neuen Code anfordern';
        clearInterval(timer);
        $('resend').focus();
      } else {
        $('code').focus();
      }
    });
  }
  $('code').addEventListener('input', function () {
    var digits = $('code').value.replace(/\D/g, '').slice(0, cfg.code_length);   // "123 456" -> "123456"
    $('code').value = digits;
    if (digits.length === cfg.code_length) submitCode();
  });
  $('code').addEventListener('paste', function (ev) {
    var t = (ev.clipboardData || window.clipboardData);
    if (!t) return;
    ev.preventDefault();
    $('code').value = String(t.getData('text')).replace(/\D/g, '').slice(0, cfg.code_length);
    $('code').dispatchEvent(new Event('input'));
  });
  $('code-form').addEventListener('submit', function (ev) { ev.preventDefault(); submitCode(); });

  $('resend').addEventListener('click', function () {
    if (busy || $('resend').disabled) return;
    say(null);
    $('resend').disabled = true;
    post('/auth/code/resend').catch(function (e) {
      if (codeOf(e) === 'no_flow' && e.status === 410) return post('/auth/code/start', { email: email, locale: 'de' });
      throw e;
    }).then(function (res) {
      toCodeStep(res, 'Wir haben dir einen neuen Code geschickt. Nur der neueste Code gilt.');
    }).catch(function (e) {
      say('bad', errorText(e));
      var ra = e && e.data && e.data.detail && e.data.detail.retry_after;
      if (codeOf(e) === 'resend_wait' && ra) countdown(ra); else $('resend').disabled = false;
    });
  });
  $('other-email').addEventListener('click', function () {
    clearInterval(timer);
    post('/auth/code/cancel').catch(function () { /* egal */ });
    say(null);
    $('email').value = email;
    show('step-email', 'email');
  });

  // ---- Alternative: Passwort
  $('to-password').addEventListener('click', function () {
    say(null);
    $('pw-email').value = $('email').value.trim();
    show('step-password', $('pw-email').value ? 'pw' : 'pw-email');
  });
  $('to-code').addEventListener('click', function () {
    say(null);
    $('email').value = $('pw-email').value.trim();
    show('step-email', 'email');
  });
  $('pw-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    if (busy) return;
    var addr = $('pw-email').value.trim(), pw = $('pw').value;
    if (!validEmail(addr) || !pw) { say('bad', 'Bitte gib E-Mail-Adresse und Passwort ein.'); return; }
    say(null);
    setBusy(true, 'pw-go');
    post('/auth/password', { email: addr, password: pw, locale: 'de' }).then(function (res) {
      setBusy(false, 'pw-go');
      $('pw').value = '';
      if (res.step_up) {
        toCodeStep(res, 'Neues Gerät: Zur Sicherheit haben wir dir zusätzlich einen Code per E-Mail geschickt.');
      } else {
        done(res);
      }
    }).catch(function (e) {
      setBusy(false, 'pw-go');
      $('pw').value = '';
      say('bad', errorText(e));
      $('pw').focus();
    });
  });

  // ---- Start: Konfiguration + ggf. schon angemeldet
  function demoBox(accounts) {
    if (!accounts || !accounts.length) return;
    var chips = $('demo-chips');
    accounts.forEach(function (a) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'chip'; b.textContent = a.label; b.title = a.email;
      b.addEventListener('click', function () {
        $('email').value = a.email; $('pw-email').value = a.email;
        show('step-email', 'email');
        requestCode(a.email);
      });
      chips.appendChild(b);
    });
    $('demo').classList.remove('hidden');
  }

  if (CV.mode() === 'dev') {
    say('info', 'Dieser Server läuft im Entwicklungsmodus (COURTVISION_AUTH_MODE=dev): Rollen werden oben auf den Seiten gewählt, eine Anmeldung ist nicht nötig.');
  }
  CV.api('/auth/config', { noRedirect: true }).then(function (c) {
    cfg = c;
    $('forgot').href = c.password_reset_url;
    $('account').href = c.account_url;
    demoBox(c.demo_accounts);
    if (!c.configured && c.mode === 'codai') say('bad', 'Die Anmeldung ist auf diesem Server noch nicht eingerichtet (Service-Token für Codai ID fehlt).');
  }).catch(function () { say('bad', 'Der Server ist nicht erreichbar.'); });

  var me = CV.me();
  if (me && me.authenticated && CV.mode() === 'codai') {
    $('done-who').textContent = (me.username || me.email || 'Konto') + ' · ' + (me.label || me.role);
    $('done-next').href = nextUrl();
    $('done-logout').addEventListener('click', function () { CV.logout(); });
    show('step-done');
  } else {
    show('step-email', 'email');
  }
  var pre = new URLSearchParams(location.search).get('email');
  if (pre) $('email').value = pre;
})();
