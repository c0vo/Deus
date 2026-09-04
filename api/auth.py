"""
Access control for the web dashboard.

Two independent gates, both enforced by a single ASGI middleware:

  1. Source-IP allowlist (settings.trusted_networks). The app binds 0.0.0.0
     because the Android Tailscale client delivers inbound packets on a tun
     interface and a loopback-only bind would make the phone unreachable. That
     bind also exposes port 8000 to whatever WiFi the phone happens to have
     joined, so the peer address is checked against the tailnet CGNAT range
     plus loopback and everything else is refused.
  2. Passphrase + HMAC-signed session cookie (settings.dashboard_passphrase).
     Tailscale authenticates devices, not people; this is what stops anyone
     who is on the tailnet but should not be in the app.

Kept as raw ASGI rather than BaseHTTPMiddleware for the same reason as
CacheControlMiddleware in api/middleware.py: that base class wraps every
response in a streaming pump, which breaks FileResponse and stalls the
long-lived SSE streams behind /api/chat/stream, /api/predict/*/stream,
/api/thesis/stream and /api/brain/stream.

Cookies rather than an Authorization header, because the frontend's four SSE
streams are hand-rolled over fetch() and the dashboard is same-origin in the
static-export deployment — a cookie rides along on every one of them for free.

No third-party crypto dependency: everything here is stdlib hmac/hashlib.
Adding passlib or itsdangerous would mean compiling on ARM under Termux.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import time
from functools import lru_cache
from html import escape
from urllib.parse import parse_qs, quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from config.logging_config import get_logger
from config.settings import settings

log = get_logger(__name__)

SESSION_COOKIE = "deus_session"

# Signing-key domain separator. Bump the suffix to invalidate every outstanding
# session across a deploy without anyone having to change their passphrase.
_KEY_CONTEXT = b"deus-session-v1"

# Paths that skip the cookie gate. The IP gate still applies to all of them.
# /login and the auth endpoints must be reachable to log in at all; /healthz
# exists so the Termux wake-lock check has something to poll that leaks nothing
# (unlike /api/status, which reports pipeline internals).
PUBLIC_PATHS = frozenset(
    {
        "/login",
        "/api/auth/login",
        "/api/auth/logout",
        "/healthz",
        "/favicon.ico",
        "/manifest.webmanifest",
    }
)

# Brute-force throttle for the login endpoint. A single shared passphrase is
# guessable at machine speed without one.
_LOGIN_MAX_FAILURES = 5
_LOGIN_WINDOW_SECONDS = 900
_failed_logins: dict[str, list[float]] = {}

# Denied source addresses already logged, so a scanner on the local WiFi cannot
# flood the log. Bounded because the set is keyed by attacker-controlled input.
_denied_logged: set[str] = set()
_DENIED_LOG_CAP = 256


# ── Session cookie ────────────────────────────────────────────────────────


def _session_key(passphrase: str) -> bytes:
    """Derive the cookie signing key from the passphrase.

    Deriving rather than storing a separate secret gives revocation for free:
    changing DASHBOARD_PASSPHRASE changes the key, which invalidates every
    outstanding cookie immediately. With a single shared passphrase that is the
    only way to cut off access for one of the people it was shared with.
    """
    return hmac.new(passphrase.encode("utf-8"), _KEY_CONTEXT, hashlib.sha256).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def issue_session_cookie(passphrase: str, ttl_seconds: int, now: float | None = None) -> str:
    """Mint a signed cookie value carrying nothing but its own expiry."""
    expires_at = int((time.time() if now is None else now) + ttl_seconds)
    payload = str(expires_at).encode("ascii")
    signature = hmac.new(_session_key(passphrase), payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(signature)}"


def verify_session_cookie(value: str, passphrase: str, now: float | None = None) -> bool:
    """Constant-time verify, then expiry check. Any malformed input is False."""
    try:
        payload_b64, signature_b64 = value.split(".", 1)
        payload = _unb64(payload_b64)
        signature = _unb64(signature_b64)
    except (ValueError, binascii.Error):
        return False

    expected = hmac.new(_session_key(passphrase), payload, hashlib.sha256).digest()
    # Signature first: never parse a payload that has not been authenticated.
    if not hmac.compare_digest(signature, expected):
        return False

    try:
        expires_at = int(payload.decode("ascii"))
    except (ValueError, UnicodeDecodeError):
        return False

    return (time.time() if now is None else now) < expires_at


# ── Source-IP allowlist ───────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _compiled_networks(cidrs: tuple[str, ...]) -> tuple:
    """Parse CIDR strings once rather than on every request.

    A malformed entry is dropped with a warning instead of raising: one typo in
    .env should narrow the allowlist, never stop the server from booting.
    """
    compiled = []
    for cidr in cidrs:
        try:
            compiled.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            log.warning("auth.bad_trusted_network", cidr=cidr)
    return tuple(compiled)


def is_trusted_client(host: str | None, networks: list[str]) -> bool:
    """Whether a peer address falls inside one of the allowed CIDRs.

    An empty `networks` list disables the check. A missing or unparseable host
    fails closed — the alternative is that a transport which does not report a
    peer silently bypasses the gate.
    """
    if not networks:
        return True
    if not host:
        return False

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False

    # ::ffff:127.0.0.1 and friends: compare as the IPv4 address they represent,
    # otherwise a v4-mapped peer never matches a v4 CIDR.
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    # ip_network.__contains__ returns False across address families rather than
    # raising, so a mixed v4/v6 allowlist is safe to iterate.
    return any(addr in net for net in _compiled_networks(tuple(networks)))


# ── Login throttle ────────────────────────────────────────────────────────


def _prune_failures(now: float) -> None:
    for key in list(_failed_logins):
        recent = [t for t in _failed_logins[key] if now - t < _LOGIN_WINDOW_SECONDS]
        if recent:
            _failed_logins[key] = recent
        else:
            del _failed_logins[key]


def _is_throttled(client: str) -> bool:
    now = time.time()
    _prune_failures(now)
    return len(_failed_logins.get(client, [])) >= _LOGIN_MAX_FAILURES


def _record_failure(client: str) -> None:
    now = time.time()
    _prune_failures(now)
    _failed_logins.setdefault(client, []).append(now)


def _clear_failures(client: str) -> None:
    _failed_logins.pop(client, None)


def reset_login_throttle() -> None:
    """Clear all recorded failures. Exists so tests do not leak state."""
    _failed_logins.clear()


# ── Middleware ────────────────────────────────────────────────────────────


def _safe_next(raw: str | None) -> str:
    """Constrain the post-login redirect to a path on this host.

    Anything protocol-relative ("//evil.com") or backslash-escaped would turn
    the login form into an open redirect.
    """
    if not raw or not raw.startswith("/") or raw.startswith("//") or "\\" in raw:
        return "/"
    return raw


class AccessControlMiddleware:
    """Refuse untrusted peers, then unauthenticated requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # ── Gate 1: source address ──
        # scope["client"] is the real peer because nothing proxies this app. If
        # a reverse proxy is ever put in front, this MUST start reading
        # X-Forwarded-For instead — otherwise every peer looks like 127.0.0.1
        # and the gate silently passes everyone.
        networks = settings.trusted_network_list
        client = scope.get("client")
        host = client[0] if client else None
        if not is_trusted_client(host, networks):
            marker = host or "<unknown>"
            if marker not in _denied_logged and len(_denied_logged) < _DENIED_LOG_CAP:
                _denied_logged.add(marker)
                log.warning("auth.untrusted_source", client=marker, path=path)
            await self._reject(scope, receive, send, 403, "Forbidden")
            return

        # ── Gate 2: session cookie ──
        if not settings.auth_enabled:
            await self.app(scope, receive, send)
            return

        # Preflight carries no cookie and no data; gating it breaks the dev
        # server's cross-origin calls without protecting anything.
        if scope.get("method") == "OPTIONS" or path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        cookie = _read_cookie(scope, SESSION_COOKIE)
        if cookie and verify_session_cookie(cookie, settings.dashboard_passphrase):
            await self.app(scope, receive, send)
            return

        await self._reject(scope, receive, send, 401, "Authentication required")

    async def _reject(
        self, scope: Scope, receive: Receive, send: Send, status: int, detail: str
    ) -> None:
        """Bounce browsers to the login form, answer fetch callers with JSON.

        These responses are generated above CacheControlMiddleware in the
        stack, so they never pass through it — Cache-Control is set here or a
        401 could be cached and pinned in place.
        """
        path = scope.get("path", "")
        accept = Headers(scope=scope).get("accept", "")
        wants_html = "text/html" in accept and not path.startswith("/api/")

        if status == 401 and wants_html:
            target = f"/login?next={_quote_path(path)}"
            response: Response = RedirectResponse(target, status_code=302)
        else:
            response = JSONResponse({"detail": detail}, status_code=status)

        response.headers["Cache-Control"] = "no-store"
        await response(scope, receive, send)


def _read_cookie(scope: Scope, name: str) -> str | None:
    """Pull one cookie out of the raw header without building a Request."""
    raw = Headers(scope=scope).get("cookie")
    if not raw:
        return None
    for part in raw.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == name:
            return value.strip('"')
    return None


def _quote_path(path: str) -> str:
    return quote(path, safe="/")


# ── Routes ────────────────────────────────────────────────────────────────

auth_router = APIRouter()


@auth_router.get("/healthz")
async def healthz() -> JSONResponse:
    """Unauthenticated liveness probe.

    Deliberately returns nothing but a constant: the Termux wake-lock check in
    the README needs a pollable endpoint, and /api/status would have leaked
    pipeline internals to anyone who could reach the port.
    """
    return JSONResponse({"ok": True})


@auth_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if not settings.auth_enabled:
        return HTMLResponse(_redirect_body("/"), status_code=200)

    next_path = _safe_next(request.query_params.get("next"))
    error = request.query_params.get("e")
    message = ""
    if error == "1":
        message = "Incorrect passphrase."
    elif error == "2":
        message = "Too many attempts. Wait 15 minutes and try again."

    html = _login_html(next_path=next_path, message=message)
    response = HTMLResponse(html, status_code=401 if error else 200)
    response.headers["Cache-Control"] = "no-store"
    return response


@auth_router.post("/api/auth/login")
async def login(request: Request) -> Response:
    """Accepts either a form post (from the login page) or JSON (from curl)."""
    content_type = request.headers.get("content-type", "")
    wants_json = content_type.startswith("application/json")

    if wants_json:
        try:
            body = await request.json()
        except Exception:
            body = {}
        passphrase = str(body.get("passphrase", ""))
        next_path = _safe_next(body.get("next"))
    else:
        # parse_qs rather than request.form(): Starlette's form parsing pulls in
        # python-multipart, and a two-field urlencoded body does not justify a
        # new dependency on the Termux install.
        raw = (await request.body()).decode("utf-8", errors="replace")
        fields = parse_qs(raw, keep_blank_values=True)
        passphrase = fields.get("passphrase", [""])[0]
        next_path = _safe_next(fields.get("next", ["/"])[0])

    if not settings.auth_enabled:
        return _login_result(wants_json, next_path, ok=True, cookie=None)

    client = request.client.host if request.client else "<unknown>"

    if _is_throttled(client):
        log.warning("auth.login_throttled", client=client)
        if wants_json:
            return JSONResponse({"detail": "Too many attempts"}, status_code=429)
        return RedirectResponse(f"/login?e=2&next={_quote_path(next_path)}", status_code=303)

    # Encode before comparing: compare_digest rejects str arguments that are not
    # ASCII-only, and a passphrase with an accent in it would raise instead of
    # returning False.
    supplied = passphrase.encode("utf-8")
    expected = settings.dashboard_passphrase.encode("utf-8")
    if not hmac.compare_digest(supplied, expected):
        _record_failure(client)
        log.warning("auth.login_failed", client=client)
        if wants_json:
            return JSONResponse({"detail": "Invalid passphrase"}, status_code=401)
        return RedirectResponse(f"/login?e=1&next={_quote_path(next_path)}", status_code=303)

    _clear_failures(client)
    log.info("auth.login_ok", client=client)
    ttl = settings.dashboard_session_days * 86400
    cookie = issue_session_cookie(settings.dashboard_passphrase, ttl)
    return _login_result(wants_json, next_path, ok=True, cookie=cookie, max_age=ttl)


@auth_router.post("/api/auth/logout")
async def logout() -> Response:
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


def _login_result(
    wants_json: bool,
    next_path: str,
    ok: bool,
    cookie: str | None,
    max_age: int = 0,
) -> Response:
    if wants_json:
        response: Response = JSONResponse({"ok": ok})
    else:
        response = RedirectResponse(next_path, status_code=303)

    if cookie is not None:
        # No Secure flag: the tailnet is plain HTTP, and Secure would stop the
        # browser sending this cookie at all. WireGuard already encrypts the
        # whole hop, so the flag would buy nothing and break everything.
        # Do not "fix" this without also terminating TLS.
        response.set_cookie(
            SESSION_COOKIE,
            cookie,
            max_age=max_age,
            httponly=True,
            samesite="lax",
            path="/",
        )
    response.headers["Cache-Control"] = "no-store"
    return response


def _redirect_body(target: str) -> str:
    return (
        f'<!doctype html><meta http-equiv="refresh" content="0;url={escape(target)}">'
        f'<a href="{escape(target)}">Continue</a>'
    )


def _login_html(next_path: str, message: str) -> str:
    """Self-contained login form.

    No /_next/* assets and no JS framework on purpose: those paths stay behind
    the gate, so anything the login page depends on would 401 and leave a blank
    screen. Inline CSS keeps it to one request.
    """
    error_block = (
        f'<p class="err">{escape(message)}</p>' if message else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0d10">
<title>Deus — Sign in</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100dvh; display: grid; place-items: center;
    background: #0b0d10; color: #d7dde5; padding: 1.5rem;
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  form {{
    width: 100%; max-width: 22rem; border: 1px solid #1e242c;
    background: #0f1319; padding: 1.5rem; border-radius: 10px;
  }}
  h1 {{ margin: 0 0 .25rem; font-size: 1.05rem; letter-spacing: .08em; color: #6ee7a8; }}
  p.sub {{ margin: 0 0 1.25rem; color: #6b7683; font-size: .8rem; }}
  label {{ display: block; margin-bottom: .4rem; font-size: .75rem; color: #8b96a4;
           text-transform: uppercase; letter-spacing: .1em; }}
  input {{
    width: 100%; padding: .7rem .75rem; border-radius: 6px; font: inherit;
    background: #070a0e; border: 1px solid #232a33; color: #e6ecf3;
  }}
  input:focus {{ outline: none; border-color: #2f6f4f; }}
  button {{
    width: 100%; margin-top: 1rem; padding: .7rem; border: 0; border-radius: 6px;
    background: #1b7a4b; color: #f2fff8; font: inherit; font-weight: 600;
    letter-spacing: .05em; cursor: pointer;
  }}
  button:hover {{ background: #1f8f57; }}
  p.err {{ margin: 0 0 1rem; padding: .55rem .7rem; border-radius: 6px;
           background: #2a1418; border: 1px solid #5b2530; color: #ffb4bd;
           font-size: .8rem; }}
</style>
</head>
<body>
<form method="post" action="/api/auth/login">
  <h1>DEUS</h1>
  <p class="sub">Private instance. Passphrase required.</p>
  {error_block}
  <label for="passphrase">Passphrase</label>
  <input id="passphrase" name="passphrase" type="password" autocomplete="current-password"
         autofocus required>
  <input type="hidden" name="next" value="{escape(next_path, quote=True)}">
  <button type="submit">Sign in</button>
</form>
</body>
</html>"""
