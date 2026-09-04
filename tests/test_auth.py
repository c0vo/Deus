"""Tests for the dashboard access-control layer (api/auth.py).

Pure-function tests plus middleware tests against a tiny stand-in app. The real
app is deliberately not imported: main.py drags in the database, scheduler and
yfinance, none of which have anything to do with the gates being tested here.

No network, no API keys, no live LLM calls.
"""

import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from api import auth
from api.auth import (
    SESSION_COOKIE,
    AccessControlMiddleware,
    auth_router,
    is_trusted_client,
    issue_session_cookie,
    verify_session_cookie,
)
from config.settings import settings

TAILNET_DEFAULTS = ["127.0.0.0/8", "::1/128", "100.64.0.0/10", "fd7a:115c:a1e0::/48"]
PASSPHRASE = "correct-horse-battery-staple"


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Minimal app carrying the real middleware, router and a guarded route."""
    application = FastAPI()
    application.add_middleware(AccessControlMiddleware)
    application.include_router(auth_router)

    @application.get("/api/secret")
    async def secret():
        return {"secret": True}

    @application.get("/dashboard")
    async def dashboard():
        return PlainTextResponse("dashboard")

    return application


@pytest.fixture
def secured(monkeypatch):
    """Auth on, tailnet defaults in force."""
    monkeypatch.setattr(settings, "dashboard_passphrase", PASSPHRASE)
    monkeypatch.setattr(settings, "dashboard_session_days", 30)
    monkeypatch.setattr(settings, "trusted_networks", ",".join(TAILNET_DEFAULTS))
    auth.reset_login_throttle()
    yield
    auth.reset_login_throttle()


def client(app, host="127.0.0.1"):
    """An httpx client whose requests present `host` as the peer address."""
    transport = httpx.ASGITransport(app=app, client=(host, 12345))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ── Session cookie ────────────────────────────────────────────────────────


class TestSessionCookie:
    def test_roundtrip(self):
        assert verify_session_cookie(issue_session_cookie(PASSPHRASE, 3600), PASSPHRASE)

    def test_expired_cookie_rejected(self):
        assert not verify_session_cookie(issue_session_cookie(PASSPHRASE, -1), PASSPHRASE)

    def test_expiry_is_honoured_at_the_boundary(self):
        cookie = issue_session_cookie(PASSPHRASE, 10, now=1_000_000)
        assert verify_session_cookie(cookie, PASSPHRASE, now=1_000_009)
        assert not verify_session_cookie(cookie, PASSPHRASE, now=1_000_011)

    def test_changing_the_passphrase_invalidates_sessions(self):
        """The revocation story: rotating the passphrase logs everyone out."""
        cookie = issue_session_cookie(PASSPHRASE, 3600)
        assert not verify_session_cookie(cookie, PASSPHRASE + "!")

    def test_tampered_signature_rejected(self):
        cookie = issue_session_cookie(PASSPHRASE, 3600)
        flipped = cookie[:-1] + ("A" if cookie[-1] != "A" else "B")
        assert not verify_session_cookie(flipped, PASSPHRASE)

    def test_tampered_payload_rejected(self):
        """A forged far-future expiry must not survive without the key."""
        payload, _, signature = issue_session_cookie(PASSPHRASE, 3600).partition(".")
        forged = auth._b64(str(int(time.time()) + 10**9).encode()) + "." + signature
        assert not verify_session_cookie(forged, PASSPHRASE)

    @pytest.mark.parametrize(
        "value", ["", "no-dot", "....", "!!!.???", "a.b", "x" * 5000]
    )
    def test_malformed_input_is_false_not_an_exception(self, value):
        assert verify_session_cookie(value, PASSPHRASE) is False

    def test_non_ascii_passphrase(self):
        """compare_digest rejects non-ASCII str, so the code must use bytes."""
        assert verify_session_cookie(issue_session_cookie("pässwörd", 60), "pässwörd")


# ── Source-IP allowlist ───────────────────────────────────────────────────


class TestTrustedNetworks:
    @pytest.mark.parametrize(
        "host,expected",
        [
            ("127.0.0.1", True),
            ("127.1.2.3", True),
            ("::1", True),
            ("100.64.0.1", True),
            ("100.101.102.103", True),
            ("100.127.255.254", True),
            ("fd7a:115c:a1e0::1234", True),
            ("::ffff:127.0.0.1", True),      # v4-mapped loopback
            ("192.168.1.50", False),         # the hostile-WiFi case
            ("10.0.0.5", False),
            ("100.128.0.1", False),          # just outside the CGNAT range
            ("8.8.8.8", False),
            ("::ffff:192.168.1.50", False),
            ("not-an-ip", False),
            (None, False),
        ],
    )
    def test_membership(self, host, expected):
        assert is_trusted_client(host, TAILNET_DEFAULTS) is expected

    def test_empty_list_disables_the_check(self):
        assert is_trusted_client("8.8.8.8", []) is True

    def test_malformed_cidr_is_skipped_not_fatal(self):
        assert is_trusted_client("127.0.0.1", ["nonsense", "127.0.0.0/8"]) is True
        assert is_trusted_client("8.8.8.8", ["nonsense"]) is False


# ── Middleware gates ──────────────────────────────────────────────────────


class TestIpGate:
    async def test_lan_peer_is_refused(self, app, secured):
        async with client(app, host="192.168.1.50") as c:
            assert (await c.get("/api/secret")).status_code == 403

    async def test_lan_peer_cannot_even_reach_the_login_page(self, app, secured):
        """The IP gate is checked before the public-path exemption."""
        async with client(app, host="192.168.1.50") as c:
            assert (await c.get("/login")).status_code == 403

    async def test_tailnet_peer_reaches_the_gate(self, app, secured):
        async with client(app, host="100.101.102.103") as c:
            assert (await c.get("/api/secret")).status_code == 401

    async def test_gate_off_when_no_networks_configured(self, app, secured, monkeypatch):
        monkeypatch.setattr(settings, "trusted_networks", "")
        async with client(app, host="8.8.8.8") as c:
            assert (await c.get("/api/secret")).status_code == 401


class TestAuthGate:
    async def test_api_request_without_cookie_is_401_json(self, app, secured):
        async with client(app) as c:
            response = await c.get("/api/secret")
        assert response.status_code == 401
        assert response.json()["detail"]
        assert response.headers["cache-control"] == "no-store"

    async def test_browser_navigation_redirects_to_login(self, app, secured):
        async with client(app) as c:
            response = await c.get("/dashboard", headers={"accept": "text/html"})
        assert response.status_code == 302
        assert response.headers["location"] == "/login?next=/dashboard"

    async def test_valid_cookie_passes(self, app, secured):
        cookie = issue_session_cookie(PASSPHRASE, 3600)
        async with client(app) as c:
            response = await c.get("/api/secret", cookies={SESSION_COOKIE: cookie})
        assert response.status_code == 200
        assert response.json() == {"secret": True}

    async def test_cookie_from_a_previous_passphrase_is_rejected(self, app, secured):
        stale = issue_session_cookie("old-passphrase", 3600)
        async with client(app) as c:
            response = await c.get("/api/secret", cookies={SESSION_COOKIE: stale})
        assert response.status_code == 401

    async def test_expired_cookie_is_rejected(self, app, secured):
        async with client(app) as c:
            response = await c.get(
                "/api/secret", cookies={SESSION_COOKIE: issue_session_cookie(PASSPHRASE, -1)}
            )
        assert response.status_code == 401

    @pytest.mark.parametrize("path", ["/login", "/healthz"])
    async def test_public_paths_bypass_the_cookie_gate(self, app, secured, path):
        async with client(app) as c:
            assert (await c.get(path)).status_code == 200

    async def test_options_preflight_is_never_gated(self, app, secured):
        """CORS preflight carries no cookie; gating it breaks dev mode."""
        async with client(app) as c:
            response = await c.request("OPTIONS", "/api/secret")
        assert response.status_code != 401

    async def test_everything_is_open_when_no_passphrase_is_set(self, app, monkeypatch):
        monkeypatch.setattr(settings, "dashboard_passphrase", "")
        monkeypatch.setattr(settings, "trusted_networks", ",".join(TAILNET_DEFAULTS))
        async with client(app) as c:
            assert (await c.get("/api/secret")).status_code == 200


# ── Login endpoint ────────────────────────────────────────────────────────


class TestLogin:
    async def test_correct_passphrase_sets_a_working_cookie(self, app, secured):
        async with client(app) as c:
            response = await c.post(
                "/api/auth/login", json={"passphrase": PASSPHRASE}
            )
            assert response.status_code == 200
            assert SESSION_COOKIE in response.cookies
            # The cookie the server just issued must actually open the door.
            assert (await c.get("/api/secret")).status_code == 200

    async def test_cookie_is_httponly_and_lax(self, app, secured):
        async with client(app) as c:
            response = await c.post("/api/auth/login", json={"passphrase": PASSPHRASE})
        header = response.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=lax" in header
        # Secure would stop the browser sending this over the plain-HTTP tailnet.
        assert "secure" not in header

    async def test_wrong_passphrase_is_401(self, app, secured):
        async with client(app) as c:
            response = await c.post("/api/auth/login", json={"passphrase": "nope"})
        assert response.status_code == 401
        assert SESSION_COOKIE not in response.cookies

    async def test_form_post_redirects_to_next(self, app, secured):
        async with client(app) as c:
            response = await c.post(
                "/api/auth/login",
                content=f"passphrase={PASSPHRASE}&next=%2Fpredict",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
        assert response.status_code == 303
        assert response.headers["location"] == "/predict"

    async def test_form_post_with_wrong_passphrase_returns_to_login(self, app, secured):
        async with client(app) as c:
            response = await c.post(
                "/api/auth/login",
                content="passphrase=wrong&next=%2Fpredict",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
        assert response.status_code == 303
        assert response.headers["location"].startswith("/login?e=1")

    async def test_open_redirect_is_not_possible(self, app, secured):
        async with client(app) as c:
            response = await c.post(
                "/api/auth/login",
                content=f"passphrase={PASSPHRASE}&next=%2F%2Fevil.com",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
        assert response.headers["location"] == "/"

    async def test_throttled_after_repeated_failures(self, app, secured):
        async with client(app) as c:
            for _ in range(5):
                await c.post("/api/auth/login", json={"passphrase": "wrong"})
            response = await c.post("/api/auth/login", json={"passphrase": "wrong"})
            assert response.status_code == 429
            # Throttling must hold even once the right passphrase shows up.
            assert (
                await c.post("/api/auth/login", json={"passphrase": PASSPHRASE})
            ).status_code == 429

    async def test_success_clears_the_failure_counter(self, app, secured):
        async with client(app) as c:
            for _ in range(4):
                await c.post("/api/auth/login", json={"passphrase": "wrong"})
            assert (
                await c.post("/api/auth/login", json={"passphrase": PASSPHRASE})
            ).status_code == 200
            for _ in range(4):
                await c.post("/api/auth/login", json={"passphrase": "wrong"})
            assert (
                await c.post("/api/auth/login", json={"passphrase": PASSPHRASE})
            ).status_code == 200

    async def test_logout_clears_the_cookie(self, app, secured):
        async with client(app) as c:
            await c.post("/api/auth/login", json={"passphrase": PASSPHRASE})
            assert (await c.get("/api/secret")).status_code == 200
            await c.post("/api/auth/logout")
            assert (await c.get("/api/secret")).status_code == 401


class TestLoginPage:
    async def test_renders_without_any_gated_assets(self, app, secured):
        """The form must not reference /_next/*, which stays behind the gate."""
        async with client(app) as c:
            response = await c.get("/login")
        assert response.status_code == 200
        assert "_next" not in response.text
        assert "<form" in response.text
        assert 'name="passphrase"' in response.text

    async def test_error_message_is_shown(self, app, secured):
        async with client(app) as c:
            response = await c.get("/login?e=1")
        assert response.status_code == 401
        assert "Incorrect passphrase" in response.text

    async def test_next_value_is_escaped(self, app, secured):
        async with client(app) as c:
            response = await c.get('/login?next=/a"><script>x</script>')
        assert "<script>" not in response.text
