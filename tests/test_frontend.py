"""Tests for the Next.js frontend — package structure and build verification."""

import json
import shutil
import subprocess
import pytest
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="Node.js is not installed",
)


class TestPackageJson:
    """Frontend package.json structure."""

    def test_package_json_exists(self):
        pkg = FRONTEND_DIR / "package.json"
        assert pkg.exists(), f"package.json not found at {pkg}"

    def test_has_nextjs_dependency(self):
        pkg_data = json.loads((FRONTEND_DIR / "package.json").read_text())
        deps = pkg_data.get("dependencies", {})
        assert "next" in deps, "next is not in dependencies"

    def test_has_react_dependency(self):
        pkg_data = json.loads((FRONTEND_DIR / "package.json").read_text())
        deps = pkg_data.get("dependencies", {})
        assert "react" in deps, "react is not in dependencies"

    def test_has_build_script(self):
        pkg_data = json.loads((FRONTEND_DIR / "package.json").read_text())
        scripts = pkg_data.get("scripts", {})
        assert "build" in scripts or "build:static" in scripts


class TestBuild:
    """Frontend build verification."""

    @pytest.mark.frontend
    def test_static_build_succeeds(self):
        """Run npm run build:static and verify exit code."""
        result = subprocess.run(
            ["npm", "run", "build:static"],
            cwd=str(FRONTEND_DIR),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"Build failed:\n{result.stderr[:500]}"

    @pytest.mark.frontend
    def test_static_export_output_exists(self):
        """After build, verify the out/ directory has expected files."""
        out_dir = FRONTEND_DIR / "out"
        if not out_dir.is_dir():
            pytest.skip("Static export directory not found — run build:static first")
        assert (out_dir / "index.html").exists(), "index.html missing from export"


class TestDevConfig:
    """Development server configuration."""

    def test_next_config_exists(self):
        config_files = list(FRONTEND_DIR.glob("next.config.*"))
        assert len(config_files) > 0, "No next.config file found"

    def test_typescript_config_exists(self):
        tsconfig = FRONTEND_DIR / "tsconfig.json"
        assert tsconfig.exists() or (FRONTEND_DIR / "tsconfig.ts").exists()


class TestDependencies:
    """Additional dependency checks."""

    def test_node_modules_exist(self):
        """node_modules should exist (or we can't run anything)."""
        nm = FRONTEND_DIR / "node_modules"
        if not nm.is_dir():
            pytest.skip("node_modules not installed — run npm install first")
        assert (nm / "next").is_dir(), "next package not installed"
        assert (nm / "react").is_dir(), "react package not installed"
