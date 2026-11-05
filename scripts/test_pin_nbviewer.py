"""Tests for pin_nbviewer.py  (run:  python -m unittest scripts/test_pin_nbviewer.py -v)"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pin_nbviewer as pn  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
NB = "https://nbviewer.org/github/Lauret-Boris/makystats/blob/main/exemples/ARMA/A%20B.ipynb"
GH = "https://github.com/Lauret-Boris/makystats/blob/main/exemples/ARMA/A%20B.ipynb"
CO = "https://colab.research.google.com/github/Lauret-Boris/makystats/blob/main/exemples/ARMA/A%20B.ipynb"
OTHER_REPO = "https://nbviewer.org/github/someone/else/blob/main/x.ipynb"
OTHER_BRANCH = "https://nbviewer.org/github/Lauret-Boris/makystats/blob/maintenance/x.ipynb"
BRANCH_PATH = "https://nbviewer.org/github/Lauret-Boris/makystats/tree/main/exemples"


def run(*argv, env=None):
    old = dict(os.environ)
    try:
        if env is not None:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = pn.main(list(argv))
            except SystemExit as e:  # resolve_sha raises SystemExit(msg)
                code = e.code if isinstance(e.code, int) else 2
                err.write(str(e.code))
        return code, out.getvalue(), err.getvalue()
    finally:
        os.environ.clear()
        os.environ.update(old)


class PinText(unittest.TestCase):
    def test_pins_only_nbviewer_main(self):
        html = f'<a href="{GH}">GitHub</a> <a href="{NB}">nbviewer</a> <a href="{CO}">Colab</a>'
        new, n = pn.pin_text(html, SHA)
        self.assertEqual(n, 1)
        self.assertIn(NB.replace("/main/", f"/{SHA}/"), new)
        self.assertIn(GH, new)  # GitHub untouched
        self.assertIn(CO, new)  # Colab untouched

    def test_other_repo_branch_and_tree_untouched(self):
        for url in (OTHER_REPO, OTHER_BRANCH, BRANCH_PATH):
            new, n = pn.pin_text(f'<a href="{url}">x</a>', SHA)
            self.assertEqual(n, 0, url)
            self.assertIn(url, new)

    def test_idempotent(self):
        once, n1 = pn.pin_text(NB, SHA)
        twice, n2 = pn.pin_text(once, SHA)
        self.assertEqual((n1, n2), (1, 0))
        self.assertEqual(once, twice)

    def test_repin_with_new_sha_not_applied(self):
        # An already pinned link is intentionally left alone (never "main" again).
        once, _ = pn.pin_text(NB, SHA)
        other = "f" * 40
        again, n = pn.pin_text(once, other)
        self.assertEqual(n, 0)
        self.assertEqual(again, once)


class Sha(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(pn.resolve_sha(SHA), SHA)
        self.assertEqual(pn.resolve_sha(SHA.upper()), SHA)

    def test_invalid(self):
        for bad in ("abc123", "g" * 40, SHA[:-1], SHA + "0", "main", "$(rm -rf /)"):
            with self.assertRaises(SystemExit, msg=bad):
                pn.resolve_sha(bad)


class Cli(unittest.TestCase):
    def make_site(self, root: Path) -> Path:
        site = root / "website"
        (site / "fr").mkdir(parents=True)
        # CRLF on purpose: must survive byte for byte.
        (site / "examples.html").write_bytes(f'<a href="{NB}">n</a>\r\n<a href="{GH}">g</a>\r\n'.encode())
        (site / "fr" / "examples.html").write_text(f'<a href="{NB}">n</a>\n', encoding="utf-8")
        (site / "index.html").write_text("<p>no notebook here</p>\n", encoding="utf-8")
        (site / "style.css").write_text(f"/* {NB} */\n", encoding="utf-8")  # not html: untouched
        return site

    def test_dst_copy_and_source_untouched(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            site = self.make_site(root)
            before = (site / "examples.html").read_bytes()
            code, out, err = run("--src", str(site), "--dst", str(root / "_site"), "--sha", SHA)
            self.assertEqual(code, 0, err)
            self.assertIn("pinned     : 2 nbviewer link(s) in 2 file(s)", out)
            self.assertEqual((site / "examples.html").read_bytes(), before)  # source intact
            out_bytes = (root / "_site" / "examples.html").read_bytes()
            self.assertIn(b"\r\n", out_bytes)  # CRLF kept
            self.assertEqual(out_bytes, before.replace(b"/blob/main/exemples/ARMA/A%20B.ipynb\">n", f"/blob/{SHA}/exemples/ARMA/A%20B.ipynb\">n".encode()))
            self.assertIn(SHA, (root / "_site" / "fr" / "examples.html").read_text(encoding="utf-8"))
            self.assertIn("/blob/main/", (root / "_site" / "style.css").read_text(encoding="utf-8"))

    def test_dst_must_not_exist(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            site = self.make_site(root)
            (root / "_site").mkdir()
            code, _, err = run("--src", str(site), "--dst", str(root / "_site"), "--sha", SHA)
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)

    def test_dst_inside_src_refused(self):
        with tempfile.TemporaryDirectory() as t:
            site = self.make_site(Path(t))
            code, _, err = run("--src", str(site), "--dst", str(site / "out"), "--sha", SHA)
            self.assertEqual(code, 2)
            self.assertIn("outside", err)

    def test_in_place_needs_ci(self):
        with tempfile.TemporaryDirectory() as t:
            site = self.make_site(Path(t))
            before = (site / "fr" / "examples.html").read_bytes()
            code, _, err = run("--src", str(site), "--in-place", "--sha", SHA, env={"CI": None})
            self.assertEqual(code, 2)
            self.assertEqual((site / "fr" / "examples.html").read_bytes(), before)
            code, out, err = run("--src", str(site), "--in-place", "--sha", SHA, env={"CI": "true"})
            self.assertEqual(code, 0, err)
            self.assertIn(SHA, (site / "fr" / "examples.html").read_text(encoding="utf-8"))

    def test_needs_exactly_one_mode(self):
        with tempfile.TemporaryDirectory() as t:
            site = self.make_site(Path(t))
            self.assertEqual(run("--src", str(site), "--sha", SHA)[0], 2)
            self.assertEqual(run("--src", str(site), "--dst", str(Path(t) / "o"), "--in-place", "--sha", SHA)[0], 2)

    def test_bad_sha_stops_before_copy(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            site = self.make_site(root)
            code, _, _ = run("--src", str(site), "--dst", str(root / "_site"), "--sha", "nope")
            self.assertEqual(code, 2)
            self.assertFalse((root / "_site").exists())

    def test_second_run_is_noop_with_warning(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            site = self.make_site(root)
            run("--src", str(site), "--dst", str(root / "a"), "--sha", SHA)
            code, out, _ = run("--src", str(root / "a"), "--dst", str(root / "b"), "--sha", SHA)
            self.assertEqual(code, 0)
            self.assertIn("warning", out)


if __name__ == "__main__":
    unittest.main()
