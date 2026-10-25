# makystats documentation site

Static HTML for **GitHub Pages**. This folder is the **only** content published
to the documentation website.

## Important: do not confuse with `docs/`

| Path | Role |
|------|------|
| **`docs/`** (repo root) | Formula PNGs, logo, plots used by `README.md` via raw GitHub URLs. **Not** the Pages root. |
| **`website/`** (this folder) | User-facing documentation site (EN + FR). Deployed to `gh-pages`. |

## Local preview

```bash
python -m http.server 8000 --directory website
# open http://127.0.0.1:8000/
```

## Publish

1. Push `website/` and `.github/workflows/pages.yml` on `main`.
2. Workflow deploys the **contents** of `website/` to branch `gh-pages` (site root).
3. Repo **Settings → Pages → Source = Deploy from branch `gh-pages` / root (/**)**.

URL: `https://Lauret-Boris.github.io/makystats/`

No Python source is copied into the published site — documentation pages only.

## Examples page

`examples.html` / `fr/examples.html` list **only** notebooks that exist under
the GitHub folder [`exemples/`](https://github.com/Lauret-Boris/makystats/tree/main/exemples)
(currently VECM). Links point to GitHub / nbviewer / Colab — notebooks are not
copied into this static site.

