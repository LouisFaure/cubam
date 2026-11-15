"""Sphinx configuration of the cubam documentation."""

from importlib import metadata

project = "cubam"
author = "Louis Faure"
copyright = "2026, Louis Faure"
try:
    release = metadata.version("cubam")
except metadata.PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxarg.ext",
    # The theme of the scanpy docs, and its autosummary templates for the API pages.
    "scverse_doc.theme",
    "scverse_doc.autodoc",
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# The API pages list each module's members in autosummary tables, one page per object.
autosummary_generate = True
autosummary_filename_map = {
    # The function and its result class differ only by case, which macOS file names don't.
    "cubam.apps.coverage.coverage": "cubam.apps.coverage.coverage-function",
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

source_repository = "https://github.com/LouisFaure/cubam"

html_theme = "scverse"
html_title = "cubam: GPU-accelerated BAM parsing into columns"
html_show_sphinx = False
html_theme_options = {
    # cubam is not an scverse package: drop the scverse menu, banner and footer.
    "show_ecosystem_dropdown": False,
    "announcement": "",
    "article_header_end": ["theme-switcher", "navbar-icon-links", "toggle-secondary-sidebar"],
    "footer_start": ["copyright"],
    # Amber, for the sepia of dark mode; custom.css keeps blue links on light
    # and pins the exact shades of both modes.
    "accent": "#e68c2c",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    # The Flux wordmark, drawn like gatac's: "cu" in green, "bam" dark grey on light pages, light grey on dark.
    "logo": {
        "image_light": "_static/logo_light.svg",
        "image_dark": "_static/logo_dark.svg",
        "alt_text": "cubam",
    },
}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]


def _drop_scverse_links(app):
    """Remove the scverse Discourse link that the theme adds to the header icons."""
    options = app.config.html_theme_options
    options["icon_links"] = [link for link in options.get("icon_links", []) if "scverse.org" not in link["url"]]


def setup(app):
    # After the theme's own builder-inited hook, which appends the link.
    app.connect("builder-inited", _drop_scverse_links, priority=900)
