# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "semnet"
author = "Ian Goodrich"
release = "0.1.0"
copyright = "2025, Ian Goodrich"

# The short X.Y version
version = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# Logo and favicon
html_logo = "_static/network-icon.svg"
html_favicon = "_static/favicon.svg"

# Theme options
html_theme_options = {
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# HTML title and project metadata
html_title = f"{project} documentation"
html_short_title = project

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# MyST settings - let MyST handle .md files automatically
# source_suffix handled by myst_parser extension

# Auto-doc settings
autodoc_member_order = "bysource"
# Mock all external dependencies since we only need the API structure
autodoc_mock_imports = [
    "annoy",
    "networkx",
    "numpy",
    "pandas",
    "tqdm",
]

# Path to the Python modules
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
