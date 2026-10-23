"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pommes"
copyright = "2024, PERSEE - Mines Paris PSL"
author = "Knibiehly Thibaut"
release = "0.2.3" # will be set in gitlab CI

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "nbsphinx",
    "sphinx.ext.mathjax",  # for LaTeX rendering
]

templates_path = ["_templates"]
exclude_patterns = []

# AutoAPI settings
autoapi_dirs = ["../../pommes"]

# Napoleon settings
napoleon_google_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_logo = "_static/logo.svg"

nbsphinx_execute = "always"
nbsphinx_allow_errors = True

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "titles_only": False,
    "logo_only": True,
}

html_static_path = ["_static"]
