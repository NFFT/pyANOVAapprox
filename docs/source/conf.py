# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Macht den src-Ordner für Sphinx und autodoc auffindbar
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ANOVAapprox"
copyright = "2025, Felix"
author = "Felix"
release = "29.07.2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Erweiterungen hinzugefügt, damit automodule, Mathe und Markdown funktionieren
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Unterstützt Google/NumPy-Style Docstrings
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Auf das im Workflow genutzte Theme umgestellt
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]