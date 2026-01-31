# Configuration file for the Sphinx documentation builder.

import os
import sys
import subprocess

# -- Path setup for autodoc (Python API) ------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# -- Project information -----------------------------------------------------

project = 'Tangent'
copyright = '2019, Kevin Sheridan'
author = 'Kevin Sheridan'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'myst_parser',
    'sphinx_inline_tabs',
]

autodoc_mock_imports = ['cppyy', 'numpy']

# -- Napoleon configuration (Google-style docstrings) -----------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'doxygen']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'Tangent'

# Furo theme options
html_theme_options = {
    "light_logo": "tangent-logo-light.svg",
    "dark_logo": "tangent-logo-dark.svg",
    "light_css_variables": {
        "color-brand-primary": "#4a90d9",
        "color-brand-content": "#4a90d9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6cb0f0",
        "color-brand-content": "#6cb0f0",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# -- Breathe configuration ---------------------------------------------------

breathe_projects = {
    'Tangent': os.path.join(os.path.dirname(__file__), 'doxygen/xml')
}
breathe_default_project = 'Tangent'
breathe_default_members = ('members', 'undoc-members')

# -- MyST configuration (for markdown support) -------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]
myst_heading_anchors = 3

# -- Source file configuration -----------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
