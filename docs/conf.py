# Configuration file for the Sphinx documentation builder.

import os
import subprocess

# -- Project information -----------------------------------------------------

project = 'ArgMin'
copyright = '2019, Kevin Sheridan'
author = 'Kevin Sheridan'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'doxygen']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'ArgMin'

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#4a90d9",
        "color-brand-content": "#4a90d9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6cb0f0",
        "color-brand-content": "#6cb0f0",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# -- Breathe configuration ---------------------------------------------------

breathe_projects = {
    'ArgMin': os.path.join(os.path.dirname(__file__), 'doxygen/xml')
}
breathe_default_project = 'ArgMin'
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
