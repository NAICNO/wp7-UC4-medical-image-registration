# Configuration file for the Sphinx documentation builder.

project = '3D Medical Image Registration & Segmentation'
copyright = '2026, NAIC / Sigma2'
author = 'NAIC Team'
release = '0.1'

extensions = ['sphinxcontrib.mermaid', 'sphinx_lesson', 'sphinx.ext.githubpages', 'sphinx_tabs.tabs']

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
sphinx_tabs_disable_css_loading = True
html_static_path = ['_static']
