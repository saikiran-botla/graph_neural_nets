# Configuration file for the Sphinx documentation builder.

import os.path as osp
import sys

# C:\Users\Kunind Sahu\Documents\graphretrievaltoolkit
# ROOT_DIR = osp.abspath('../../')
sys.path.insert(0, osp.abspath('../../'))

# -- Project information

project = 'Graph Retrieval Toolkit'
# copyright = ''
author = 'Kunind Sahu, Sanidhya Anand'

# release = ''
# version = ''

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'torch': ('https://pytorch.org/docs/master', None),
    'pyg': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
autodoc_member_order = 'bysource'

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# -- Options for EPUB output
epub_show_urls = 'footnote'