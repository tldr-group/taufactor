import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'taufactor'
copyright = "2023, tldr group"
author = "tldr group"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx',
]

myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]

nb_execution_mode = 'off'
html_theme = 'sphinx_rtd_theme'
autodoc_mock_imports = [
    'matplotlib',
    'pyvista',
    'psutil',
    'IPython',
    'numpy',
    'torch'
]
master_doc = 'index'
exclude_patterns = ['paper/*', '_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
    'inherited-members': False,
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
nbsphinx_execute = 'never'
