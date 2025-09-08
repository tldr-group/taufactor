import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import taufactor

project = 'taufactor'
copyright = "2023, tldr group"
author = "tldr group"

# The short X.Y version.
version = taufactor.__version__
# The full version, including alpha/beta/rc tags.
release = taufactor.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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
