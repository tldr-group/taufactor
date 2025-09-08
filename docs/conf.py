import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import taufactor

project = 'taufactor'
copyright = "2023, tldr group"
author = "tldr group"

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
html_static_path = ['_static']
autodoc_mock_imports = [
    'matplotlib',
    'pyvista',
    'psutil',
    'IPython',
    'numpy',
    'torch',
    'taufactor'
]
master_doc = 'index'
source_suffix = ['.rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
    'inherited-members': False,
}

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
# The short X.Y version.
version = taufactor.__version__
# The full version, including alpha/beta/rc tags.
release = taufactor.__version__

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
nbsphinx_execute = 'never'
