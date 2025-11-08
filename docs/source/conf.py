# docs/source/conf.py
import os
import sys
from datetime import datetime

# --- Make the package importable ---
# Repo layout: <repo> / ExoCcycle / ...
# conf.py lives in <repo>/docs/source/conf.py
sys.path.insert(0, os.path.abspath(os.path.join('..', '/home/bogumil/Documents/External_fids/ExoCcycle-workspace/ExoCcycle')))  # add <repo> to path

project = "ExoCcycle"
author = "ExoCcycle authors"
copyright = f"{datetime.now():%Y}, {author}"
extensions = [
    "myst_parser",              # allow Markdown (.md)
    "sphinx.ext.autodoc",       # pulls docstrings
    "sphinx.ext.autosummary",   # auto-build API summary tables
    "sphinx.ext.napoleon",      # Google/NumPy style docstrings
    "sphinx.ext.viewcode",      # source code links
    "sphinx.ext.intersphinx",   # cross-link to Python/NumPy/etc
    "sphinx_autodoc_typehints", # nicer type-hint rendering
    "sphinx_copybutton",        # copy buttons on code blocks
]
#    "sphinx.copybutton",        

# If you prefer ReST only, you can omit myst_parser.

# Auto-generate stub pages from autosummary directives:
autosummary_generate = True

# Show class members, including those without explicit docstrings:
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

# Put type hints into the description rather than the signature (cleaner)
autodoc_typehints = "description"


# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {}

# Napoleon settings (for Google/NumPy style)
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_use_param = True
# napoleon_use_rtype = True

# Intersphinx (optional but handy)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"  # or "sphinx_rtd_theme" if installed
