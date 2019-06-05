# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

autodoc_mock_imports = ['torch', 'torchbearer', 'torch.nn.utils.clip_grad_norm', 'torchvision', 'torchvision.utils', 'torchvision.datasets', 'torchvision.datasets.folder', 'torch.nn', 'torch.nn.functional', 'torch.nn.modules', 'torch.optim', 'torch.distributions.utils', 'torch.distributions', 'torch.utils', 'torch.utils.data', 'numpy', 'sklearn', 'sklearn.metrics', 'tqdm', 'tensorboardX', 'tensorboardX.torchvis', 'livelossplot', 'IPython']


# -- Project information -----------------------------------------------------

project = 'visual'
copyright = '2019, Ethan Harris, Matthew Painter'
author = 'Ethan Harris, Matthew Painter'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

version_dict = {}
exec(open("../torchbearer/version.py").read(), version_dict)

# The short X.Y version.
version = version_dict['__version__']
# The full version, including alpha/beta/rc tags.
release = version_dict['__version__']

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.intersphinx']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

