# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys


# Get version number
with open('../gradslam/version.py', 'r') as f:
    for row in f:
        if '__version__' in row:
            gradslam_version = row.split("\"")[-2]
            break

sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'gradslam')))
# import sphinx_rtd_theme

# The master toctree document.
master_doc = 'index'


# -- Project information -----------------------------------------------------

project = 'gradslam'
copyright = '2020, Montreal Robotics'
author = 'MontrealRobotics'
version = gradslam_version
# The full version, including alpha/beta/rc tags
release = gradslam_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
]

napoleon_use_ivar = True

# Mock CUDA Imports
autodoc_mock_imports = [
    'gradslam.chamferdistcuda',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = True

# Do not prepend module name to functions
add_module_names = False

# nbsphinx parameters (for jupyter notebooks)
if os.environ.get('READTHEDOCS') == 'True':
    nbsphinx_execute = 'never'
else:
    nbsphinx_execute = 'auto'

    # Controls when a cell will time out (use -1 for no timeout)
    nbsphinx_timeout = 60


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_path = ['_themes']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

html_logo = '_static/img/gradslam-logo.png'
html_favicon = '_static/img/gradslam-favicon-32x32.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'css_files': [
        'https://fonts.googleapis.com/css?family=Lato',
        # 'https://fonts.googleapis.com/css?family=Ubuntu',
        '_static/css/pytorch_theme.css'
    ],
}
# html_css_files = [
#     'copybutton.css',
# ]
# html_js_files = [
#     'clipboard.min.js',
#     'copybutton.js',
# ]

# A list of paths that contain extra files not directly related to the documentation.
# Relative paths are taken as relative to the configuration directory. They are copied
# to the output directory. They will overwrite any existing file of the same name.
html_extra_path = []


# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper')
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt')
    #
    # 'pointsize': '10pt',

    # Font packages
    'fontpkg': '\\usepackage{amsmath, amsfonts, amssymb, amsthm}'

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LateX files. List of tuples
# (source start file, target name, title,
# author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'gradslam.tex', u'gradslam Documentation',
        [author], 1),
]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'gradslam', u'gradslam Documentation', 
        [author], 1)
]


# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author, 
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'gradslam', 'gradslam Documentation', 
        author, 'gradslam', 'Dense SLAM meets Automatic Differentiation', 
        'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('http://pytorch.org/docs/master', None),
}
