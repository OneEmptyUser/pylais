# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import pathlib
import sys
import os
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
# Añadir el directorio raíz del proyecto a sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pylais')))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pylais'
copyright = '2024, Ernesto Curbelo'
author = 'Ernesto Curbelo'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#     'private-members': False,  # Set to True if you want to include private methods
#     'special-members': '__init__',  # Add more special members if needed
#     'inherited-members': True,
#     'show-inheritance': True,
# }