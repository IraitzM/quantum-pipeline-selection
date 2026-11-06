"""The library of the EPJ companion repository.

Deliberately empty. The experiment scripts import the modules they need
directly, ``from code.qnn import ...`` and so on, and the working repository's
``code/__init__.py`` pulls in modules this repository does not ship (the
fast-experiment runner and its preprocessing helpers). Importing ``code``
triggers none of them here.

Note: the package name ``code`` shadows the standard-library module, exactly as
in the working repository, where the experiment scripts insert the repository
root at the front of ``sys.path`` before importing it.
"""
