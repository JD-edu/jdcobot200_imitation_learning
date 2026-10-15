"""Compatibility metadata for older bring-up environments.

Modern installers read ``pyproject.toml``.  The lab machine currently ships an
older setuptools, so the same minimal metadata is retained here as a fallback.
"""

from setuptools import find_packages, setup


setup(
    name="jdcobot200",
    version="0.2.0",
    description="Hardware, configuration, and robot-description utilities for JDCobot200",
    python_requires=">=3.9",
    packages=find_packages(include=("jdcobot200", "jdcobot200.*")),
    install_requires=["pyserial>=3.5", "PyYAML>=6.0"],
)
