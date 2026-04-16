# Copyright 2026 The IntoSheet Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level installer for IntoSheet.

This project is composed of local `t5x/` and `mt3/` packages plus additional
runtime dependencies used by Partitur. A fresh environment previously required
several manual pip commands. This setup.py turns the repository root into the
single installation entry point.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Iterable, List

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install


ROOT = pathlib.Path(__file__).resolve().parent
README = (ROOT / "README_EN.md").read_text(encoding="utf-8")


def _read_requirements(*relative_paths: str) -> List[str]:
  requirements = []
  seen = set()
  for relative_path in relative_paths:
    path = ROOT / relative_path
    for raw_line in path.read_text(encoding="utf-8").splitlines():
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      if line not in seen:
        seen.add(line)
        requirements.append(line)
  return requirements


def _run_pip_install(args: Iterable[str]) -> None:
  command = [sys.executable, "-m", "pip", "install", *args]
  subprocess.check_call(command)


def _install_local_packages(editable: bool) -> None:
  pip_args = ["--no-deps"]
  if editable:
    pip_args.append("-e")

  for local_project in ("t5x", "mt3"):
    target = str(ROOT / local_project)
    if editable:
      _run_pip_install([*pip_args, target])
    else:
      _run_pip_install([target, *pip_args])


class _InstallIntoSheet(install):
  """Installs third-party deps, then local mt3/t5x packages."""

  def run(self) -> None:
    super().run()
    _install_local_packages(editable=False)


class _DevelopIntoSheet(develop):
  """Editable install for the root project plus local mt3/t5x packages."""

  def run(self) -> None:
    super().run()
    _install_local_packages(editable=True)


setuptools.setup(
    name="intosheet",
    version="0.1.0",
    description="MT3-based music transcription and sheet generation toolkit",
    long_description=README,
    long_description_content_type="text/markdown",
    author="IntoSheet Authors",
    python_requires=">=3.11,<3.12",
    packages=[],
    install_requires=_read_requirements(
        "requirements-py311.txt",
        "Partitur/requirements.txt",
    ),
    cmdclass={
        "install": _InstallIntoSheet,
        "develop": _DevelopIntoSheet,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
