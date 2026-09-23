"""Pytest plugin recording every renderer output for exact before/after comparison.

    DRAWCV_CAPTURE_DIR=DIR python -m pytest -p benchmarks.capture_renders

Each test that renders writes DIR/<test id>.npz holding its outputs in call
order (Canvas buffers from render(), and the target buffer after
render_drawable()). Compare two runs with
``python -m benchmarks.references compare-dirs BEFORE AFTER``.
"""
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pytest

from drawcv import OpenCVRenderer

_captured = []


def _install():
    render, render_drawable = OpenCVRenderer.render, OpenCVRenderer.render_drawable

    def capture_render(self, scene, **kwargs):
        canvas = render(self, scene, **kwargs)
        _captured.append(canvas.buffer.copy())
        return canvas

    def capture_render_drawable(self, drawable, canvas):
        render_drawable(self, drawable, canvas)
        _captured.append(canvas.buffer.copy())

    OpenCVRenderer.render = capture_render
    OpenCVRenderer.render_drawable = capture_render_drawable


def _file_name(nodeid):
    readable = re.sub(r'[^A-Za-z0-9_.-]+', '_', nodeid)[:120]
    return f'{readable}-{hashlib.sha1(nodeid.encode()).hexdigest()[:10]}.npz'


def pytest_configure(config):
    if os.environ.get('DRAWCV_CAPTURE_DIR'):
        _install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _captured.clear()
    yield
    directory = os.environ.get('DRAWCV_CAPTURE_DIR')
    if directory and _captured:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / _file_name(item.nodeid), **{f'r{i:03d}': a for i, a in enumerate(_captured)})
    _captured.clear()
