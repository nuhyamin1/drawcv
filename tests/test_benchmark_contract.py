"""Benchmark fixture determinism and exact-image comparison gates; no timing limits."""
import copy
import pytest
from benchmarks.scenes import CASES, build_scene
from benchmarks.render import compare, digest


@pytest.mark.parametrize('case', CASES)
def test_benchmark_sources_are_reproducible(case):
    assert digest(build_scene(case).to_json().encode()) == digest(build_scene(case).to_json().encode())


@pytest.mark.parametrize('field', ['scene_sha256', 'frame_hashes'])
def test_comparison_rejects_changed_scene_or_pixels(field):
    result = {'case': 'fixture', 'scene_sha256': 'scene', 'frame_hashes': ['pixels'], 'median_ms': 2}
    reference = {'results': [copy.deepcopy(result)]}
    compare(result, reference)
    assert result['reference_images_exact'] and result['median_speedup'] == 1
    reference['results'][0][field] = 'changed'
    with pytest.raises(AssertionError, match='exact frame hashes differ'):
        compare(result, reference)
