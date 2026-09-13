"""Run isolated benchmarks: python -m benchmarks.render --output report.json.

Timing, tracemalloc and cProfile are separate passes. Every case runs in a fresh
process. --reference compares exact output hashes and reports speed ratios.
"""
import argparse
import cProfile
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time
import tracemalloc

import cv2
import numpy as np
from drawcv import OpenCVRenderer
from benchmarks.scenes import CASES, STRESS_CASES, TIMES, build_scene, describe


def digest(data):
    return hashlib.sha256(data).hexdigest()


def process_memory():
    """Absolute OS process high-water memory, including imports/scene/warmup."""
    if sys.platform == 'win32':
        import ctypes as c
        from ctypes import wintypes as w
        class Counters(c.Structure):
            _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [(name, c.c_size_t) for name in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = c.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = w.HANDLE
        psapi = c.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Counters), w.DWORD]
        data = Counters()
        data.cb = c.sizeof(data)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), c.byref(data), data.cb):
            raise c.WinError(c.get_last_error())
        return {'rss_bytes': data.WorkingSetSize, 'peak_rss_bytes': data.PeakWorkingSetSize}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'rss_bytes': None, 'peak_rss_bytes': int(peak*(1 if sys.platform == 'darwin' else 1024))}


def environment():
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    try:
        commit = subprocess.check_output(['git', '-c', f'safe.directory={Path.cwd().as_posix()}',
            'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    import drawcv
    source_root = Path(drawcv.__file__).parent
    source_digest = hashlib.sha256()
    for source in sorted(source_root.rglob('*.py')):
        source_digest.update(source.relative_to(source_root).as_posix().encode()+b'\0')
        source_digest.update(source.read_bytes()+b'\0')
    return {'platform': platform.platform(), 'cpu': cpu, 'logical_cpus': os.cpu_count(),
            'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
            'opencv_threads': cv2.getNumThreads(), 'git_head': commit,
            'renderer_source_sha256': digest(Path(sys.modules[OpenCVRenderer.__module__].__file__).read_bytes()),
            'library_source_sha256': source_digest.hexdigest()}


def worker(case, warmup, samples, threads):
    cv2.setNumThreads(threads)
    scene = build_scene(case)
    original = scene.to_json()
    renderer = OpenCVRenderer()
    alpha = case != 'dense_bgr'
    def frame(index):
        return (scene.render_at_time(TIMES[index%len(TIMES)], renderer, alpha=alpha).buffer
                if case == 'animation' else renderer.render(scene, alpha=alpha).buffer)
    memory_before = process_memory()
    for i in range(warmup):
        frame(i)
    durations = []
    for i in range(samples):
        start = time.perf_counter_ns()
        pixels = frame(i)
        durations.append((time.perf_counter_ns()-start)/1e6)
        del pixels
    memory_after = process_memory()
    print(f'{case}: timing complete; collecting hashes, memory and profile', file=sys.stderr, flush=True)
    hashes = [digest(frame(i).tobytes()) for i in range(3 if case == 'animation' else 1)]
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    pixels = frame(0)
    _, peak = tracemalloc.get_traced_memory()
    after = tracemalloc.take_snapshot()
    retained = after.compare_to(before, 'filename')
    tracemalloc.stop()
    del pixels
    profile = cProfile.Profile()
    profile.runcall(frame, 0)
    stats = pstats.Stats(profile)
    top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:15]
    if original != scene.to_json():
        raise AssertionError('Benchmark rendering mutated authored scene state')
    return {'case': case, 'scene': describe(scene), 'scene_sha256': digest(original.encode()),
        'alpha': alpha, 'frame_hashes': hashes, 'warmup_frames': warmup, 'samples': samples,
        'times_seconds': list(TIMES) if case == 'animation' else None,
        'frame_ms': durations, 'median_ms': float(np.median(durations)),
        'p95_ms': float(np.percentile(durations, 95)), 'min_ms': min(durations), 'max_ms': max(durations),
        'memory_before_warmup': memory_before, 'memory_after_timing': memory_after,
        'traced_peak_bytes': peak,
        'traced_positive_retained_blocks': sum(max(0, item.count_diff) for item in retained),
        'traced_positive_retained_bytes': sum(max(0, item.size_diff) for item in retained),
        'profile': [{'file': Path(key[0]).name, 'line': key[1], 'function': key[2],
            'calls': value[1], 'self_seconds': value[2], 'cumulative_seconds': value[3]} for key, value in top]}


def compare(result, reference):
    previous = next(r for r in reference['results'] if r['case'] == result['case'])
    if previous['scene_sha256'] != result['scene_sha256'] or previous['frame_hashes'] != result['frame_hashes']:
        raise AssertionError(f"{result['case']}: reference scene or exact frame hashes differ")
    result['reference_median_ms'] = previous['median_ms']
    result['median_speedup'] = previous['median_ms']/result['median_ms']
    result['reference_images_exact'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmark.json'))
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--resume', action='store_true', help='Reuse completed cases from the same checkout/settings')
    parser.add_argument('--cases', nargs='+', choices=CASES+STRESS_CASES, default=list(CASES))
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--worker', choices=CASES+STRESS_CASES)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 3 or args.threads < 1:
        parser.error('warmup >= 0, samples >= 3 and threads >= 1 are required')
    if args.worker:
        print(json.dumps(worker(args.worker, args.warmup, args.samples, args.threads)))
        return
    cv2.setNumThreads(args.threads)
    results = []
    child_env = os.environ.copy()
    child_env['PYTHONPATH'] = os.pathsep.join(str(Path(p or '.').resolve()) for p in sys.path)
    report = {'format_version': 1, 'environment': environment(), 'results': results}
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text(encoding='utf-8'))
        if saved['environment'] != report['environment']:
            raise ValueError('Resume requires matching environment and renderer source')
        for item in saved['results']:
            if item['case'] not in args.cases:
                continue
            if item['samples'] != args.samples or item['warmup_frames'] != args.warmup:
                raise ValueError('Resume frame counts differ')
            if item['scene_sha256'] != digest(build_scene(item['case']).to_json().encode()):
                raise ValueError('Resume scene fixture differs')
            results.append(item)
    reference = json.loads(args.reference.read_text(encoding='utf-8')) if args.reference else None
    if reference and reference['environment']['opencv_threads'] != args.threads:
        raise ValueError('Reference OpenCV thread count differs')
    if reference:
        for result in results:
            compare(result, reference)
    for case in args.cases:
        if any(item['case'] == case for item in results):
            continue
        print(f'Running {case}...', flush=True)
        output = subprocess.check_output([sys.executable, '-m', 'benchmarks.render', '--worker', case,
            '--warmup', str(args.warmup), '--samples', str(args.samples), '--threads', str(args.threads)],
            env=child_env, text=True)
        result = json.loads(output)
        if reference:
            compare(result, reference)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"{case}: median {result['median_ms']:.2f} ms, p95 {result['p95_ms']:.2f} ms", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
