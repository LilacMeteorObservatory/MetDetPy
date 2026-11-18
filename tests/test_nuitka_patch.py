"""Patch compatibility and compiler isolation without requiring Nuitka or MSVC."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from MetLib.packaging.nuitka_patches import patch_onefile_bootstrap as patch


def bootstrap_source():
    return '\n'.join((patch.ANCHOR_HELPER, patch.ANCHOR_FORK, patch.ANCHOR_CREATE)) + '\n'


@pytest.mark.parametrize('version', ['4.0.5', '4.1.3', '99.0'])
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_versions_keep_anchor_checks(tmp_path, capsys, version, newline):
    source = tmp_path/'OnefileBootstrap.c'
    source.write_bytes(bootstrap_source().replace('\n', newline).encode())
    patch.patch_bootstrap(source, version)
    assert patch.BINARY_MARKER in source.read_text()
    assert ('WARNING' in capsys.readouterr().err) == (version not in patch.SUPPORTED_NUITKA_VERSIONS)
    if newline == '\r\n':
        assert b'\n' not in source.read_bytes().replace(b'\r\n', b'')


@pytest.mark.parametrize('anchor', [patch.ANCHOR_HELPER, patch.ANCHOR_FORK, patch.ANCHOR_CREATE])
@pytest.mark.parametrize('count', [0, 2])
def test_bad_anchor_never_changes_source(tmp_path, anchor, count):
    source = tmp_path/'OnefileBootstrap.c'
    original = bootstrap_source().replace(anchor, anchor*count).encode()
    source.write_bytes(original)
    with pytest.raises(SystemExit):
        patch.patch_bootstrap(source, '4.0.5')
    assert source.read_bytes() == original


@pytest.fixture
def installed_nuitka(tmp_path, monkeypatch):
    package = tmp_path/'installed'/'nuitka'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    source = package/'build'/'static_src'/'OnefileBootstrap.c'
    source.parent.mkdir(parents=True)
    source.write_text(bootstrap_source())
    import importlib.util
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: SimpleNamespace(
        submodule_search_locations=[str(package)]))
    monkeypatch.setattr(patch, 'get_nuitka_version', lambda: '99.0')
    original_tempdir = patch.tempfile.TemporaryDirectory
    monkeypatch.setattr(patch.tempfile, 'TemporaryDirectory',
                        lambda **kwargs: original_tempdir(dir=tmp_path, **kwargs))
    return source


@pytest.mark.parametrize('abort', [False, True])
def test_private_copy_selection_preserves_environment_and_cleans_up(installed_nuitka, monkeypatch, abort):
    original = installed_nuitka.read_bytes()
    monkeypatch.setenv('PYTHONPATH', 'parent-path')
    input_env = {'PYTHONPATH': 'pyexiv2-stage'+os.pathsep+'existing-path'}
    stage = None
    try:
        with patch.isolated_nuitka(input_env) as env:
            stage = Path(env['PYTHONPATH'].split(os.pathsep)[0])
            assert env['PYTHONPATH'].split(os.pathsep)[1:] == ['pyexiv2-stage', 'existing-path']
            assert patch.BINARY_MARKER in (stage/'nuitka/build/static_src/OnefileBootstrap.c').read_text()
            assert installed_nuitka.read_bytes() == original
            if abort:
                raise RuntimeError('compilation failed')
    except RuntimeError:
        assert abort
    assert stage is not None and not stage.exists()
    assert installed_nuitka.read_bytes() == original
    assert not list(installed_nuitka.parent.glob('*.metdetpy-orig'))
    assert os.environ['PYTHONPATH'] == 'parent-path'
    assert input_env['PYTHONPATH'].startswith('pyexiv2-stage')


def test_patch_failure_cleans_up_copy(installed_nuitka):
    installed_nuitka.write_text(patch.SOURCE_MARKER)
    with pytest.raises(SystemExit):
        with patch.isolated_nuitka():
            pytest.fail('Already patched installations must be rejected')
    assert not list(installed_nuitka.parents[3].parent.glob('metdetpy-nuitka-*'))
    assert installed_nuitka.read_text() == patch.SOURCE_MARKER


@pytest.mark.parametrize('kind', ['plain', 'packed', 'missing', 'unmarked', 'unpack-failed'])
def test_artifact_marker_verification(tmp_path, monkeypatch, kind):
    from MetLib.packaging import nuitka as backend

    marker = patch.BINARY_MARKER.encode('ascii')
    monkeypatch.setattr(backend, 'SCRIPTS', ['sample.py'])
    exe = tmp_path/'sample.exe'
    if kind != 'missing':
        exe.write_bytes(marker if kind == 'plain' else b'packed executable')
    original = exe.read_bytes() if exe.exists() else None
    calls = []

    def unpack(command):
        calls.append(command)
        assert command[-1] == str(exe)
        Path(command[3]).write_bytes(marker if kind == 'packed' else b'no marker')
        return (1 if kind == 'unpack-failed' else 0), 0

    monkeypatch.setattr(backend, 'run_cmd', unpack)
    if kind in ('plain', 'packed'):
        backend.verify_msix_onefile_patch(str(tmp_path), marker, upx_binary='upx')
    else:
        with pytest.raises(SystemExit) as error:
            backend.verify_msix_onefile_patch(str(tmp_path), marker, upx_binary='upx')
        assert {'missing': 'does not exist', 'unmarked': 'missing after UPX',
                'unpack-failed': 'unpack failed'}[kind] in str(error.value)
    assert len(calls) == (kind not in ('plain', 'missing'))
    assert not list(tmp_path.glob('patch-check-*'))
    assert (exe.read_bytes() if exe.exists() else None) == original


def test_uncompressed_missing_marker_is_rejected(tmp_path):
    from MetLib.packaging import nuitka as backend

    for script in backend.SCRIPTS:
        (tmp_path/Path(script).with_suffix('.exe')).write_bytes(b'no patch')
    with pytest.raises(SystemExit, match='patch marker missing'):
        backend.verify_msix_onefile_patch(str(tmp_path), b'marker')
