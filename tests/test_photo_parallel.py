"""Device identity, bounded scheduling and sequence entrypoint regressions."""
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import MetDetPhoto
from MetLib import model, onnx_devices, photo_inference
from MetLib.metstruct import MockVideoObject
from MetLib.onnx_devices import DMLAdapter, classify_adapters, parse_photo_devices
from MetLib.photo_inference import PhotoInferencePool, PhotoTask, create_photo_models


def adapter(index, physical=(), luid=None, **kwargs):
    return DMLAdapter(index, 'Same model', luid or (index + 100, 0),
                      physical_ids=physical, **kwargs)


def fake_model(key):
    provider = {'cpu': 'CPUExecutionProvider', 'coreml': 'CoreMLExecutionProvider',
                'default': 'CPUExecutionProvider'}.get(key, 'DmlExecutionProvider')
    return SimpleNamespace(backend=SimpleNamespace(device=provider),
                           forward=lambda image: (np.array([[0, 0, 2, 2]]),
                                                  np.full((1, 9), .9)))


def test_identity_not_name_and_original_indices():
    devices = [adapter(0, ('igpu-instance',)), adapter(2, ('other-instance',)),
               adapter(3, ('igpu-instance',)), adapter(4, luid=(100, 0)),
               adapter(6, software=True), adapter(8, identity_error='denied')]
    classify_adapters(devices)
    assert [a.index for a in devices if not a.software and a.duplicate_of is None] == [0, 2, 8]
    assert devices[2].duplicate_of == devices[3].duplicate_of == 0
    assert devices[-1].identity_error == 'denied'


@pytest.mark.parametrize('value', ['gpu,default', 'cpu,', ',cpu', 'gpu,dml:0', 'cpu:0', 'dml,dml:1'])
def test_invalid_selection(value):
    with pytest.raises(ValueError):
        parse_photo_devices(value)


def test_selection_deduplicates_and_coreml_cpu():
    assert parse_photo_devices('dml:1,dml:1,cpu') == ['dml:1', 'cpu']
    assert parse_photo_devices('coreml,cpu') == ['coreml', 'cpu']


def test_auto_skips_failed_representative_without_trying_alias(monkeypatch):
    monkeypatch.setattr(photo_inference, 'discover_dml_adapters', lambda: [
        adapter(0, ('igpu',)), adapter(1, ('dgpu',)), adapter(3, ('igpu',)),
        adapter(4, software=True)])
    def initialize(key, threads):
        if key == 'dml:0':
            raise RuntimeError('unusable alias')
        return fake_model(key)
    factory = Mock(side_effect=initialize)
    models = create_photo_models(['gpu', 'cpu'], factory, ['dml', 'cpu'], Mock())
    assert [key for key, _ in models] == ['dml:1', 'cpu']
    assert [call.args[0] for call in factory.call_args_list] == ['dml:0', 'dml:1', 'cpu']
    assert factory.call_args_list[-1].args == ('cpu', photo_inference.CPU_THREAD_NUM)


def test_explicit_aliases_and_cpu_threads(monkeypatch):
    monkeypatch.setattr(photo_inference, 'discover_dml_adapters', lambda: [
        adapter(0, ('igpu',)), adapter(2, ('igpu',))])
    factory = Mock(side_effect=lambda key, threads: fake_model(key))
    models = create_photo_models(['dml:0', 'dml:2', 'cpu'], factory, ['dml', 'cpu'], Mock(), 3)
    assert [key for key, _ in models] == ['dml:0', 'cpu']
    assert factory.call_args_list[-1].args == ('cpu', 3)
    factory.reset_mock()
    create_photo_models(['coreml', 'cpu'], factory, ['coreml', 'cpu'], Mock())
    assert [call.args for call in factory.call_args_list] == [('coreml', 0), ('cpu', photo_inference.CPU_THREAD_NUM)]


def test_all_gpu_fail_falls_back_but_explicit_failure_raises(monkeypatch):
    monkeypatch.setattr(photo_inference, 'discover_dml_adapters', lambda: [adapter(1)])
    def initialize(key, threads):
        if key.startswith('dml'):
            raise RuntimeError('initialization failed')
        return fake_model(key)
    factory = Mock(side_effect=initialize)
    assert [key for key, _ in create_photo_models(['gpu'], factory, ['dml'], Mock())] == ['cpu']
    assert factory.call_args.args == ('cpu', 0)
    with pytest.raises(RuntimeError, match='initialization failed'):
        create_photo_models(['dml:1'], factory, ['dml'], Mock())


def test_non_dml_default_cpu_is_not_duplicated():
    factory = Mock(side_effect=lambda key, threads: fake_model(key))
    models = create_photo_models(['gpu', 'cpu'], factory, ['cpu'], Mock(), 4)
    assert [key for key, _ in models] == ['default']
    factory.assert_called_once_with('default', 4)


def test_cross_device_parallelism_and_single_model_confinement():
    barrier = threading.Barrier(2)
    models = []
    thread_ids = {}
    for key in ('a', 'b'):
        def forward(image, key=key):
            thread_ids.setdefault(key, set()).add(threading.get_ident())
            barrier.wait(timeout=3)
            return image, image
        models.append((key, SimpleNamespace(forward=forward)))
    with PhotoInferencePool(models) as pool:
        results = list(pool.map(PhotoTask(i, str(i), i) for i in range(4)))
    assert [r.task.index for r in results] == list(range(4))
    assert len({next(iter(ids)) for ids in thread_ids.values()}) == 2
    assert all(len(ids) == 1 for ids in thread_ids.values())
    assert all(not t.is_alive() for t in pool.threads)


def test_order_bound_and_fast_device_takes_more_tasks():
    release = threading.Event()
    first_started = threading.Event()
    fast_done = threading.Event()
    consumed = []
    calls = {'slow': 0, 'fast': 0}
    def slow(image):
        calls['slow'] += 1
        first_started.set()
        assert release.wait(3)
        time.sleep(.01)
        return image, image
    def fast(image):
        assert first_started.wait(3)
        calls['fast'] += 1
        if calls['fast'] >= 3:
            fast_done.set()
        return image, image
    def tasks():
        for i in range(30):
            consumed.append(i)
            yield PhotoTask(i, str(i), i)
    with PhotoInferencePool([('slow', SimpleNamespace(forward=slow)),
                             ('fast', SimpleNamespace(forward=fast))]) as pool:
        iterator = pool.map(tasks())
        received = []
        reader = threading.Thread(target=lambda: received.extend(iterator))
        reader.start()
        assert first_started.wait(3)
        assert fast_done.wait(3)
        assert len(consumed) - len(received) == pool.max_in_flight == 4
        release.set()
        reader.join(5)
        assert not reader.is_alive()
    assert [r.task.index for r in received] == list(range(30))
    assert calls['fast'] > calls['slow']


def test_stop_cancels_unstarted_work_and_closes_source():
    source_closed = []
    def tasks():
        try:
            for i in range(100):
                yield PhotoTask(i, str(i), i)
        finally:
            source_closed.append(True)
    instance = SimpleNamespace(forward=lambda image: (image, image))
    with PhotoInferencePool([('cpu', instance)]) as pool:
        predictions = pool.map(tasks())
        next(predictions)
        predictions.close()
    assert source_closed == [True]
    assert pool.stats['cpu'][0] <= 2
    assert not pool.pending and all(not t.is_alive() for t in pool.threads)


def test_exception_identifies_device_and_image():
    instance = SimpleNamespace(forward=Mock(side_effect=ValueError('failed')))
    with PhotoInferencePool([('coreml', instance)]) as pool:
        with pytest.raises(RuntimeError, match='coreml, input file.png'):
            list(pool.map([PhotoTask(0, 'file.png', 0)]))


@pytest.fixture
def locked_backend_factory(monkeypatch):
    monkeypatch.setattr(model.ort, 'get_available_providers',
                        lambda: ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    monkeypatch.setattr(model, 'is_lfs_pointer', lambda _: False)
    monkeypatch.setattr(model, 'discover_dml_adapters', lambda: classify_adapters([
        adapter(0, ('igpu',)), adapter(1, ('dgpu',)), adapter(2, ('igpu',))]))
    monkeypatch.setitem(model.DEVICE_MAPPING, 'default', ['DmlExecutionProvider', 'CPUExecutionProvider'])
    def session_constructor(*args, **kwargs):
        session = Mock()
        requested = kwargs['providers'][0]
        provider, options = requested if isinstance(requested, tuple) else (requested, {})
        session.get_providers.return_value = [provider]
        session.get_provider_options.return_value = {provider: options}
        session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 2, 2], name='images')]
        session.run.return_value = []
        return session
    monkeypatch.setattr(model.ort, 'InferenceSession', session_constructor)
    return lambda key, warmup=False: model.ONNXBackend('x.onnx', np.float32, warmup, key)


def test_automatic_dml_shares_only_automatic_lock(locked_backend_factory):
    automatic = [locked_backend_factory(key) for key in ('dml', 'default')]
    assert {device._device_key for device in automatic} == {'dml'}
    assert automatic[0]._lock is automatic[1]._lock
    assert locked_backend_factory('cpu')._lock is locked_backend_factory('cpu')._lock
    assert locked_backend_factory('coreml')._lock is locked_backend_factory('coreml')._lock


def test_photo_bare_dml_not_rewritten(monkeypatch):
    monkeypatch.setattr(photo_inference, 'discover_dml_adapters', lambda: [adapter(0), adapter(1)])
    factory = Mock(side_effect=lambda key, threads: fake_model(key))
    models = create_photo_models(['dml'], factory, ['dml'], Mock())
    assert [key for key, _ in models] == ['dml']
    factory.assert_called_once_with('dml', 0)


def test_yolo_initialization_canonicalizes_physical_alias(monkeypatch):
    monkeypatch.setattr(model, 'discover_dml_adapters', lambda: classify_adapters([
        adapter(0, ('igpu',)), adapter(2, ('igpu',))]))
    backend = Mock()
    backend.input_shape = [[1, 3, 8, 8]]
    constructor = Mock(return_value=backend)
    monkeypatch.setitem(model.SUFFIX2BACKEND, 'onnx', constructor)
    model.YOLOModel('x.onnx', 'float32', providers_key='dml:2', logger=Mock())
    assert constructor.call_args.args[3] == 'dml:0'


def test_bare_dml_cpu_selection_is_allowed(monkeypatch):
    assert parse_photo_devices('dml,cpu') == ['dml', 'cpu']
    factory = Mock(side_effect=lambda key, threads: fake_model(key))
    monkeypatch.setattr(photo_inference, 'discover_dml_adapters', lambda: [adapter(0)])
    models = create_photo_models(['dml', 'cpu'], factory, ['dml', 'cpu'], Mock())
    assert [key for key, _ in models] == ['dml', 'cpu']
    assert factory.call_args_list[-1].args == ('cpu', photo_inference.CPU_THREAD_NUM)


@pytest.mark.parametrize('first,second', [
    ('dml', 'dml:1'), ('dml:0', 'dml'),
    ('default', 'dml:0'), ('dml:1', 'default'),
])
def test_backend_rejects_dml_mode_mix_before_session_creation(
        locked_backend_factory, monkeypatch, first, second):
    locked_backend_factory(first)
    constructor = Mock()
    monkeypatch.setattr(model.ort, 'InferenceSession', constructor)
    with pytest.raises(ValueError, match='Cannot mix automatic DML'):
        locked_backend_factory(second)
    constructor.assert_not_called()


def test_different_device_locks_allow_parallel_run(locked_backend_factory):
    a, b = (locked_backend_factory(key) for key in ('dml:0', 'dml:1'))
    barrier = threading.Barrier(2)
    for backend in (a, b):
        backend.model_session.run.side_effect = lambda *args: barrier.wait(timeout=3)
    errors = []
    def run(backend):
        try:
            backend.forward(np.zeros((1, 3, 2, 2)))
        except Exception as error:
            errors.append(error)
    threads = [threading.Thread(target=run, args=(backend,)) for backend in (a, b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not errors and all(not t.is_alive() for t in threads)


def test_same_device_sessions_do_not_overlap(locked_backend_factory):
    a, b = (locked_backend_factory(key) for key in ('dml:0', 'dml:0'))
    active = 0
    peak = 0
    counter_lock = threading.Lock()
    start = threading.Barrier(2)
    def inference(*args):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.02)
        with counter_lock:
            active -= 1
        return []
    for backend in (a, b):
        backend.model_session.run.side_effect = inference
    def run(backend):
        start.wait(timeout=3)
        backend.forward(np.zeros((1, 3, 2, 2)))
    threads = [threading.Thread(target=run, args=(backend,)) for backend in (a, b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert peak == 1 and active == 0 and all(not t.is_alive() for t in threads)


def test_registry_creation_is_thread_safe():
    barrier = threading.Barrier(8)
    locks = []
    def request():
        barrier.wait(timeout=3)
        locks.append(model.ONNXBackend._get_device_lock('registry-test'))
    threads = [threading.Thread(target=request) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(locks) == 8 and len({id(lock) for lock in locks}) == 1


def test_conflicting_registry_requests_are_atomic():
    barrier = threading.Barrier(2)
    accepted, rejected = [], []
    def register(key):
        barrier.wait(timeout=3)
        try:
            model.ONNXBackend._get_device_lock(key)
            accepted.append(key)
        except ValueError:
            rejected.append(key)
    threads = [threading.Thread(target=register, args=(key,))
               for key in ('dml', 'dml:1')]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(accepted) == len(rejected) == 1
    assert set(model.ONNXBackend._device_locks) == set(accepted)
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize('key', ['cpu', 'coreml', 'cuda:1'])
def test_other_backends_can_coexist_with_automatic_dml(locked_backend_factory, key):
    automatic = locked_backend_factory('dml')
    other = locked_backend_factory(key)
    assert automatic._lock is not other._lock


def test_equivalent_index_spellings_share_lock(locked_backend_factory):
    assert locked_backend_factory('dml:01')._lock is locked_backend_factory('dml:1')._lock


def test_default_cuda_uses_requested_provider_family(locked_backend_factory, monkeypatch):
    monkeypatch.setitem(model.DEVICE_MAPPING, 'default',
                        ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    default = locked_backend_factory('default')
    assert default._device_key == 'cuda'
    assert default._lock is locked_backend_factory('cuda')._lock
    default.model_session.get_provider_options.assert_not_called()


def test_cpu_fallback_preserves_requested_dml_lock(locked_backend_factory, monkeypatch):
    session = Mock()
    session.get_providers.return_value = ['CPUExecutionProvider']
    session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 2, 2], name='images')]
    monkeypatch.setattr(model.ort, 'InferenceSession', lambda *a, **kw: session)
    backend = locked_backend_factory('default')
    assert backend._device_key == 'dml'
    assert backend._lock is model.ONNXBackend._get_device_lock('dml')
    assert backend._lock is not locked_backend_factory('cpu')._lock
    session.get_provider_options.assert_not_called()


def test_warmup_holds_device_lock(locked_backend_factory, monkeypatch):
    session = Mock()
    session.get_providers.return_value = ['CPUExecutionProvider']
    session.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 2, 2], name='images')]
    def warmup(*args):
        assert model.ONNXBackend._get_device_lock('cpu').locked()
        return []
    session.run.side_effect = warmup
    monkeypatch.setattr(model.ort, 'InferenceSession', lambda *a, **kw: session)
    locked_backend_factory('cpu', warmup=True)
    session.run.assert_called_once()


def test_list_devices_does_not_load_model(monkeypatch, capsys):
    constructor = Mock()
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', constructor)
    monkeypatch.setattr(MetDetPhoto, 'discover_dml_adapters', lambda: classify_adapters([
        adapter(0, ('igpu',)), adapter(2, ('igpu',))]))
    MetDetPhoto.main(['--list-devices'])
    assert 'alias of dml:0' in capsys.readouterr().out
    constructor.assert_not_called()


def test_timelapse_keeps_frame_numbers_and_releases_reader(tmp_path, monkeypatch):
    target = tmp_path / 'sequence.mp4'
    target.touch()
    output = tmp_path / 'result.json'
    frame = np.zeros((8, 8, 3), np.uint8)
    loader = Mock()
    loader.iterations = 4
    loader.pop.side_effect = [frame, None, frame, frame]
    loader.summary.return_value = MockVideoObject(image_folder=str(tmp_path)).summary()
    monkeypatch.setattr(MetDetPhoto, 'ThreadVideoLoader', Mock(return_value=loader))
    monkeypatch.setattr(MetDetPhoto, 'AVAILABLE_DEVICE_ALIAS', ['coreml', 'cpu'])
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', Mock(side_effect=lambda *a, **kw: fake_model(kw['providers_key'])))
    monkeypatch.setattr(MetDetPhoto, 'get_default_logger', lambda: Mock())
    MetDetPhoto.main([str(target), '--device', 'coreml,cpu', '--save-path', str(output)])
    data = json.loads(output.read_text(encoding='utf-8'))
    assert data['type'] == 'timelapse-prediction'
    assert [r['num_frame'] for r in data['results']] == [0, 2, 3]
    loader.release.assert_called_once()


def test_single_image_rejects_multi_device():
    with pytest.raises(SystemExit) as error:
        MetDetPhoto.main(['image.png', '--device', 'coreml,cpu'])
    assert error.value.code == 2


def test_mask_preparation_happens_in_worker():
    main_thread = threading.get_ident()
    def prepare(image):
        assert threading.get_ident() != main_thread
        return image * 0
    instance = fake_model('cpu')
    instance.forward = Mock(return_value=([], []))
    with PhotoInferencePool([('cpu', instance)], prepare=prepare) as pool:
        predictions = list(pool.map([PhotoTask(0, 'file', np.ones((8, 8, 3), np.uint8))]))
    assert not np.any(instance.forward.call_args.args[0])
    assert not np.any(predictions[0].image)


def test_manual_stop_keeps_reader_bounded_and_joins_workers(tmp_path, monkeypatch):
    target = tmp_path / 'sequence.mp4'
    target.touch()
    output = tmp_path / 'result.json'
    loader = Mock()
    loader.iterations = 100
    loader.pop.return_value = np.zeros((8, 8, 3), np.uint8)
    loader.summary.return_value = MockVideoObject(image_folder=str(tmp_path)).summary()
    monkeypatch.setattr(MetDetPhoto, 'ThreadVideoLoader', Mock(return_value=loader))
    monkeypatch.setattr(MetDetPhoto, 'AVAILABLE_DEVICE_ALIAS', ['cpu'])
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', Mock(return_value=fake_model('cpu')))
    monkeypatch.setattr(MetDetPhoto, 'get_default_logger', lambda: Mock())
    display = Mock()
    display.manual_stop = True
    monkeypatch.setattr(MetDetPhoto, 'OpenCVMetVisu', Mock(return_value=display))
    MetDetPhoto.main([str(target), '--device', 'cpu', '--visu', '--save-path', str(output)])
    assert loader.pop.call_count == 2
    loader.release.assert_called_once()
    assert json.loads(output.read_text(encoding='utf-8'))['results'] == []
    assert not any(t.name.startswith('photo-') for t in threading.enumerate())


def test_explicit_initialization_failure_does_not_save(tmp_path, monkeypatch):
    output = tmp_path / 'result.json'
    monkeypatch.setattr(MetDetPhoto, 'AVAILABLE_DEVICE_ALIAS', ['cpu'])
    monkeypatch.setattr(MetDetPhoto, 'YOLOModel', Mock(side_effect=RuntimeError('bad session')))
    logger = Mock()
    monkeypatch.setattr(MetDetPhoto, 'get_default_logger', lambda: logger)
    MetDetPhoto.main([str(tmp_path), '--device', 'cpu', '--save-path', str(output)])
    assert not output.exists()
    assert any('bad session' in call.args[0] for call in logger.error.call_args_list)
    logger.stop.assert_called_once()


def test_native_physical_identity_uses_sdk_enum_values(monkeypatch):
    """Exercise the ctypes ABI wrapper without GPU or Windows dependencies."""
    import ctypes as ct
    queries = []
    def open_adapter(pointer):
        pointer._obj.handle = 12
        return 0
    def query_adapter(pointer):
        q = pointer._obj
        queries.append(q.type)
        if q.type == 30:
            ct.cast(q.data, ct.POINTER(ct.c_uint32))[0] = 1
        elif q.type == 41:
            class PnP(ct.Structure):
                _fields_ = [('index', ct.c_uint32), ('key_type', ct.c_int32),
                            ('dest', ct.c_void_p), ('length', ct.POINTER(ct.c_uint32))]
            pnp = ct.cast(q.data, ct.POINTER(PnP)).contents
            assert pnp.key_type == 1 and pnp.index == 0
            text = ct.create_unicode_buffer('Physical-Instance')
            ct.memmove(pnp.dest, text, ct.sizeof(text))
        else:
            pytest.fail(f'Unexpected query type {q.type}')
        return 0
    native = SimpleNamespace(D3DKMTOpenAdapterFromLuid=Mock(side_effect=open_adapter),
                             D3DKMTQueryAdapterInfo=Mock(side_effect=query_adapter),
                             D3DKMTCloseAdapter=Mock(return_value=0))
    monkeypatch.setattr(ct, 'WinDLL', lambda name: native, raising=False)
    assert onnx_devices._physical_ids(onnx_devices._LUID(1, 0)) == ('physical-instance',)
    assert queries == [30, 41]
    native.D3DKMTCloseAdapter.assert_called_once()
