import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import types
tqdm_stub = types.ModuleType('tqdm')
class _Tqdm(list):
    def __init__(self, iterable=None, **kwargs):
        super().__init__(iterable or [])
tqdm_stub.tqdm = _Tqdm
sys.modules['tqdm'] = tqdm_stub

colorama_stub = types.ModuleType('colorama')
class _Fore:
    WHITE = CYAN = GREEN = YELLOW = ''
class _Style:
    BRIGHT = ''
def _init(*args, **kwargs):
    pass
colorama_stub.Fore = _Fore
colorama_stub.Style = _Style
colorama_stub.init = _init
sys.modules['colorama'] = colorama_stub

from gpu_etl_pipeline.log_cleaning import _split_sampling_targets, xdf

def test_split_sampling_targets():
    df = xdf.DataFrame({
        'crlevel': ['1', 'unknown', '2', '3', 'none'],
        'is_attack': [1, 1, 0, 1, 0]
    })
    targets, others = _split_sampling_targets(df)
    assert list(targets.index) == [0, 3]
    assert list(others.index) == [1, 2, 4]
