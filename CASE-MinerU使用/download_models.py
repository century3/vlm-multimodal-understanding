import json
import os
import time

import requests
from modelscope import snapshot_download

# #region agent log
_DBG_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug-92fc8b.log')


def _dbg(hypothesis_id, location, message, data=None, run_id='pre-fix'):
    payload = {
        'sessionId': '92fc8b',
        'runId': run_id,
        'hypothesisId': hypothesis_id,
        'location': location,
        'message': message,
        'data': data or {},
        'timestamp': int(time.time() * 1000),
    }
    with open(_DBG_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')
# #endregion


FALLBACK_JSON_URLS = [
    'https://raw.githubusercontent.com/opendatalab/MinerU/release-1.2.2/magic-pdf.template.json',
    'https://github.com/opendatalab/MinerU/raw/release-1.2.2/magic-pdf.template.json',
    'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json',
]

EMBEDDED_TEMPLATE = {
    'bucket_info': {
        'bucket-name-1': ['ak', 'sk', 'endpoint'],
        'bucket-name-2': ['ak', 'sk', 'endpoint'],
    },
    'models-dir': '/tmp/models',
    'layoutreader-model-dir': '/tmp/layoutreader',
    'device-mode': 'cpu',
    'layout-config': {'model': 'doclayout_yolo'},
    'formula-config': {
        'mfd_model': 'yolo_v8_mfd',
        'mfr_model': 'unimernet_small',
        'enable': True,
    },
    'table-config': {
        'model': 'rapid_table',
        'sub_model': 'slanet_plus',
        'enable': True,
        'max_time': 400,
    },
    'config_version': '1.2.1',
}


def download_json(urls):
    last_error = None
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            # #region agent log
            _dbg('H1', 'download_models.py:download_json', 'template fetch attempt', {
                'url': url, 'status': response.status_code, 'ok': response.ok,
            }, run_id='post-fix')
            # #endregion
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            # #region agent log
            _dbg('H1', 'download_models.py:download_json', 'template fetch failed', {
                'url': url, 'error': type(exc).__name__, 'detail': str(exc)[:300],
            }, run_id='post-fix')
            # #endregion
    # #region agent log
    _dbg('H1', 'download_models.py:download_json', 'all remotes failed, using embedded template', {
        'last_error': str(last_error)[:300] if last_error else None,
    }, run_id='post-fix')
    # #endregion
    return json.loads(json.dumps(EMBEDDED_TEMPLATE))


def download_and_modify_json(urls, local_filename, modifications):
    if os.path.exists(local_filename):
        with open(local_filename, encoding='utf-8') as f:
            data = json.load(f)
        config_version = data.get('config_version', '0.0.0')
        # #region agent log
        _dbg('H3', 'download_models.py:download_and_modify_json', 'existing config found', {
            'path': local_filename, 'config_version': config_version,
        }, run_id='post-fix')
        # #endregion
        if config_version < '1.2.0':
            data = download_json(urls)
    else:
        # #region agent log
        _dbg('H3', 'download_models.py:download_and_modify_json', 'no existing config, will download template', {
            'path': local_filename,
        }, run_id='post-fix')
        # #endregion
        data = download_json(urls)

    for key, value in modifications.items():
        data[key] = value

    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # #region agent log
    _dbg('H5', 'download_models.py:download_and_modify_json', 'config written', {
        'path': local_filename,
        'exists': os.path.exists(local_filename),
        'size': os.path.getsize(local_filename) if os.path.exists(local_filename) else 0,
        'models_dir': data.get('models-dir'),
        'layoutreader_dir': data.get('layoutreader-model-dir'),
    }, run_id='post-fix')
    # #endregion


def _models_ready(model_root):
    yolo = os.path.join(model_root, 'models', 'Layout', 'YOLO')
    ocr_marker = os.path.join(model_root, 'models', 'OCR', 'paddleocr_torch', 'ch_PP-OCRv5_rec_infer.pth')
    layoutreader = os.path.join(model_root, 'pytorch_model.bin')
    ready = os.path.isdir(yolo) and os.path.isfile(ocr_marker) and os.path.isfile(layoutreader)
    # #region agent log
    _dbg('H2', 'download_models.py:_models_ready', 'local model presence', {
        'model_root': model_root,
        'yolo': os.path.isdir(yolo),
        'ocr_marker': os.path.isfile(ocr_marker),
        'layoutreader': os.path.isfile(layoutreader),
        'ready': ready,
    }, run_id='post-fix')
    # #endregion
    return ready


if __name__ == '__main__':
    mineru_patterns = [
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    local_model_dir = os.path.join(script_dir, 'modelscope_models')
    workspace_model_dir = os.path.join(os.path.dirname(script_dir), 'modelscope_models')
    # #region agent log
    _dbg('H2', 'download_models.py:main', 'path context', {
        'cwd': cwd,
        'script_dir': script_dir,
        'local_model_dir': local_model_dir,
        'workspace_model_dir': workspace_model_dir,
    }, run_id='post-fix')
    # #endregion

    if _models_ready(local_model_dir):
        model_root = local_model_dir
        print(f'Skip download, using existing models: {model_root}')
    elif _models_ready(workspace_model_dir):
        model_root = workspace_model_dir
        print(f'Skip download, using existing models: {model_root}')
    else:
        model_root = local_model_dir
        snapshot_download('opendatalab/PDF-Extract-Kit-1.0', allow_patterns=mineru_patterns, local_dir=model_root)
        snapshot_download('ppaanngggg/layoutreader', local_dir=model_root)

    model_dir = model_root + '/models'
    layoutreader_model_dir = model_root
    print(f'model_dir is: {model_dir}')
    print(f'layoutreader_model_dir is: {layoutreader_model_dir}')

    json_urls = FALLBACK_JSON_URLS
    config_file_name = 'magic-pdf.json'
    home_dir = os.path.expanduser('~')
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        'models-dir': model_dir,
        'layoutreader-model-dir': layoutreader_model_dir,
    }

    download_and_modify_json(json_urls, config_file, json_mods)
    print(f'The configuration file has been configured successfully, the path is: {config_file}')
