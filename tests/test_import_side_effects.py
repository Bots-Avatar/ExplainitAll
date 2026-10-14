import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("transformers")
pytest.importorskip("nltk")

ROOT = Path(__file__).resolve().parents[1]


def test_imports_do_not_download_or_create_files(tmp_path):
    code = textwrap.dedent("""
        # На Windows импорт nltk раньше torch может обрушить процесс из-за порядка загрузки DLL
        import torch
        import nltk

        def forbidden(*args, **kwargs):
            raise RuntimeError('nltk.download при импорте модуля')

        nltk.download = forbidden

        import explainitall.metrics.RougeAndPPL.create_database
        import explainitall.QA.interp_qa.KNNWithGenerative
    """)
    env = dict(os.environ, PYTHONPATH=str(ROOT))

    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_get_engine_creates_table(tmp_path):
    import sqlalchemy
    from explainitall.metrics.RougeAndPPL.create_database import get_engine

    engine = get_engine(str(tmp_path / 'metrics.sqlite'))

    assert sqlalchemy.inspect(engine).get_table_names() == ['data']
