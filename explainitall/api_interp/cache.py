import hashlib
import json
import sqlite3
import threading
import time
from typing import Optional


def make_cache_key(payload: dict, sample_id: int = 0) -> str:
    """
    Ключ кеша: хеш запроса вместе с номером выборки.
    Разные sample_id позволяют хранить несколько независимых ответов на один запрос
    (нужно для оценки шума при temperature > 0).
    """
    canonical = json.dumps({'payload': payload, 'sample_id': sample_id}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class ResponseCache:
    """
    Кеш ответов API в SQLite.
    Внимание: промпты и ответы хранятся на диске в открытом виде.
    Для чувствительных данных используйте ':memory:' или не используйте кеш.
    """

    def __init__(self, path: str = ':memory:'):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS responses ('
            'key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL, created INTEGER NOT NULL)'
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute('SELECT response FROM responses WHERE key = ?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, request: dict, response: dict):
        with self._lock:
            self._conn.execute(
                'INSERT OR REPLACE INTO responses (key, request, response, created) VALUES (?, ?, ?, ?)',
                (key, json.dumps(request, ensure_ascii=False), json.dumps(response, ensure_ascii=False),
                 int(time.time())))
            self._conn.commit()

    def __len__(self):
        with self._lock:
            return self._conn.execute('SELECT COUNT(*) FROM responses').fetchone()[0]

    def close(self):
        with self._lock:
            self._conn.close()
