# Freesound API integration

import os
import logging
import requests
from typing import List, Dict, Optional
from pathlib import Path

class FreesoundIntegration:
    """
    Интеграция с Freesound API для поиска и загрузки бесплатных сэмплов.
    """

    def __init__(self, api_key: Optional[str] = None, download_dir: str = "Samples for AKAI/freesound"):
        self.api_key = api_key or os.getenv("FREESOUND_API_KEY")
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://freesound.org/apiv2"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            self.logger.warning("Freesound API key not provided - integration disabled")
            self.enabled = False
        else:
            self.enabled = True
            self.logger.info("Freesound integration initialized")

    def _headers(self):
        return {"Authorization": f"Token {self.api_key}"}

    async def search_samples(self, query: str, genre: str = "", max_results: int = 10) -> List[Dict]:
        """Поиск сэмплов на Freesound по запросу и жанру."""
        if not self.enabled:
            return []

        search_query = f"{query} {genre}".strip()
        self.logger.info(f"🔍 Searching Freesound: '{search_query}'")

        params = {
            "query": search_query,
            "fields": "id,name,previews,download,username,tags",
            "page_size": max_results
        }

        try:
            resp = requests.get(f"{self.base_url}/search/text/", headers=self._headers(), params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return results
        except Exception as e:
            self.logger.error(f"❌ Error searching Freesound: {e}")
            return []

    async def download_sample(self, sample_id: str) -> Optional[str]:
        """Загрузка сэмпла по ID."""
        if not self.enabled:
            return None

        try:
            # Получаем инфу о сэмпле
            meta_resp = requests.get(f"{self.base_url}/sounds/{sample_id}/", headers=self._headers())
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            download_url = meta.get("download")

            if not download_url:
                self.logger.warning(f"⚠️ No download link for sample {sample_id}")
                return None

            # Скачиваем
            file_path = self.download_dir / f"{meta['name']}.wav"
            with requests.get(download_url, headers=self._headers(), stream=True) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            self.logger.info(f"✅ Downloaded sample: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"❌ Error downloading sample {sample_id}: {e}")
            return None