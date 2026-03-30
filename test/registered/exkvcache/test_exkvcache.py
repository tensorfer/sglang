"""
E2E tests for ExKVCache Storage functionality.
Usage:
    python3 -m pytest test/registered/exkvcache/test_exkvcache.py -v
"""

import os
import random
import tempfile
import time
import unittest
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.explicit_kvcache_kit import ExKVCacheFileBackend
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    flush_cache_with_retry,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=200, stage="stage-b", runner_config="2-gpu-large")
register_amd_ci(est_time=526, stage="stage-b", runner_config="2-gpu-large-amd")


class ExKVCacheStorageBaseMixin:
    """Base mixin class with common setup and utilities"""

    # Constants for test configuration
    DEFAULT_EXKVCACHE_ID = "test_exkvcache"
    DEFAULT_TOKEN_LENGTH = 1024
    CACHE_HIT_THRESHOLD = 800
    PROMPT_TOKEN_LENGTH = 768
    MAX_TOKENS = 150

    @classmethod
    def setUpClass(cls):
        """Set up test environment and launch server once for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST

        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)

        # Prepare tokenizer for prompt generation
        cls.tokenizer = get_tokenizer(cls.model)

        # Launch server with ExKVCache enabled and cache report
        cls.process = cls._launch_server_with_exkvcache()
        cls._wait_for_server_ready(process=cls.process)

        print(f"Test server launched successfully at {cls.base_url}")
        print(f"Cache directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

        import shutil

        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _get_model_name(cls):
        """Get model name for the test configuration - override in subclasses"""
        return DEFAULT_MODEL_NAME_FOR_TEST

    @classmethod
    def _get_base_server_args(cls):
        """Get base server arguments - can be extended in subclasses"""
        return {
            "--enable-explicit-kvcache": True,
            "--mem-fraction-static": 0.85,
            "--page-size": 64,
            "--enable-cache-report": True,
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        return {}, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    @classmethod
    def _launch_server_with_exkvcache(cls):
        """Launch server with ExKVCache enabled"""

        additional_server_args, env_vars = cls._get_additional_server_args_and_env()
        env_vars["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
        server_args = cls._get_base_server_args()
        if additional_server_args:
            server_args.update(additional_server_args)

        final_server_args = []
        for k, v in server_args.items():
            if isinstance(v, bool):
                final_server_args.append(str(k))
            else:
                final_server_args.append(str(k))
                final_server_args.append(str(v))

        print(f"final_server_args: {final_server_args}")

        env_vars = {
            **os.environ,
            **env_vars,
        }

        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=final_server_args,
            env=env_vars,
        )

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 60, process=None) -> bool:
        """Wait for server to be ready"""
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=timeout,
            process=process,
        )
        return True

    def send_request_with_exkvcache(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.0,
        kvcache_params: Optional[Dict] = None,
        timeout: int = 60,
    ) -> Dict:
        """Send a generate request and return response with proper error handling"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
                "kvcache_params": kvcache_params,
            },
            timeout=timeout,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def get_cached_tokens(self, response_json: Dict) -> int:
        """Extract cached tokens count from /generate response"""
        meta = response_json.get("meta_info", {})
        return int(meta.get("cached_tokens", 0))

    def flush_cache(self) -> bool:
        """Flush device cache to force remote storage access."""
        return flush_cache_with_retry(self.base_url)

    def gen_prompt(self, token_num: int) -> str:
        """Generate a random prompt of specified token length using tokenizer vocabulary."""
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def _extract_text_from_response(self, response: Dict) -> str:
        """Extract text from response with validation"""
        text = response.get("text")
        self.assertIsNotNone(text, "Response should contain text")
        self.assertIsInstance(text, str, "text should be a string")
        return text

    def _extract_kvcache_params_from_response(self, response: Dict) -> Dict:
        """Extract kvcache_params from response with validation"""
        kvcache_params = response.get("kvcache_params")
        self.assertIsNotNone(kvcache_params, "Response should contain kvcache_params")
        self.assertIsInstance(kvcache_params, dict, "kvcache_params should be a dict")
        return kvcache_params

    def trigger_offloading_and_flush(self):
        """Helper method to trigger offloading and flush cache"""
        # Trigger offloading
        self.send_request_with_exkvcache(self.gen_prompt(1), max_tokens=150)

        # Flush device cache to force remote storage access
        time.sleep(2)
        self.assertTrue(self.flush_cache(), "Cache flush should succeed")

    def test_basic_backup_and_prefetch(self):
        """Test storage and retrieval of large context through explicit kvcache"""
        print(
            "\n=== Testing Large Context Cache Storage & Retrieval by Explicit KVCache ==="
        )

        # Generate substantial context that will be cached
        base_prompt = self.gen_prompt(self.PROMPT_TOKEN_LENGTH)

        file_backend = ExKVCacheFileBackend(self.DEFAULT_EXKVCACHE_ID, self.temp_dir)

        # Prepare explicit kvcache file
        exkvcache_file = file_backend.prepare_exkvcache_file(self.DEFAULT_EXKVCACHE_ID)
        print(f"Explicit KVCache file created: {exkvcache_file}")

        # Populate cache with initial request
        print("Step 1: Populating cache with large context...")
        kvcache_params1 = file_backend.kvcache_params(
            self.DEFAULT_TOKEN_LENGTH, self.DEFAULT_EXKVCACHE_ID
        )
        response1 = self.send_request_with_exkvcache(
            base_prompt, kvcache_params=kvcache_params1
        )

        self.assertIsNotNone(response1, "First response should not be None")

        # Extract data from first response
        generated_text = self._extract_text_from_response(response1)
        resp_kvcache_params = self._extract_kvcache_params_from_response(response1)
        file_backend.update_kvcache(resp_kvcache_params["fresh_kvcache"])

        # Flush device cache to force remote storage access
        self.trigger_offloading_and_flush()

        # Test cache hit from explicit kvcache
        print("Step 2: Testing cache hit from explicit kvcache...")
        kvcache_params2 = file_backend.kvcache_params(
            self.DEFAULT_TOKEN_LENGTH, self.DEFAULT_EXKVCACHE_ID
        )

        start_time = time.time()
        response2 = self.send_request_with_exkvcache(
            base_prompt + generated_text, kvcache_params=kvcache_params2
        )
        retrieval_time = time.time() - start_time

        self.assertIsNotNone(response2, "Second response should not be None")

        cached_tokens = self.get_cached_tokens(response2)
        print(
            f"Remote cache retrieval time: {retrieval_time:.3f}s, cached_tokens={cached_tokens}"
        )

        # Assert cached tokens indicate a remote hit
        self.assertGreater(
            cached_tokens,
            self.CACHE_HIT_THRESHOLD,
            f"Expected at least {self.CACHE_HIT_THRESHOLD} cached tokens for remote hit, got {cached_tokens}",
        )


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestExKVCache(ExKVCacheStorageBaseMixin, CustomTestCase):
    """Test class for explicit kvcache functionality"""

    pass


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestExKVCacheWithHiCache(ExKVCacheStorageBaseMixin, CustomTestCase):
    """Test class for explicit kvcache functionality with hierarchical cache"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        return {"--enable-hierarchical-cache": True}, {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
