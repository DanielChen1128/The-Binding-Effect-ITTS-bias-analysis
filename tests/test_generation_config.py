import json
import os
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import generate_wav
from generate_wav import ParlerTTSGenerator, preflight, resolve_model_config


class GenerationConfigTests(unittest.TestCase):
    def test_public_parler_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_model_config("parler-mini")
        self.assertEqual(config["model_id"], "parler-tts/parler-tts-mini-v1")

    def test_cli_precedes_environment_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"models": {"parler-mini": {"model_id": "file"}}}), encoding="utf-8")
            with patch.dict(os.environ, {"BINDING_PARLER_MINI_MODEL": "environment"}):
                config = resolve_model_config("parler-mini", model_id="cli", config_path=str(path))
        self.assertEqual(config["model_id"], "cli")

    def test_preflight_reports_concrete_external_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "backend"
            model = root / "model"
            backend.mkdir()
            model.mkdir()
            with patch("generate_wav.importlib.util.find_spec", return_value=object()):
                errors = preflight("promptttspp", {
                    "model_id": str(model), "backend_path": str(backend),
                })
        self.assertTrue(any("demo.yaml" in error for error in errors))
        self.assertTrue(any("last.ckpt" in error for error in errors))

    def test_batch_reports_item_failures(self):
        generator = ParlerTTSGenerator.__new__(ParlerTTSGenerator)
        generator.model_name = "parler-mini"
        generator.output_dir = Path("unused")
        generator.skip_existing = False
        generator.generate_single = lambda item: item["id"] != "bad"
        with patch.object(generate_wav, "tqdm", side_effect=lambda data, **kwargs: data, create=True):
            failed = generator.batch_generate([{"id": "ok"}, {"id": "bad"}])
        self.assertEqual(failed, 1)

    def test_main_returns_nonzero_for_partial_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.json"
            prompts.write_text(json.dumps([{
                "id": "1", "description": "voice", "prompt_text": "hello",
            }]), encoding="utf-8")
            generator = Mock()
            generator.batch_generate.return_value = 1
            argv = ["generate_wav.py", "--model", "parler-mini", "--json", str(prompts), "--output", directory]
            with patch.object(sys, "argv", argv), \
                    patch("generate_wav.preflight", return_value=[]), \
                    patch("generate_wav.ParlerTTSGenerator", return_value=generator):
                result = generate_wav.main()
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
