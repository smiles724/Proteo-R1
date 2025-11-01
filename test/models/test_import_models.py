import unittest

from lmms_engine.models import ModelFactory


class TestModels(unittest.TestCase):
    def test_model_import(self):
        config = {
            "model_name_or_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            "model_class": "LlavaOnevisionForConditionalGeneration",
        }

        model = ModelFactory.create_model(config["model_class"])
        model = model.from_pretrained(config["model_name_or_path"])


if __name__ == "__main__":
    unittest.main()
