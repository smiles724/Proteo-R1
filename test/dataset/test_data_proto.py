import json
import os
import pathlib
import unittest

from pydantic import ValidationError

from lmms_engine.protocol import SFTChatData

current_dir = pathlib.Path().resolve()
data_folder = current_dir.parent.parent / "examples" / "sample_json_data"


class TestDataProto(unittest.TestCase):
    def test_json_data(self):
        with open(os.path.join(str(data_folder), "lmms_engine.json"), "r") as f:
            data = json.load(f)

        for da in data:
            try:
                checked = SFTChatData(**da)
            except ValidationError as e:
                print(e.errors())


if __name__ == "__main__":
    unittest.main()
