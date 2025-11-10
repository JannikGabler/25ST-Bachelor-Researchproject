import tempfile
import textwrap
import unittest
from pathlib import Path

from exceptions.duplicate_value_error import DuplicateError
from file_handling.ini_handling.ini_file_manager import INIFileManager


class LoadFileAsKeyValuePairs(unittest.TestCase):
    def test_simple_content(self):
        file_content: bytes = textwrap.dedent(
            """\
                                                name="TestPipeline"
                                                data_type=jnp.float32
                                                node_count=5
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            result: dict[str, str] = INIFileManager.load_file_as_key_value_pairs(
                temp_file
            )
            expected: dict[str, str] = {
                "name": '"TestPipeline"',
                "data_type": "jnp.float32",
                "node_count": "5",
            }

            self.assertEqual(expected, result)

    def test_whitespace(self):
        file_content: bytes = textwrap.dedent(
            """
                               name ="TestPipeline"  
                               data_type= jnp.float32  
                               
                               node_count=5  
                               """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            result: dict[str, str] = INIFileManager.load_file_as_key_value_pairs(
                temp_file
            )
            expected: dict[str, str] = {
                "name": '"TestPipeline"',
                "data_type": "jnp.float32",
                "node_count": "5",
            }

            self.assertEqual(expected, result)

    def test_comment_1(self):
        file_content: bytes = textwrap.dedent(
            """\
                                                # Comment 1
                                                name="TestPipeline"
                                                # Comment 2
                                                data_type=jnp.float32
                                                # Comment 3
                                                node_count=5
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            result: dict[str, str] = INIFileManager.load_file(temp_file)
            expected: dict[str, str] = {
                "name": '"TestPipeline"',
                "data_type": "jnp.float32",
                "node_count": "5",
            }

            self.assertEqual(expected, result)

    def test_comment_2(self):
        file_content: bytes = textwrap.dedent(
            """\
                                                # Comment 1
                                                name="TestPipeline"
                                                # Comment 2
                                                data_type=jnp.float32
                                                node_count=5
                                                # Comment 3
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            result: dict[str, str] = INIFileManager.load_file(temp_file)
            expected: dict[str, str] = {
                "name": '"TestPipeline"',
                "data_type": "jnp.float32",
                "node_count": "5",
            }

            self.assertEqual(expected, result)

    def test_section(self):
        file_content: bytes = textwrap.dedent(
            """\
                                                [AAas45]
                                                name="TestPipeline"
                                                data_type=jnp.float32
                                                node_count=5
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            result: dict[str, str] = INIFileManager.load_file(temp_file)
            expected: dict[str, str] = {
                "name": '"TestPipeline"',
                "data_type": "jnp.float32",
                "node_count": "5",
            }

            self.assertEqual(expected, result)

    def test_invalid_line(self):
        file_content: bytes = textwrap.dedent(
            """\
                                                name="TestPipeline"
                                                data_type is jnp.float32
                                                node_count=5
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            with self.assertRaises(ValueError):
                INIFileManager.load_file(temp_file)

    def test_duplicate_key(self):
        file_content: bytes = textwrap.dedent(
            """\
                                                name="TestPipeline"
                                                data_type=jnp.float32
                                                name=5
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            with self.assertRaises(DuplicateError):
                INIFileManager.load_file(temp_file)

    def test_multiple_line_entry(self):
        file_content: bytes = textwrap.dedent(
            """
                                                name="TestPipeline"
                                                §tree=Tree(\"\"\"\\
                                                root\\
                                                child1\\
                                                child1.2\\
                                                child2\\
                                                \"\"\")
                                                data_type=jnp.float32
                                                node_count=5
                                                """
        ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file: Path = Path(temp_dir + "/temp_file.ini")

            with open(temp_file, "wb") as f:
                f.write(file_content)

            result: dict[str, str] = INIFileManager.load_file(temp_file)
            expected: dict[str, str] = {
                "name": '"TestPipeline"',
                "§tree": 'Tree("""\nroot\nchild1\nchild1.2\nchild2\n""")',
                "data_type": "jnp.float32",
                "node_count": "5",
            }

            self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
