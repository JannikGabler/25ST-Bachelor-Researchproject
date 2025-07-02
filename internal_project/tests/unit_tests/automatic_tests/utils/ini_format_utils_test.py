import textwrap
import unittest

from exceptions.duplicate_value_error import DuplicateError
from exceptions.format_error import FormatError
from utils.ini_format_utils import INIFormatUtils


class TestSplitIntoEntries(unittest.TestCase):
    def test_single_line_values(self):
        input_string: str = textwrap.dedent("""
            a=1 + 9
            b="aas"
            c=5as745as
        """)

        result: list[str] = INIFormatUtils.split_into_entries(input_string)
        expected: list[str] = ["a=1 + 9", "b=\"aas\"", "c=5as745as"]

        self.assertEqual(expected, result)


    def test_single_value_over_multiple_lines(self):
        input_string: str = textwrap.dedent("""
            a=1 + 9
                + 7
                - 6
                 + 2
        """)

        result: list[str] = INIFormatUtils.split_into_entries(input_string)
        expected: list[str] = ["a=1 + 9\n+ 7\n- 6\n + 2"]

        self.assertEqual(expected, result)


    def test_multiple_values_over_multiple_lines(self):
        input_string: str = textwrap.dedent("""
            a = 1 + 9
                + 7
            b = "asasasmsa\\
                asmjasi4as5asasas"
        """)

        result: list[str] = INIFormatUtils.split_into_entries(input_string)
        expected: list[str] = ["a = 1 + 9\n+ 7", "b = \"asasasmsa\\\nasmjasi4as5asasas\""]

        self.assertEqual(expected, result)


    def test_empty_lines(self):
        input_string: str = textwrap.dedent("""
        
            a = 1 + 9
                + 7
                
            b = "asasasmsa\\
            
                asmjasi4as5asasas"
                
        """)

        result: list[str] = INIFormatUtils.split_into_entries(input_string)
        expected: list[str] = ["a = 1 + 9\n+ 7", "b = \"asasasmsa\\\n\nasmjasi4as5asasas\""]

        self.assertEqual(expected, result)


    def test_beginning_with_a_opened_line(self):
        input_string: str = textwrap.dedent("""
                + 7

            b = "asasasmsa\\
                asmjasi4as5asasas"
        """)

        with self.assertRaises(FormatError):
            INIFormatUtils.split_into_entries(input_string)


    def test_comment(self):
        input_string: str = textwrap.dedent("""
            # This is a comment
            a = 1 + 9
                + 7
                - 6
                # Should not be removed
            # This is a comment
            b = 3
            # This a comment
        """)

        result: list[str] = INIFormatUtils.split_into_entries(input_string)
        expected: list[str] = ["a = 1 + 9\n+ 7\n- 6\n# Should not be removed", "b = 3"]

        self.assertEqual(expected, result)


    def test_section(self):
        input_string: str = textwrap.dedent("""
            [Section]
            a = 1 + 9
                + 7
                - 6
                [Should not be removed]
            [Section]
            b = 3
            [Section]
        """)

        result: list[str] = INIFormatUtils.split_into_entries(input_string)
        expected: list[str] = ["a = 1 + 9\n+ 7\n- 6\n[Should not be removed]", "b = 3"]

        self.assertEqual(expected, result)



class SplitEntriesIntoKeyValuePairs(unittest.TestCase):
    def test_simple_content(self):
        input_string: str = textwrap.dedent("""\
            name="TestPipeline"
            data_type=jnp.float32
            node_count=5
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"", "data_type": "jnp.float32", "node_count": "5"}

        self.assertEqual(expected, result)


    def test_whitespace_1(self):
        input_string: str = textwrap.dedent("""\
            name ="TestPipeline"  
            data_type= jnp.float32  

            node_count=5  
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"", "data_type": "jnp.float32", "node_count": "5"}

        self.assertEqual(expected, result)


    def test_whitespace_2(self):
        input_string: str = textwrap.dedent("""\
            name ="TestPipeline"  
            data_type= jnp.float32  
                
                + hallo 
                
            node_count=5  
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"", "data_type": "jnp.float32\n\n+ hallo", "node_count": "5"}

        self.assertEqual(expected, result)


    def test_comment_1(self):
        input_string: str = textwrap.dedent("""\
            # Comment 1
            name="TestPipeline"
            # Comment 2
            data_type=jnp.float32
            # Comment 3
            node_count=5
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"", "data_type": "jnp.float32", "node_count": "5"}

        self.assertEqual(expected, result)


    def test_comment_2(self):
        input_string: str = textwrap.dedent("""\
            # Comment 1
            name="TestPipeline"
            # Comment 2
            data_type=jnp.float32
            node_count=5
            # Comment 3
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"", "data_type": "jnp.float32", "node_count": "5"}

        self.assertEqual(expected, result)


    def test_section(self):
        input_string: str = textwrap.dedent("""\
            [AAas45]
            name="TestPipeline"
            data_type=jnp.float32
            [AAas45]
            node_count=5
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"", "data_type": "jnp.float32", "node_count": "5"}

        self.assertEqual(expected, result)


    def test_invalid_line(self):
        input_string: str = textwrap.dedent("""\
            name="TestPipeline"
            data_type is jnp.float32
            node_count=5
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)

        with self.assertRaises(FormatError):
            INIFormatUtils.split_entries_into_key_value_pairs(entry_list)


    def test_duplicate_key(self):
        input_string: str = textwrap.dedent("""\
            name="TestPipeline"
            data_type=jnp.float32
            name=5
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)

        with self.assertRaises(DuplicateError):
            INIFormatUtils.split_entries_into_key_value_pairs(entry_list)


    def test_multiple_line_entry(self):
        input_string: str = textwrap.dedent("""\
            name="TestPipeline"
            §tree=Tree(\"\"\"
                root
                child1
                child1.2
                child2
                \"\"\")
            data_type=jnp.float32
            node_count=5
            """)

        entry_list: list[str] = INIFormatUtils.split_into_entries(input_string)
        result: dict[str, str] = INIFormatUtils.split_entries_into_key_value_pairs(entry_list)
        expected: dict[str, str] = {"name": "\"TestPipeline\"",
                                    "§tree": "Tree(\"\"\"\nroot\nchild1\nchild1.2\nchild2\n\"\"\")",
                                    "data_type": "jnp.float32", "node_count": "5"}

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
