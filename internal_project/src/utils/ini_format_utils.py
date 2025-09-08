import re
import textwrap

from exceptions.duplicate_value_error import DuplicateError
from exceptions.format_error import FormatError
from exceptions.not_instantiable_error import NotInstantiableError


class INIFormatUtils:
    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        raise NotInstantiableError(f"The class '{self.__class__.__name__}' can not be instantiated.")


    ######################
    ### Public methods ###
    ######################
    @staticmethod
    def split_into_entries(configuration_string: str) -> list[str]:
        entries: list[str] = []
        raw_lines: list[str] = configuration_string.rstrip().splitlines()
        lines_of_current_entry: list[str] = []

        for raw_line in raw_lines:
            if raw_line and not raw_line.isspace() and not re.match(r'^\s', raw_line): # Line contains content with no indent
                if not raw_line.startswith(('[', '#')):
                    INIFormatUtils._add_entry_lines_to_entry_list_(lines_of_current_entry, entries)
                    lines_of_current_entry = [raw_line.rstrip()]
            else:
                INIFormatUtils._handle_additional_entry_line_(raw_line, lines_of_current_entry)

        INIFormatUtils._add_entry_lines_to_entry_list_(lines_of_current_entry, entries) # Add the last entry

        return entries



    @staticmethod
    def split_entries_into_key_value_pairs(entry_list: list[str]) -> dict[str, str]:
        entry_dict: dict[str, str] = {}

        for entry in entry_list:
            if "=" not in entry:
                raise FormatError(f"The entry {repr(entry)} represents no valid key-value pair.")

            splittings = entry.split("=", maxsplit=1)
            key: str = splittings[0].strip()
            value: str = splittings[1].strip()

            if key in entry_dict:
                raise DuplicateError(f"There are multiple entries with the same key {repr(key)}.")
            if key.isspace():
                raise FormatError(f"The key cannot be empty.")

            entry_dict[key] = value

        return entry_dict



    #######################
    ### Private methods ###
    #######################
    @staticmethod
    def _add_entry_lines_to_entry_list_(lines_of_current_entry: list[str], entries: list[str]) -> None:
        if lines_of_current_entry:
            new_entry: str = INIFormatUtils._build_entry_from_lines_(lines_of_current_entry)
            entries.append(new_entry)



    @staticmethod
    def _build_entry_from_lines_(lines_of_entry: list[str]) -> str:
        if len(lines_of_entry) == 1:
            return lines_of_entry[0]
        else:
            non_stripped: str = '\n'.join(lines_of_entry[1:])
            stripped: str = textwrap.dedent(non_stripped)
            entry: str = lines_of_entry[0] + '\n' + stripped
            return entry.rstrip()


    @staticmethod
    def _handle_additional_entry_line_(raw_line: str, lines_of_current_entry: list[str]) -> None:
        if not lines_of_current_entry:
            if raw_line and not raw_line.isspace():
                raise FormatError("A ini configuration cannot start with an indented line.")
        else:
            lines_of_current_entry.append(raw_line.rstrip())


