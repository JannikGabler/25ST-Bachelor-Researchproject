from colorama import Fore


class UserOutputUtilities:
    """
    Utility class for printing formatted and colored text output to the console. Provides methods for headers, sub-headers,
    indented text, and coloring strings.
    """


    @classmethod
    def print_header(cls, msg: str, color: Fore = Fore.RESET):
        """
        Print a header with surrounding markers.

        Args:
            msg (str): The message to print as header.
            color (Fore, optional): The text color. Defaults to Fore.RESET.
        """

        formatted_msg = f"\n\n=== {msg} ==="
        colored_msg = cls.color_string_into_color(formatted_msg, color)
        print(colored_msg)


    @classmethod
    def print_sub_header(cls, msg: str, color: Fore = Fore.RESET):
        """
        Print a sub-header with surrounding markers.

        Args:
            msg (str): The message to print as sub-header.
            color (Fore, optional): The text color. Defaults to Fore.RESET.
        """

        formatted_msg = f"\n --- {msg} ---"
        colored_msg = cls.color_string_into_color(formatted_msg, color)
        print(colored_msg)


    @classmethod
    def print_text(cls, msg: str, level: int = 1, color: Fore = Fore.RESET):
        """
        Print indented multi-line text.

        Args:
            msg (str): The text to print.
            level (int, optional): Indentation level. Defaults to 1.
            color (Fore, optional): The text color. Defaults to Fore.RESET.
        """

        lines: list[str] = msg.splitlines()

        if msg.endswith("\n"):
            lines.append("")

        indent: str = " " * (3 * level - 2)

        for i, line in enumerate(lines):
            lines[i] = f"{indent}{line}"

        colored_msg = cls.color_string_into_color("\n".join(lines), color)
        print(colored_msg)


    @classmethod
    def color_string_into_color(cls, string: str, color: Fore) -> str:
        """
        Appply a given color to a string.

        Args:
            string (str): The text to colorize.
            color (Fore): The color from colorama. Fore to apply.

        Returns:
            str: The colorized string.
        """

        return color + string + Fore.RESET
