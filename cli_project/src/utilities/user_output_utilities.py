from colorama import Fore


class UserOutputUtilities:
    # TODO __init__

    @classmethod
    def print_header(cls, msg: str, color: Fore = Fore.RESET):
        formatted_msg = f"\n\n=== {msg} ==="
        colored_msg = cls.color_string_into_color(formatted_msg, color)
        print(colored_msg)

    @classmethod
    def print_sub_header(cls, msg: str, color: Fore = Fore.RESET):
        formatted_msg = f"\n --- {msg} ---"
        colored_msg = cls.color_string_into_color(formatted_msg, color)
        print(colored_msg)

    @classmethod
    def print_text(cls, msg: str, level: int = 1, color: Fore = Fore.RESET):
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
        return color + string + Fore.RESET
