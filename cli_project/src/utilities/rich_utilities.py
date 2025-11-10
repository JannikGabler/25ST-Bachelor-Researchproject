from __future__ import annotations

import threading
import time

from threading import Lock, Thread
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live


class RichUtilities:
    """
    Utility class for displaying and updating Rich panels in the console. Provides methods to open a panel, write styled
    lines to it, close it, and interactively query yes/no input from the user.
    """

    _REFRESH_RATE_: int = 1

    _console_: Console = Console()
    _panel_opened_: bool = False
    _current_panel_title_: str
    _current_panel_text_: Text
    _is_current_panel_empty_: bool = True
    _lock_: Lock = threading.Lock()
    _live_thread_: Thread


    @classmethod
    def open_panel(cls, title: str):
        """
        Open a new Rich panel with the given title. If a panel is already open, it will be closed first.
        Starts a background thread to continuously refresh the panel display.

        Args:
            title (str): The title to display at the top of the panel.
        """

        if cls._panel_opened_:
            cls.close_panel()

        if hasattr(cls, "_live_thread_"):
            if cls._live_thread_.is_alive():
                cls._live_thread_.join()

        with cls._lock_:
            cls._current_panel_title_ = title
            cls._current_panel_text_ = Text(no_wrap=True)
            cls._panel_opened_ = True

            cls._console_.print("")

        cls._live_thread_ = Thread(target=RichUtilities._live_worker_operation_, daemon=True)
        cls._live_thread_.start()


    @classmethod
    def _live_worker_operation_(cls):
        sleep_delay = 1 / cls._REFRESH_RATE_

        with Live(cls._render_(), refresh_per_second=cls._REFRESH_RATE_, console=cls._console_) as live:
            while cls._panel_opened_:

                with cls._lock_:
                    live.update(cls._render_())

                time.sleep(sleep_delay)


    @classmethod
    def _render_(cls):
        return Panel(cls._current_panel_text_, title=cls._current_panel_title_)


    @classmethod
    def write_lines_in_panel(cls, lines: str, style: str | None = None, indent_level: int = 0):
        """
        Write one or multiple lines into the currently opened panel.

        Args:
            lines (str): The text lines to write.
            style (str | None): Optional Rich style string for formatting text.
            indent_level (int): Indentation level, each level adds two spaces.

        Raises:
            RuntimeError: If no panel is opened.
        """

        if not cls._panel_opened_:
            raise RuntimeError("Can't write line into panel because there is no panel opened.")

        indent: str = " " * 2 * indent_level
        lines_array = lines.splitlines()

        if lines.endswith("\n"):
            lines_array.append("")

        for i, line in enumerate(lines_array):
            lines_array[i] = f"{indent}{line}"

        lines = "\n".join(lines_array)

        with cls._lock_:
            for i, line in enumerate(lines_array):
                text_line = Text(f"{indent}{line}", style=style)
                if cls._is_current_panel_empty_:
                    cls._current_panel_text_.append(text_line)
                    cls._is_current_panel_empty_ = False
                else:
                    cls._current_panel_text_.append(Text("\n"))
                    cls._current_panel_text_.append(text_line)


    @classmethod
    def close_panel(cls):
        """
        Close the currently opened panel and stop its background refresh thread. If no panel is open, nothing happens.
        """

        if cls._panel_opened_:
            cls._panel_opened_ = False

            cls._live_thread_.join()

            cls._is_current_panel_empty_ = True


    @classmethod
    def get_yes_no_input(cls, repeat_after_invalid_input: bool = True, default_yes: bool = True) -> bool | None:
        """
        Prompt the user for a yes/no answer via console input.

        Args:
            repeat_after_invalid_input (bool, optional): If True, the prompt repeats until valid input is given.
                If False, the method returns None for invalid input.
            default_yes (bool): Determines the default answer when the user presses Enter without typing anything.
                If True, an empty input counts as "yes"; if False, as "no".

        Returns:
            bool | None: True if the user enters "y" or "yes", False if the user enters "n" or "no".
                None if the input is invalid and repeat_after_invalid_input is False.
        """

        prompt = "-> "

        while True:
            user_input = Prompt.ask(prompt).lower()  # input(prompt).lower()

            if user_input == "y" or user_input == "yes" or (default_yes and user_input == ""):
                return True
            elif user_input == "n" or user_input == "no" or (not default_yes and user_input == ""):
                return False
            elif not repeat_after_invalid_input:
                return None
