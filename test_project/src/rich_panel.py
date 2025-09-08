# from rich.console import Console
# from rich.panel import Panel
# from rich.text import Text
# from rich.live import Live
#
# class RichLivePanelHelper:
#     _console = Console()
#     _title = None
#     _content = None
#     _live = None
#     _live_context = None
#
#     @staticmethod
#     def open_panel(title: str):
#         if RichLivePanelHelper._live_context is not None:
#             raise RuntimeError("Ein Panel ist bereits aktiv.")
#
#         RichLivePanelHelper._title = title
#         RichLivePanelHelper._content = Text()
#         RichLivePanelHelper._live_context = Live(
#             RichLivePanelHelper._render(),
#             console=RichLivePanelHelper._console,
#             refresh_per_second=10,
#             auto_refresh=False
#         )
#         RichLivePanelHelper._live_context.__enter__()  # explizit öffnen
#         RichLivePanelHelper._live = RichLivePanelHelper._live_context
#
#     @staticmethod
#     def _render():
#         return Panel(RichLivePanelHelper._content, title=RichLivePanelHelper._title)
#
#     @staticmethod
#     def write_line_in_panel(line: str):
#         if RichLivePanelHelper._live is None:
#             raise RuntimeError("Kein aktives Live-Panel.")
#
#         RichLivePanelHelper._content.append(line + "\n")
#         RichLivePanelHelper._live.update(RichLivePanelHelper._render())
#
#     @staticmethod
#     def close_panel():
#         if RichLivePanelHelper._live_context is None:
#             raise RuntimeError("Kein aktives Live-Panel.")
#
#         RichLivePanelHelper._live_context.__exit__(None, None, None)
#
#         # Zustand zurücksetzen
#         RichLivePanelHelper._title = None
#         RichLivePanelHelper._content = None
#         RichLivePanelHelper._live = None
#         RichLivePanelHelper._live_context = None


import time

from utilities.rich_utilities import RichUtilities


def main():
    RichUtilities.open_panel("Live-Protokoll")
    for i in range(5):
        RichUtilities.write_lines_in_panel(f"[green]Schritt {i + 1} abgeschlossen[/green]")
        time.sleep(1)
    RichUtilities.close_panel()

if __name__ == "__main__":
    main()
