from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.tree import Tree
from rich.layout import Layout
import time

console = Console()


def make_tree(depth):
    t = Tree("🌲 Dynamischer Baum")
    for i in range(depth):
        t.add(f"Zweig {i}")
    return t


def make_layout(tree):
    layout = Layout()
    layout.split_column(
        Layout(Panel("=== Meine Überschrift ===\nStatus: OK", title="Header"), size=3),
        Layout(Panel(tree, title="Baum")),
    )
    return layout


def main():
    with Live(console=console, refresh_per_second=4) as live:
        for i in range(8):
            tree = make_tree(i)
            layout = make_layout(tree)
            live.update(layout)
            time.sleep(1)


if __name__ == "__main__":
    main()
