import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console
from rich.text import Text


def draw_graph_image(graph, filename="graph.png"):
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=1500,
        font_size=14,
    )
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved as {filename}")


def ascii_graph_rich(graph):
    # Positions (x,y) als float in [0,1]
    pos = nx.spring_layout(graph, seed=42)

    # Rastergröße für ASCII-Ausgabe
    width, height = 40, 20

    # Leeres Raster mit Leerzeichen
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Knotenpositionen auf Raster abbilden
    node_positions = {}
    for node, (x, y) in pos.items():
        col = int(x * (width - 1))
        row = int((1 - y) * (height - 1))  # y invertieren, da 0 oben in Raster
        # Clippen, um Indexfehler zu verhindern
        row = max(0, min(height - 1, row))
        col = max(0, min(width - 1, col))
        node_positions[node] = (row, col)

    # Knoten als Buchstaben oder Zahlen eintragen
    for node, (r, c) in node_positions.items():
        label = str(node)
        for i, ch in enumerate(label):
            if 0 <= r < height and 0 <= c + i < width:
                grid[r][c + i] = ch

    # Kanten zeichnen (nur einfache Linien vertikal/horizontal)
    for u, v in graph.edges():
        r1, c1 = node_positions[u]
        r2, c2 = node_positions[v]

        # Einfacher Fall: gleiche Zeile -> horizontale Linie
        if r1 == r2:
            for cc in range(min(c1, c2) + 1, max(c1, c2)):
                grid[r1][cc] = "-"
        # gleiche Spalte -> vertikale Linie
        elif c1 == c2:
            for rr in range(min(r1, r2) + 1, max(r1, r2)):
                grid[rr][c1] = "|"
        else:
            # Diagonale Linien: hier sehr simpel "L-förmig"
            # horizontal zuerst dann vertikal
            for cc in range(min(c1, c2) + 1, max(c1, c2)):
                grid[r1][cc] = "-"
            for rr in range(min(r1, r2) + 1, max(r1, r2)):
                grid[rr][c2] = "|"

    # Grid in Text umwandeln
    lines = ["".join(row) for row in grid]

    # Rich-Console Ausgabe: Knoten blau, Kanten gelb
    console = Console()
    for line in lines:
        text = Text()
        for ch in line:
            if ch.isalnum():
                text.append(ch, style="bold blue")
            elif ch in "-|":
                text.append(ch, style="yellow")
            else:
                text.append(ch)
        console.print(text)


if __name__ == "__main__":
    # Beispielgraph
    G = nx.Graph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
            ("C", "E"),
            ("E", "F"),
            ("D", "F"),
        ]
    )

    draw_graph_image(G)  # Optional: PNG speichern
    ascii_graph_rich(G)  # ASCII + rich Ausgabe
