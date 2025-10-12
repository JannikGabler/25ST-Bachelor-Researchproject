import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from rich.console import Console


# ---------- 1) PNG rendern (ohne Rand) und Layout zurückgeben ----------
def draw_graph_image(graph, filename="graph.png", seed=42, return_pos=True):
    pos = nx.spring_layout(graph, seed=seed)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_axis_off()
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=1500,
        font_size=14,
    )
    plt.savefig(filename, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Graph saved as {filename}")
    return pos if return_pos else None


# ---------- 2) PNG -> Konsole (█/▀/▄) + Labels zentriert in Knoten ----------
def png_to_console_fullblocks_with_labels(filename, pos, label_map, max_width=120):
    # Bild vorbereiten und skalieren
    img = Image.open(filename).convert("RGBA")
    w, h = img.size
    aspect = h / w
    new_w = min(w, max_width)
    new_h = int(aspect * new_w)
    if new_h % 2 != 0:
        new_h += 1
    img = img.resize((new_w, new_h))

    # Pixel-Sichtbarkeit
    def is_visible_pixel(r, g, b, a):
        return a > 0 and not (r > 240 and g > 240 and b > 240)

    # Raster initialisieren
    rows = new_h // 2
    cols = new_w
    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    # Zeichen generieren
    for y in range(0, new_h, 2):
        r_row = y // 2
        for x in range(new_w):
            r1, g1, b1, a1 = img.getpixel((x, y))
            r2, g2, b2, a2 = img.getpixel((x, y + 1))
            up = is_visible_pixel(r1, g1, b1, a1)
            down = is_visible_pixel(r2, g2, b2, a2)
            grid[r_row][x] = "█" if up and down else "▀" if up else "▄" if down else " "

    # Koordinatentransformation von NetworkX zu Konsolenkoordinaten
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(maxx - minx, 1e-9)
    spany = max(maxy - miny, 1e-9)

    def node_to_console_coords(x, y):
        # Normalisieren und Bildpixel- zu Rasterkoordinaten
        nxn = (x - minx) / spanx
        nyn = (y - miny) / spany
        pix_x = int(round(nxn * (new_w - 1)))
        pix_y = int(round((1 - nyn) * (new_h - 1)))
        # Zeile und Spalte im char-Grid
        return pix_y // 2, pix_x

    # Labels zentriert über theoretischem Knotenmittelpunkt platzieren
    for node, (x, y) in pos.items():
        label = label_map.get(node, str(node))
        r, c = node_to_console_coords(x, y)
        label_len = len(label)
        text_r = max(0, min(rows - 1, r))
        text_c = max(0, min(cols - label_len, c - label_len // 2))
        # Umgebung freiräumen
        for dr in (-1, 0, 1):
            rr = text_r + dr
            if 0 <= rr < rows:
                for cc in range(text_c - 1, text_c + label_len + 1):
                    if 0 <= cc < cols:
                        grid[rr][cc] = " "
        # Text setzen
        for i, ch in enumerate(label):
            cc = text_c + i
            if 0 <= text_r < rows and 0 <= cc < cols:
                grid[text_r][cc] = ch

    # Konsolenausgabe
    console = Console()
    for row in grid:
        console.print("".join(row))


# Beispielnutzung
if __name__ == "__main__":
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
    pos = draw_graph_image(G, filename="graph.png", seed=42, return_pos=True)
    label_map = {
        "A": "Alpha",
        "B": "Beta",
        "C": "Gamma",
        "D": "Delta",
        "E": "Epsilon",
        "F": "Phi",
    }
    png_to_console_fullblocks_with_labels("graph.png", pos, label_map, max_width=120)
