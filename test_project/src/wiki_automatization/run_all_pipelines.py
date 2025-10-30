import os
import subprocess
import shutil
from pathlib import Path

# Root-Verzeichnisse
INPUT_ROOT = Path("research/src")
OUTPUT_ROOT = Path("output")

def find_valid_folders(root: Path):
    """Findet alle Unterordner, die beide INI-Dateien enthalten."""
    for dirpath, dirnames, filenames in os.walk(root):
        if {"pipeline_input.ini", "pipeline_configuration.ini"} <= set(filenames):
            yield Path(dirpath)

def read_input_files(folder: Path) -> tuple[str, str]:
    input_file = folder / "pipeline_input.ini"
    config_file = folder / "pipeline_configuration.ini"
    with open(input_file, "r") as f:
        input_text = f.read()
    with open(config_file, "r") as f:
        config_text = f.read()
    return input_text, config_text


def run_interpolation_pipeline(folder: Path):
    """Führt interpolation_pipeline aus und gibt die relevante Konsolenausgabe zurück."""
    print(f"Starte Pipeline für: {folder}")
    # Umgebung mit verdoppelter Terminalbreite
    env = os.environ.copy()
    env["COLUMNS"] = "240"  # typischerweise doppelt so breit wie Standard-Terminals
    env["LINES"] = "60"     # optional, falls das Tool Zeilenhöhe nutzt

    result = subprocess.run(
        ["interpolation_pipeline", str(folder), "--skip-trust-warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    output_lines = result.stdout.splitlines()
    # Finde den Marker
    marker = "#####################################################################################################"
    start_index = None
    for i, line in enumerate(output_lines):
        if marker in line:
            start_index = max(i - 1, 0)
            break
    if start_index is not None:
        filtered_output = "\n".join(output_lines[start_index:])
    else:
        filtered_output = result.stdout  # falls Marker fehlt, alles speichern
    return filtered_output

def copy_plot_files(src_folder: Path, dest_folder: Path):
    """Kopiert beide plot.png Dateien in den Ausgabepfad."""
    plots = [
        OUTPUT_ROOT / "runs/latest/components/interpolant plotter/plots/plot.svg",
        OUTPUT_ROOT / "runs/latest/components/absolute error plotter/plots/plot.svg",
        OUTPUT_ROOT / "runs/latest/components/absolute round off error plotter/plots/plot.svg",
    ]
    for i, plot in enumerate(plots):
        if plot.exists():
            target_path = dest_folder
            target_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plot, target_path / f"plot {plot.parent.parent.name}.png")
            print("Copied .png files to ", target_path)
        else:
            print(f"⚠️ Plot fehlt: {plot}")

def main():
    print("Start")
    for folder in find_valid_folders(INPUT_ROOT):
        print("Found folder ", folder)
        # relative Struktur beibehalten
        rel_path = folder.relative_to(INPUT_ROOT)
        dest_folder = OUTPUT_ROOT / rel_path
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Pipeline ausführen und Konsolenausgabe speichern
        input_text, config_text = read_input_files(folder)

        output_text: str = (
            f"## {folder.name}\n"
            "\n"
            f"### Pipeline input\n"
            "```\n"
            f"{input_text}\n"
            "```\n"
            "\n"
            f"### Pipeline configuration\n"
            "```\n"
            f"{config_text}\n"
            "```\n"
            "\n"
            f"### Output\n"
            "```\n"
            f"{run_interpolation_pipeline(folder)}\n"
            "```\n"
            "\n"
            "### Plots\n"
        )

        with open(dest_folder / "output.txt", "w") as f:
            f.write(output_text)

        # Plot-Dateien kopieren
        copy_plot_files(OUTPUT_ROOT, dest_folder)

        # output/runs/latest löschen
        latest_dir = folder / "output/runs/latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir, ignore_errors=True)

        print(f"✅ Fertig: {folder}\n---")

if __name__ == "__main__":
    main()
