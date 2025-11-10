import colorama

from cli_launcher.cli import CLI


def main() -> None:
    colorama.init()

    cli: CLI = CLI()
    cli.start()


if __name__ == "__main__":
    main()
