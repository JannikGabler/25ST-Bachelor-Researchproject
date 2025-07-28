import colorama

from cli_launcher.new_cli import CLI


def main() -> None:
    colorama.init()

    cli: CLI = CLI()
    cli.start()



if __name__ == '__main__':
    main()