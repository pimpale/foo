# my_package/cli.py

import click

@click.command()
@click.argument('name')
def greet(name):
    """Simple program that greets NAME."""
    click.echo(f"Hello, {name}!")

if __name__ == '__main__':
    greet()