import click

@click.command()
@click.argument('url')
def browse_goto(url: str) -> None:
    """Goes to the url URL"""
    click.echo(f"Hello, {url}!")
