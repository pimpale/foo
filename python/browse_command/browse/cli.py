from typing import Literal, Union
import click
from dataclasses import dataclass
import subprocess
import os
from multiprocessing.connection import Client, Listener
import time
import playwright
import asyncio
from .browser import Browser


@dataclass
class GotoCommand:
    url: str


@dataclass
class ClickCommand:
    id: int


@dataclass
class TypeCommand:
    id: int
    text: str
    enter: bool


@dataclass
class ScrollCommand:
    direction: Literal["up", "down"]


@dataclass
class BackCommand:
    pass


Command = GotoCommand | ClickCommand | TypeCommand | ScrollCommand | BackCommand

SERVER_ADDRESS = ("localhost", 6000)

async def browse_start_async() -> None:
    with Listener(SERVER_ADDRESS) as listener:
        with playwright.async_api.async_playwright() as playwright:
            browser = Browser(playwright)
            await browser.setup()
            while True:
                with listener.accept() as conn:
                    request: Command = conn.recv()
                    match request:
                        case GotoCommand(url):
                            conn.send(await browser.goto(url))
                        case ClickCommand(id):
                            conn.send(await browser.click(id))
                        case TypeCommand(id, text, enter):
                            conn.send(await browser.type(id, text, enter))
                        case ScrollCommand(direction):
                            conn.send(await browser.scroll(direction))
                        case BackCommand():
                            conn.send(await browser.back())

@click.command()
def browse_start() -> None:
    """Runs the server loop"""
    asyncio.run(browse_start_async())


def browse_start_nohup():
    subprocess.Popen(
        ["nohup", "browse-start"],
        stdout=open("/dev/null", "w"),
        stderr=open("/dev/null", "a"),
        preexec_fn=os.setpgrp,
    )
    time.sleep(0.2)


@click.command()
@click.argument("url")
def browse_goto(url: str) -> None:
    """Goes to the url URL"""
    browse_start_nohup()
    with Client(SERVER_ADDRESS) as conn:
        conn.send(GotoCommand(url))
        click.echo(conn.recv())


@click.command()
@click.argument("id", type=int)
def browse_click(id: int) -> None:
    """Clicks on the element ID"""
    browse_start_nohup()
    with Client(SERVER_ADDRESS) as conn:
        conn.send(ClickCommand(id))
        click.echo(conn.recv())


@click.command()
@click.argument("id", type=int)
@click.argument("text")
@click.option("--enter", is_flag=True)
def browse_type(id: int, text: str, enter: bool) -> None:
    """Types the text TEXT in the element ID"""
    browse_start_nohup()
    with Client(SERVER_ADDRESS) as conn:
        conn.send(TypeCommand(id, text, enter))
        click.echo(conn.recv())


@click.command()
@click.argument("direction", type=click.Choice(["up", "down"]))
def browse_scroll(direction: str) -> None:
    """Scrolls the page in the DIRECTION direction"""
    browse_start_nohup()
    with Client(SERVER_ADDRESS) as conn:
        conn.send(ScrollCommand(direction))
        click.echo(conn.recv())


@click.command()
def browse_back() -> None:
    """Goes back in the browser history"""
    browse_start_nohup()
    with Client(SERVER_ADDRESS) as conn:
        conn.send(BackCommand())
        click.echo(conn.recv())
