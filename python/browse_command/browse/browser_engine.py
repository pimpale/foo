import asyncio
from dataclasses import dataclass
from typing import Literal
from playwright.async_api import async_playwright, Playwright, ViewportSize
from .observation_processor import get_element_center, process


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


@dataclass
class ForwardCommand:
    pass


BrowserCommand = (
    GotoCommand
    | ClickCommand
    | TypeCommand
    | ScrollCommand
    | BackCommand
    | ForwardCommand
)


class BrowserEngine:
    def __init__(self, playwright: Playwright, viewport_size: ViewportSize):
        self.playwright = playwright
        self.viewport_size = viewport_size

    async def setup(self):
        self.browser = await self.playwright.chromium.launch(
            headless=False,
        )
        self.context = await self.browser.new_context(viewport=self.viewport_size)
        self.page = await self.context.new_page()
        self.cdpsession = await self.context.new_cdp_session(self.page)

    async def do(self, command: BrowserCommand):
        _, obs_nodes = await self.observation_processor.process(self.page, self.cdpsession)
        match command:
            case GotoCommand(url):
                await self.page.goto(url)
            case ClickCommand(id):
                x, y = get_element_center(obs_nodes, id)
                await self.page.mouse.move(x, y, steps=20)
                await self.page.mouse.click(x, y)
            case TypeCommand(id, text, enter):
                x, y = get_element_center(obs_nodes, id)
                await self.page.mouse.move(x, y, steps=20)
                await self.page.mouse.click(x, y)
                focused = await self.page.locator("*:focus").all()
                if focused == []:
                    raise ValueError("Element was not focusable")
                text_input = focused[0]
                # clear
                await text_input.clear()
                if enter:
                    text += "\n"
                await text_input.type(text, delay=100)
            case ScrollCommand(direction):
                await self.page.evaluate(
                    f"window.scrollBy(0, {'-100' if direction == 'up' else '100'})"
                )
            case BackCommand():
                await self.page.go_back()
            case ForwardCommand():
                await self.page.go_forward()
                
    async def observe(self) -> str:
        content, _ = await process(self.page, self.cdpsession)
        return content
