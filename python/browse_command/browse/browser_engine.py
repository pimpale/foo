import asyncio
from dataclasses import dataclass
from typing import Literal
from playwright.async_api import async_playwright, Playwright, ViewportSize
from .observation_processor import TextObservationProcessor


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
        self.observation_processor = TextObservationProcessor(
            current_viewport_only=True, viewport_size=viewport_size
        )

    async def setup(self):
        self.browser = await self.playwright.chromium.launch(
            headless=False,
        )
        self.context = await self.browser.new_context(viewport=self.viewport_size)
        self.page = await self.context.new_page()
        self.cdpsession = await self.context.new_cdp_session(self.page)

    async def do(self, command: BrowserCommand):
        match command:
            case GotoCommand(url):
                await self.page.goto(url)
            case ClickCommand(id):
                x, y = self.observation_processor.get_element_center(id)
                await self.page.mouse.move(x, y, steps=20)
                await self.page.mouse.click(x, y)
            case TypeCommand(id, text, enter):
                x, y = self.observation_processor.get_element_center(id)
                await self.page.mouse.move(x, y, steps=20)
                await self.page.mouse.click(x, y)
                await self.page.keyboard.type(text, delay=100)
                if enter:
                    await self.page.keyboard.press("Enter")
            case ScrollCommand(direction):
                await self.page.evaluate(
                    f"window.scrollBy(0, {'-100' if direction == 'up' else '100'})"
                )
            case BackCommand():
                await self.page.go_back()
            case ForwardCommand():
                await self.page.go_forward()
                
    async def observe(self) -> str:
        await self.page.wait_for_load_state()
        return await self.observation_processor.process(self.page, self.cdpsession)
