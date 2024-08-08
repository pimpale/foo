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


BrowserCommand = GotoCommand | ClickCommand | TypeCommand | ScrollCommand | BackCommand

class BrowserEngine:
    def __init__(self, playwright: Playwright, viewport_size: ViewportSize):
        self.playwright = playwright
        self.observation_processor = TextObservationProcessor(
            current_viewport_only=True,
            viewport_size = viewport_size
        )
    
    async def setup(self):
        chromium = self.playwright.chromium # or "firefox" or "webkit".
        self.browser = await chromium.launch()
        self.page = await self.browser.new_page()
        self.cdpsession = await self.page.context.new_cdp_session(self.page)
        
    async def do(self, command: BrowserCommand):
        match command:
            case GotoCommand(url):
                await self.page.goto(url)
            case ClickCommand(id):
                await self.page.click(f"#{id}")
            case TypeCommand(id, text, enter):
                await self.page.type(f"#{id}", text)
                if enter:
                    await self.page.press(f"#{id}", "Enter")
            case ScrollCommand(direction):
                await self.page.evaluate(f"window.scrollBy(0, {'-100' if direction == 'up' else '100'})")
            case BackCommand():
                await self.page.goBack()
                
    async def observe(self) -> str:
        return await self.observation_processor.process(self.page, self.cdpsession)