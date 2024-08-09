import asyncio
from dataclasses import dataclass
from typing import Literal
from playwright.async_api import async_playwright, Playwright, ViewportSize
from .observation_processor import get_element_center, process
import math

@dataclass
class NoOpCommand:
    pass


@dataclass
class GotoCommand:
    url: str


@dataclass
class ClickCommand:
    id: str


@dataclass
class TypeCommand:
    id: str
    text: str
    enter: bool


@dataclass
class ScrollCommand:
    direction: Literal["up", "down"]


@dataclass
class NavigateCommand:
    direction: Literal["back", "forward"]


@dataclass
class ReloadCommand:
    pass


BrowserCommand = (
    NoOpCommand
    | GotoCommand
    | ClickCommand
    | TypeCommand
    | ScrollCommand
    | NavigateCommand
    | ReloadCommand
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
        content, obs_nodes = await process(self.page, self.cdpsession)
        print(content)
        match command:
            case NoOpCommand():
                pass
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
            case NavigateCommand(direction):
                match direction:
                    case "back":
                        await self.page.go_back()
                    case "forward":
                        await self.page.go_forward()
            case ReloadCommand():
                await self.page.reload()

    async def scroll_percentage(self) -> float:
        return await self.page.evaluate(
            "(document.documentElement.scrollTop + document.body.scrollTop) / (document.documentElement.scrollHeight - document.documentElement.clientHeight) * 100"
        )

    async def user_friendly_observation(self) -> str:
        content, _ = await process(self.page, self.cdpsession)

        scroll_percentage = await self.scroll_percentage()

        scroll_text = "You are viewing the entire page." if math.isnan(scroll_percentage) else f"You are only viewing part of the page. Scroll percentage: {scroll_percentage:.2f}%" 

        return f"{scroll_text}\n\nContent:\n\n{content}"


    async def user_friendly_error(self, e: ValueError) -> str:
        observation = await self.user_friendly_observation()
        return f"Error: {e.args[0]}\n\n{observation}"
