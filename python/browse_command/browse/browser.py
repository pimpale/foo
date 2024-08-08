import asyncio
from playwright.async_api import async_playwright, Playwright

class Browser:
    def __init__(self, playwright: Playwright):
        self.playwright = playwright
    
    async def setup(self):
        self.chromium = self.playwright.chromium # or "firefox" or "webkit".
        self.browser = await self.chromium.launch()
        self.page = await self.browser.new_page()
        
    async def goto(self, url: str) -> str: