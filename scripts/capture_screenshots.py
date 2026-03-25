import asyncio
from playwright.async_api import async_playwright
import time
from pathlib import Path

async def capture_dashboard():
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Larger viewport to fit the SHAP panels
        page = await browser.new_page(viewport={"width": 1920, "height": 1400})
        
        print("Navigating to dashboard...")
        await page.goto("http://localhost:8501")
        
        # Wait for Streamlit to load the first element (the sidebar or title)
        await page.wait_for_selector("text=Blink — GPU Performance Predictor", timeout=15000)
        time.sleep(2) # extra padding for streamlit hydration
        
        print("Selecting Pre-trained Models...")
        # Click the model type dropdown (first selectbox)
        await page.click("div[data-baseweb='select'] >> nth=0")
        await page.click("text=Pre-trained Models")
        time.sleep(1)
        
        print("Selecting ResNet50...")
        # The model dropdown
        await page.click("div[data-baseweb='select'] >> nth=1")
        await page.click("text=ResNet50")
        time.sleep(1)
        
        print("Selecting Batch Size 32...")
        # Clear default batch size by pressing Backspace/Delete, then select 32
        await page.click("div[data-baseweb='select'] >> nth=2")
        await page.keyboard.press("Backspace")
        await page.keyboard.press("Backspace")
        await page.click("text=32")
        time.sleep(1)
        
        print("Clicking Predict...")
        # Click predict
        await page.click("button >> text=Predict GPU Usage")
        
        # Wait for prediction metrics to appear
        print("Waiting for results & SHAP...")
        await page.wait_for_selector("text=Confidence Interval (80%)", timeout=20000)
        await page.wait_for_selector("text=Waterfall Plot", timeout=20000)
        time.sleep(3) # Wait for Plotly SHAP charts to render
        
        # Hide the sidebar to make the screenshot cleaner
        await page.evaluate("""
            document.querySelector('[data-testid="stSidebar"]').style.display = 'none';
        """)
        
        screenshot_path = results_dir / "dashboard_shap_demo.png"
        await page.screenshot(path=str(screenshot_path))
        print(f"Saved: {screenshot_path}")
        
        # Now navigate to Batch Size Optimizer
        print("Going to Batch Optimizer...")
        await page.evaluate("""
            document.querySelector('[data-testid="stSidebar"]').style.display = 'block';
        """)
        await page.click("text=Batch Size Optimizer")
        time.sleep(2)
        
        # Wait for the optimizer to render
        await page.wait_for_selector("text=Pareto Frontier: Memory vs. Execution Time", timeout=20000)
        time.sleep(3) # Wait for plotly
        
        await page.evaluate("""
            document.querySelector('[data-testid="stSidebar"]').style.display = 'none';
        """)
        screenshot_path2 = results_dir / "dashboard_batch_optimizer.png"
        await page.screenshot(path=str(screenshot_path2))
        print(f"Saved: {screenshot_path2}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_dashboard())
