import pytest


pytestmark = pytest.mark.e2e
playwright = pytest.importorskip("playwright.sync_api")


def test_analysis_flow_and_new_analysis_navigation(live_server):
    with playwright.sync_playwright() as manager:
        browser = manager.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(live_server)
        page.locator('input[name="titulo"]').fill("Introducción a Machine Learning")
        page.locator('textarea[name="texto"]').fill("Algoritmos de aprendizaje automático para clasificación")
        page.locator('button[type="submit"]').click()
        page.wait_for_load_state("networkidle")
        body = page.locator("body").inner_text().upper()
        assert "CATEGORÍA" in body
        assert "CONFIANZA" in body
        assert "PALABRAS CLAVE" in body
        if page.locator(".swal2-confirm").is_visible():
            page.locator(".swal2-confirm").click()
        page.locator("a.btn-outline-primary").click()
        page.wait_for_url(f"{live_server}/")
        assert page.locator('button[type="submit"]').is_visible()
        browser.close()


def test_blank_form_error_and_mobile_layout(live_server):
    with playwright.sync_playwright() as manager:
        browser = manager.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(live_server)
        page.locator('button[type="submit"]').click()
        page.locator(".swal2-popup, #server-error:not(.d-none)").wait_for(state="visible")
        assert "INGRESA UN TÍTULO O CONTENIDO TÉCNICO" in page.locator("body").inner_text().upper()
        assert page.locator('input[name="titulo"]').is_visible()
        browser.close()
