"""Smoke test: every page renders. Uses the locally cached data (falls back to cache if offline)."""
import time

import pytest


@pytest.fixture(scope="module")
def client():
    import app as app_module
    import live

    for _ in range(120):  # the first Monte Carlo runs in a background thread
        if live.season_odds() is not None:
            break
        time.sleep(0.5)
    return app_module.app.test_client()


@pytest.mark.parametrize("url", [
    "/", "/?home=BUF&away=SF&neutral=1", "/?home=KC&away=KC", "/week", "/season", "/about",
    "/simulate", "/simulate?seed=42", "/simulate?mode=chalk", "/simulate?mode=average", "/health",
])
def test_page_renders(client, url):
    assert client.get(url).status_code == 200


def test_simulate_opens_on_the_average_season(client):
    r = client.get("/simulate")
    assert r.status_code == 200 and "The average" in r.get_data(as_text=True)


@pytest.mark.parametrize("url", ["/simulate?mode=random", "/simulate?seed=abc", "/simulate?seed=-5"])
def test_random_mode_without_a_valid_seed_redirects_to_a_seeded_url(client, url):
    r = client.get(url)
    assert r.status_code == 302 and "seed=" in r.headers["Location"]


def test_mode_switch_order_is_average_chalk_random(client):
    html = client.get("/simulate").get_data(as_text=True)
    switch = html.split('class="mode-switch"')[1].split("</nav>")[0]
    assert switch.index(">Average<") < switch.index(">Chalk<") < switch.index(">Random<")


def test_simulate_is_reproducible_from_its_link(client):
    a = client.get("/simulate?seed=1234").get_data(as_text=True)
    b = client.get("/simulate?seed=1234").get_data(as_text=True)
    strip = lambda html: html.split('<section class="section" id="regular-season">')[1].split("</main>")[0]
    assert strip(a) == strip(b)


def test_unknown_game_is_404(client):
    assert client.get("/game/nope").status_code == 404
