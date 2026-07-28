import io
import httpx

QUICKCHART_URL = "https://quickchart.io/chart"

async def get_price_chart(ticker: str, prices: list[float], dates: list[str]) -> io.BytesIO:
    """Generates a line chart for prices and returns it as an in-memory buffer."""
    payload = {
        "backgroundColor": "#121212",
        "width": 800,
        "height": 400,
        "format": "png",
        "chart": {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": [{
                    "label": f"{ticker} Price",
                    "data": prices,
                    "borderColor": "#3b82f6",
                    "backgroundColor": "rgba(59, 130, 246, 0.2)",
                    "borderWidth": 3,
                    "pointRadius": 0,
                    "pointHoverRadius": 5,
                    "tension": 0.4,
                    "fill": True
                }]
            },
            "options": {
                "legend": {
                    "display": False
                },
                "title": {
                    "display": True,
                    "text": f"{ticker} Price Chart",
                    "fontColor": "#ffffff",
                    "fontSize": 18,
                    "padding": 20
                },
                "scales": {
                    "xAxes": [{
                        "ticks": {"fontColor": "#9ca3af", "maxTicksLimit": 10},
                        "gridLines": {"display": False}
                    }],
                    "yAxes": [{
                        "ticks": {"fontColor": "#9ca3af"},
                        "gridLines": {"color": "rgba(255, 255, 255, 0.05)", "zeroLineColor": "rgba(255, 255, 255, 0.1)"}
                    }]
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(QUICKCHART_URL, json=payload, timeout=10.0)
        response.raise_for_status()
        return io.BytesIO(response.content)

async def get_sentiment_chart(title: str, values: list[float], labels: list[str]) -> io.BytesIO:
    """Generates a bar chart and returns it as an in-memory buffer."""
    payload = {
        "backgroundColor": "#121212",
        "width": 800,
        "height": 400,
        "format": "png",
        "chart": {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": title,
                    "data": values,
                    "backgroundColor": "rgba(59, 130, 246, 0.8)",
                    "borderColor": "#3b82f6",
                    "borderWidth": 1,
                    "borderRadius": 4
                }]
            },
            "options": {
                "legend": {
                    "display": False
                },
                "title": {
                    "display": True,
                    "text": title,
                    "fontColor": "#ffffff",
                    "fontSize": 18,
                    "padding": 20
                },
                "scales": {
                    "xAxes": [{
                        "ticks": {"fontColor": "#9ca3af", "maxTicksLimit": 10},
                        "gridLines": {"display": False}
                    }],
                    "yAxes": [{
                        "ticks": {"fontColor": "#9ca3af", "suggestedMin": -1, "suggestedMax": 1},
                        "gridLines": {"color": "rgba(255, 255, 255, 0.05)", "zeroLineColor": "rgba(255, 255, 255, 0.2)"}
                    }]
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(QUICKCHART_URL, json=payload, timeout=10.0)
        response.raise_for_status()
        return io.BytesIO(response.content)
