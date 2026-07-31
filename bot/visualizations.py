import io
import httpx

QUICKCHART_URL = "https://quickchart.io/chart"

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
