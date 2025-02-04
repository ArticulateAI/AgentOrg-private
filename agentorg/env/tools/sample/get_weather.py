from agentorg.env.tools.tools import register_tool
import logging
import requests

logger = logging.getLogger(__name__)

description = "Get the weather information of a city given the latitude and longitude of the city"
slots = [
    {
        "name": "latitude",
        "type": "number",
        "description": "The latitude of the city",
        "prompt": "In order to proceed, please provide the latitude of the city.",
        "required": True,
    },
    {
        "name": "longitude",
        "type": "number",
        "description": "The longitude of the city.",
        "prompt": "In order to proceed, please provide the longitude of the city.",
        "required": True,
    }
]
outputs = [
    {
        "name": "weather_info",
        "type": "string",
        "description": "The weather information of the city. such as 'Temperature: 20°C, Wind: 10 km/h'",
    }
]


@register_tool(description, slots, outputs)
def get_weather(latitude: float, longitude: float, **kwargs) -> str:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"

    try:
        response = requests.get(url)
        data = response.json()
        logger.info(f"weather data: {data}")
        temperature = data["current"]["temperature_2m"]
        wind_speed = data["current"]["wind_speed_10m"]
        return f"The weather information of the requested city - Temperature: {temperature}°C, Wind: {wind_speed} km/h"
    except Exception as e:
        logger.error(f"Error: {e}")
        return "error: weather information not found"
