from llmai.shared.tools import Tool

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "minWords": 2,
            "maxWords": 5,
        },
    },
    "required": ["prompt"],
}

SLIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "minLength": 20,
            "maxLength": 100,
        },
        "email": {
            "type": "string",
            "format": "email",
        },
        "url": {
            "type": "string",
            "format": "uri",
        },
        "bulletPoints": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "minItems": 1,
            "maxItems": 3,
        },
        "image": IMAGE_SCHEMA,
    },
    "required": ["title", "image", "url", "email", "bulletPoints"],
}

TOOL_DEFINITIONS = [
    Tool(
        name="get_weather",
        description="Get the current weather for a city",
        schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    ),
    Tool(
        name="get_time",
        description="Get the current local time for a city",
        schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    ),
    Tool(
        name="get_timezone",
        description="Get the timezone identifier for a city",
        schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    ),
]

TOOL_CHOICE = {
    "required": ["get_weather", "get_time"],
    "optional": ["get_timezone"],
}
