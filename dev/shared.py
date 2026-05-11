import logging
import os

from llmai.shared.tools import Tool, ToolChoiceMode, WebSearchTool


def get_dev_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"llmai.dev.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(os.getenv("LLMAI_LOG_LEVEL", "INFO").upper())
    return logger


STRING_DEFS = {
    "ShortLabel": {
        "type": "string",
        "minLength": 2,
        "maxLength": 32,
        "pattern": "^[A-Za-z][A-Za-z0-9 _-]*$",
    },
    "SlideTitle": {
        "type": "string",
        "minLength": 20,
        "maxLength": 100,
    },
    "MarkdownParagraph": {
        "type": "string",
        "minLength": 40,
        "maxLength": 500,
    },
    "ImagePrompt": {
        "type": "string",
        "minWords": 2,
        "maxWords": 12,
    },
    "EmailAddress": {
        "type": "string",
        "format": "email",
    },
    "Url": {
        "type": "string",
        "format": "uri",
        "examples": ["https://example.com/slides/global-warming"],
    },
    "Slug": {
        "type": "string",
        "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$",
        "minLength": 3,
        "maxLength": 80,
    },
    "Username": {
        "type": "string",
        "pattern": "^@[a-zA-Z0-9_]{3,30}$",
    },
    "IsoDate": {
        "type": "string",
        "format": "date",
    },
    "IsoDateTime": {
        "type": "string",
        "format": "date-time",
    },
    "Uuid": {
        "type": "string",
        "format": "uuid",
    },
    "Hostname": {
        "type": "string",
        "format": "hostname",
    },
    "IPv4Address": {
        "type": "string",
        "format": "ipv4",
    },
    "IPv6Address": {
        "type": "string",
        "format": "ipv6",
    },
    "LanguageTag": {
        "type": "string",
        "pattern": "^[a-z]{2,3}(?:-[A-Z]{2})?$",
        "examples": ["en-US", "ne-NP"],
    },
    "HexColor": {
        "type": "string",
        "pattern": "^#[0-9A-Fa-f]{6}$",
    },
}

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "$ref": "#/$defs/ImagePrompt",
        },
        "altText": {
            "$ref": "#/$defs/MarkdownParagraph",
        },
        "style": {
            "type": "string",
            "enum": ["photo", "diagram", "infographic", "illustration"],
        },
        "sourceUrl": {
            "$ref": "#/$defs/Url",
        },
    },
    "required": ["prompt", "altText", "style", "sourceUrl"],
    "additionalProperties": False,
    "$defs": STRING_DEFS,
}

SLIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {
            "$ref": "#/$defs/Uuid",
        },
        "title": {
            "$ref": "#/$defs/SlideTitle",
        },
        "subtitle": {
            "$ref": "#/$defs/MarkdownParagraph",
        },
        "email": {
            "$ref": "#/$defs/EmailAddress",
        },
        "authorHandle": {
            "$ref": "#/$defs/Username",
        },
        "url": {
            "$ref": "#/$defs/Url",
        },
        "slug": {
            "$ref": "#/$defs/Slug",
        },
        "language": {
            "$ref": "#/$defs/LanguageTag",
        },
        "publishedOn": {
            "$ref": "#/$defs/IsoDate",
        },
        "updatedAt": {
            "$ref": "#/$defs/IsoDateTime",
        },
        "originHost": {
            "$ref": "#/$defs/Hostname",
        },
        "requestIpV4": {
            "$ref": "#/$defs/IPv4Address",
        },
        "requestIpV6": {
            "$ref": "#/$defs/IPv6Address",
        },
        "bulletPoints": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/BulletPoint",
            },
            "minItems": 1,
            "maxItems": 5,
        },
        "image": {
            "$ref": "#/$defs/Image",
        },
        "theme": {
            "$ref": "#/$defs/Theme",
        },
        "citations": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/Citation",
            },
            "minItems": 1,
            "maxItems": 4,
        },
        "speakerNotes": {
            "allOf": [
                {
                    "$ref": "#/$defs/MarkdownParagraph",
                }
            ],
            "description": "Presenter notes in markdown-friendly plain text.",
        },
    },
    "required": [
        "id",
        "title",
        "subtitle",
        "email",
        "authorHandle",
        "url",
        "slug",
        "language",
        "publishedOn",
        "updatedAt",
        "originHost",
        "requestIpV4",
        "requestIpV6",
        "bulletPoints",
        "image",
        "theme",
        "citations",
        "speakerNotes",
    ],
    # "additionalProperties": False,
    "$defs": {
        **STRING_DEFS,
        "BulletPoint": {
            "allOf": [
                {
                    "$ref": "#/$defs/MarkdownParagraph",
                }
            ],
            "description": "One concise slide bullet.",
        },
        "Image": IMAGE_SCHEMA,
        "Theme": {
            "type": "object",
            "properties": {
                "name": {
                    "$ref": "#/$defs/ShortLabel",
                },
                "primaryColor": {
                    "$ref": "#/$defs/HexColor",
                },
                "accentColor": {
                    "$ref": "#/$defs/HexColor",
                },
            },
            "required": ["name", "primaryColor", "accentColor"],
            "additionalProperties": False,
        },
        "Citation": {
            "type": "object",
            "properties": {
                "label": {
                    "$ref": "#/$defs/ShortLabel",
                },
                "url": {
                    "$ref": "#/$defs/Url",
                },
                "retrievedAt": {
                    "$ref": "#/$defs/IsoDateTime",
                },
            },
            "required": ["label", "url", "retrievedAt"],
            "additionalProperties": False,
        },
    },
}

TOOL_LOCATION_DEFS = {
    "CityName": {
        "type": "string",
        "minLength": 2,
        "maxLength": 80,
        "pattern": "^[A-Za-z][A-Za-z .'-]*$",
    },
    "CityWithRegion": {
        "type": "string",
        "minLength": 5,
        "maxLength": 120,
        "pattern": "^[A-Za-z][A-Za-z .'-]+, [A-Za-z][A-Za-z .'-]+$",
    },
    "CountryCode": {
        "type": "string",
        "pattern": "^[A-Z]{2}$",
    },
    "CountryName": {
        "type": "string",
        "minLength": 2,
        "maxLength": 80,
        "pattern": "^[A-Za-z][A-Za-z .'-]*$",
    },
    "Locale": STRING_DEFS["LanguageTag"],
    "LocationText": {
        "anyOf": [
            {
                "$ref": "#/$defs/CityName",
            },
            {
                "$ref": "#/$defs/CityWithRegion",
            },
        ],
    },
    "Location": {
        "type": "object",
        "properties": {
            "city": {
                "allOf": [
                    {
                        "$ref": "#/$defs/LocationText",
                    }
                ],
                "description": "City name, optionally including a region.",
            },
            "country": {
                "anyOf": [
                    {
                        "$ref": "#/$defs/CountryCode",
                    },
                    {
                        "$ref": "#/$defs/CountryName",
                    },
                ],
            },
            "locale": {
                "$ref": "#/$defs/Locale",
            },
        },
        "required": ["city", "country", "locale"],
        "additionalProperties": False,
    },
}

WEATHER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {
            "allOf": [
                {
                    "$ref": "#/$defs/Location",
                }
            ],
            "description": "The place to look up weather for.",
        },
        "units": {
            "anyOf": [
                {
                    "$ref": "#/$defs/MetricUnits",
                },
                {
                    "$ref": "#/$defs/ImperialUnits",
                },
            ],
        },
        "detail": {
            "allOf": [
                {
                    "$ref": "#/$defs/WeatherDetail",
                }
            ],
        },
    },
    "required": ["location", "units", "detail"],
    "$defs": {
        **TOOL_LOCATION_DEFS,
        "MetricUnits": {
            "type": "string",
            "enum": ["celsius", "metric"],
        },
        "ImperialUnits": {
            "type": "string",
            "enum": ["fahrenheit", "imperial"],
        },
        "WeatherDetail": {
            "type": "string",
            "enum": ["current", "hourly", "daily"],
        },
    },
}

TIME_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {
            "allOf": [
                {
                    "$ref": "#/$defs/Location",
                }
            ],
            "description": "The place to look up local time for.",
        },
        "timeFormat": {
            "anyOf": [
                {
                    "$ref": "#/$defs/TwelveHourFormat",
                },
                {
                    "$ref": "#/$defs/TwentyFourHourFormat",
                },
            ],
        },
        "referenceDate": {
            "allOf": [
                {
                    "$ref": "#/$defs/IsoDate",
                }
            ],
        },
    },
    "required": ["location", "timeFormat", "referenceDate"],
    "$defs": {
        **TOOL_LOCATION_DEFS,
        "IsoDate": STRING_DEFS["IsoDate"],
        "TwelveHourFormat": {
            "type": "string",
            "enum": ["12h", "clock12"],
        },
        "TwentyFourHourFormat": {
            "type": "string",
            "enum": ["24h", "clock24"],
        },
    },
}

TIMEZONE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {
            "allOf": [
                {
                    "$ref": "#/$defs/Location",
                }
            ],
            "description": "The place to resolve into a timezone identifier.",
        },
        "identifierStyle": {
            "anyOf": [
                {
                    "$ref": "#/$defs/IanaStyle",
                },
                {
                    "$ref": "#/$defs/OffsetStyle",
                },
            ],
        },
        "observedAt": {
            "allOf": [
                {
                    "$ref": "#/$defs/IsoDateTime",
                }
            ],
        },
    },
    "required": ["location", "identifierStyle", "observedAt"],
    "$defs": {
        **TOOL_LOCATION_DEFS,
        "IsoDateTime": STRING_DEFS["IsoDateTime"],
        "IanaStyle": {
            "type": "string",
            "enum": ["iana", "region/name"],
        },
        "OffsetStyle": {
            "type": "string",
            "enum": ["utc-offset", "abbreviation"],
        },
    },
}

TOOL_DEFINITIONS = [
    Tool(
        name="get_weather",
        description="Get the current weather for a city",
        strict=True,
        schema=WEATHER_TOOL_SCHEMA,
    ),
    Tool(
        name="get_time",
        description="Get the current local time for a city",
        schema=TIME_TOOL_SCHEMA,
    ),
    Tool(
        name="get_timezone",
        description="Get the timezone identifier for a city",
        schema=TIMEZONE_TOOL_SCHEMA,
    ),
]

TOOL_CHOICE = {
    "mode": ToolChoiceMode.REQUIRED,
    "tools": ["get_weather", "get_time", "get_timezone"],
}

WEB_SEARCH_TOOL = WebSearchTool()
