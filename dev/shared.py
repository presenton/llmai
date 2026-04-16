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
