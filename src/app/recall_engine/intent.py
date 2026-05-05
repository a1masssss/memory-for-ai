from __future__ import annotations

import re


QUERY_INTENT_RULES = (
    {
        "pattern": re.compile(
            r"\b(name|called|call them|who is|identity)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("name", "called", "identity"),
        "keys": {("personal_context", "name")},
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(where|live|based|located|location|city|home base|call home|hometown)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("location", "city", "live", "moved", "based", "home"),
        "keys": {("personal_context", "current_location")},
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(work|job|company|employer|role|joined|career|day job)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("employment", "work", "job", "company", "role", "joined"),
        "keys": {
            ("personal_context", "employment"),
            ("personal_context", "current_role"),
        },
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(food|diet|dietary|eat|eats|allerg|restriction|vegetarian|vegan|shellfish)\b",
            flags=re.IGNORECASE,
        ),
        "terms": (
            "diet",
            "dietary",
            "vegetarian",
            "vegan",
            "allergy",
            "allergic",
            "restriction",
            "food",
        ),
        "keys": {
            ("personal_context", "dietary_preference"),
            ("personal_context", "allergy"),
        },
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(reply|respond|response|communication|tone|chatty|brief|concise|direct|wordy|verbose)\b",
            flags=re.IGNORECASE,
        ),
        "terms": (
            "communication",
            "answer",
            "style",
            "concise",
            "direct",
            "brief",
            "verbose",
        ),
        "keys": {("communication", "answer_style")},
        "categories": {"communication"},
    },
    {
        "pattern": re.compile(
            r"\b(style|styled|aesthetic|look|visual|vibe|feel|fashioned)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("visual", "style", "aesthetic", "look", "creative", "video"),
        "keys": {("creative_style", "visual_style")},
        "categories": {"creative_style"},
    },
    {
        "pattern": re.compile(
            r"\b(avoid|negative|constraint|glossy|artifacts?|wrong|failed|shouldn't)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("avoid", "negative", "constraint", "artifacts", "feedback"),
        "keys": set(),
        "categories": {"negative_constraints", "generation_feedback"},
    },
    {
        "pattern": re.compile(
            r"\b(camera|framing|frame|shot|angle|composition)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("camera", "composition", "framing", "shot", "angle"),
        "keys": {("camera_language", "camera_direction")},
        "categories": {"camera_language"},
    },
    {
        "pattern": re.compile(
            r"\b(motion|movement|pace|pacing|energy|slow motion|fast paced)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("motion", "movement", "pacing", "speed", "energy"),
        "keys": {("motion_language", "motion_style")},
        "categories": {"motion_language"},
    },
    {
        "pattern": re.compile(
            r"\b(light|lighting|lit|mood|neon|daylight|golden hour)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("lighting", "light", "mood", "daylight", "neon"),
        "keys": {("lighting", "lighting_style")},
        "categories": {"lighting"},
    },
    {
        "pattern": re.compile(
            r"\b(pet|dog|cat|animal|owner|named)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("pet", "dog", "cat", "animal", "named"),
        "keys": {("personal_context", "pet")},
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(family|child|kid|son|daughter|wife|husband|partner|spouse)\b",
            flags=re.IGNORECASE,
        ),
        "terms": ("family", "child", "son", "daughter", "partner", "spouse"),
        "keys": {("personal_context", "family")},
        "categories": {"personal_context"},
    },
    {
        "pattern": re.compile(
            r"\b(preparing|interview|presentation|exam|trip|travel|planning|upcoming|goal|focus)\b",
            flags=re.IGNORECASE,
        ),
        "terms": (
            "preparing",
            "interview",
            "trip",
            "travel",
            "upcoming",
            "goal",
            "focus",
        ),
        "keys": {
            ("project_goal", "upcoming_focus"),
            ("project_goal", "travel_plan"),
        },
        "categories": {"project_goal"},
    },
)
