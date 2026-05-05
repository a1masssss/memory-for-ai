from __future__ import annotations


SECTION_TITLES = {
    "personal_context": "Known Facts About This User",
    "creative_style": "Stable Creative Profile",
    "camera_language": "Camera And Composition Preferences",
    "motion_language": "Motion Preferences",
    "lighting": "Lighting Preferences",
    "negative_constraints": "Negative Prompt Memory",
    "generation_feedback": "Relevant Prior Generation Feedback",
    "character_continuity": "Continuity Anchors",
    "communication": "Communication Preferences",
    "opinions": "Known Opinions",
    "project_goal": "Current Goals And Upcoming Events",
}

RECENT_CONTEXT_TITLE = "Relevant From Recent Conversations"
MAX_MEMORIES_PER_SECTION = 6
MAX_HOP_ANCHORS = 8
HOP_RELATION_BOOSTS = {
    "same_profile": 0.45,
    "co_mentioned": 0.3,
    "opinion_arc": 0.2,
    "opinion_refinement": 0.28,
    "opinion_correction": 0.35,
}

STOPWORDS = {
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
    "user",
    "assistant",
    "tool",
    "their",
    "them",
    "they",
    "there",
    "about",
}

TYPE_BOOSTS = {
    "fact": 1.5,
    "preference": 1.4,
    "constraint": 1.35,
    "style_profile": 1.35,
    "generation_feedback": 1.15,
    "continuity_anchor": 1.25,
    "event": 1.0,
    "opinion": 1.0,
}

QUERY_ALIASES = {
    ("personal_context", "name"): "name called identity who",
    ("personal_context", "current_location"): "location city live lives where moved based",
    ("personal_context", "employment"): "employment work works job company role joined",
    ("personal_context", "current_role"): "role title position job work career",
    ("personal_context", "pet"): "pet dog cat animal name named",
    ("personal_context", "dietary_preference"): "diet food vegetarian vegan eats",
    ("personal_context", "allergy"): "allergy allergic avoid food constraint",
    ("personal_context", "family"): "family child son daughter kid wife husband partner spouse",
    ("communication", "answer_style"): "communication answer style concise direct verbose",
    ("project_goal", "upcoming_focus"): "preparing interview upcoming goal plan focus",
    ("project_goal", "travel_plan"): "travel trip destination plan",
    ("creative_style", "visual_style"): "visual style aesthetic look generation video",
    ("camera_language", "camera_direction"): "camera composition framing shot angle video",
    ("motion_language", "motion_style"): "motion movement pacing speed video",
    ("lighting", "lighting_style"): "lighting light color mood video",
    ("negative_constraints", "avoid_glossy_polish"): "avoid negative constraint glossy polish",
}
