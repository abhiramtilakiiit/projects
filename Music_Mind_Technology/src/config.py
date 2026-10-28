"""Configuration and constants for the MMT emotion & narrative experiment."""

import re
from typing import Optional

# Standard 20 PANAS Adjectives (10 Positive Affect, 10 Negative Affect)
POSITIVE_ADJECTIVES = [
    'attentive', 'inspired', 'enthusiastic', 'excited', 'interested',
    'active', 'proud', 'determined', 'strong', 'alert'
]

NEGATIVE_ADJECTIVES = [
    'nervous', 'guilty', 'upset', 'jittery', 'irritable',
    'ashamed', 'hostile', 'afraid', 'distressed', 'scared'
]

# Survey Column Names
IP_COLUMN_NAME = "Please put the IP address you were provided below\n"
ADJECTIVE_RECALL_COLUMN_NAME = (
    "List as many adjectives as possible from the list of adjectives "
    "which you were initially asked to rate yourself depending on your mood . "
    "Space separated. Do not go back :)\n"
)
THOUGHTS_COLUMN_NAME = (
    "Describe what kind of thoughts and impressions went through your mind while listening to the music:"
)

# 7-Point Likert Scale Mapping (used in Picture Judgment Facial Ratings)
LIKERT_7PT_MAP = {
    'not at all': 1,
    'very slightly or not at all': 1,
    'slightly': 2,
    'somewhat': 3,
    'moderately': 4,
    'quite a bit': 5,
    'quite': 5,
    'really': 6,
    'extremely': 7
}

# 5-Point Likert Scale Mapping (used in PANAS Ratings)
LIKERT_5PT_MAP = {
    'very slightly or not at all': 1,
    'a little': 2,
    'moderately': 3,
    'quite a bit': 4,
    'extremely': 5
}

# Experimental Groups & Conditions
CONDITION_MAP = {
    '8001': 'Sad Music + Happy Narrative (Incongruent)',
    '8002': 'Sad Music + Sad Narrative (Congruent)',
    '8003': 'Sad Music + No Narrative (Control)',
    '8004': 'Happy Music + Happy Narrative (Congruent)',
    '8005': 'Happy Music + Sad Narrative (Incongruent)',
    '8006': 'Happy Music + No Narrative (Control)',
}

CONDITION_SHORT_MAP = {
    '8001': 'Sad Music +\nHappy Narrative',
    '8002': 'Sad Music +\nSad Narrative',
    '8003': 'Sad Music +\nNo Narrative',
    '8004': 'Happy Music +\nHappy Narrative',
    '8005': 'Happy Music +\nSad Narrative',
    '8006': 'Happy Music +\nNo Narrative',
}

GROUP_NAMES = [str(i) for i in range(8001, 8007)]


def extract_group_port(ip_address: Optional[str]) -> str:
    """Extract port/group identifier (8001-8006) from participant IP address string."""
    if isinstance(ip_address, str):
        match = re.search(r':(\d+)/?$', ip_address.strip())
        if match:
            port = match.group(1)
            if port in GROUP_NAMES:
                return port
    return 'Other'
