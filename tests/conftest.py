import os

# Tests exercise the rule-based engines; never call the Claude API from tests
os.environ.pop("ANTHROPIC_API_KEY", None)
