"""Generates data/prompts/load_test_prompts.json: ~500 diverse prompts for
Step 10's load test (docs/ROADMAP.md).

Reuses the 229 prompts already in data/labeled_prompts.json (built for
classifier training) as a base, then adds ~270 new prompts across fresh
topic pools so the load test isn't just re-sending the classifier's own
training set. No tier labels here - the live classifier assigns tiers, this
script only needs prompt text.

Usage: python scripts/generate_load_test_prompts.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_router import settings  # noqa: E402

LABELED_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"
OUTPUT_PATH = settings.PROJECT_ROOT / "data" / "prompts" / "load_test_prompts.json"

random.seed(7)

# ------------------------------------------------------------- simple ---

US_STATES = [
    "Oregon", "Vermont", "Arizona", "Ohio", "Georgia", "Montana", "Nevada", "Maine",
    "Colorado", "Wisconsin", "Alabama", "Utah", "Idaho", "Kansas", "Delaware", "Wyoming",
]
STATE_QUESTIONS = [
    "What is the capital of {s}?",
    "What is the two-letter abbreviation for {s}?",
    "What region of the US is {s} typically considered part of?",
    "What is {s} best known for?",
]

ARITHMETIC = [
    "What is 17 + 26?",
    "What is 144 divided by 12?",
    "What is 8 times 13?",
    "What is 100 minus 37?",
    "What is the square root of 81?",
    "What is 15% of 200?",
    "What is 9 squared?",
    "What is 250 divided by 5?",
    "What is 63 + 89?",
    "What is 7 times 14?",
    "What is 500 minus 245?",
    "What is 12 cubed?",
    "What is 20% of 350?",
    "What is 91 divided by 7?",
    "What is the square root of 144?",
    "What is 45 + 78?",
]

PROGRAMMING_TERMS = [
    "recursion", "API", "variable", "compiler", "algorithm", "database index",
    "hash map", "garbage collection", "closure", "thread", "cache", "middleware", "checksum",
    "interpreter",
]
TERM_QUESTIONS = [
    "Define the term '{t}' in one sentence.",
    "Give a one-sentence example of {t}.",
    "Explain '{t}' the way you would to a beginner, in one sentence.",
]

CURRENCY_CONVERSIONS = [
    (100, "USD", "EUR"), (50, "GBP", "USD"), (200, "CAD", "USD"),
    (75, "USD", "JPY"), (30, "EUR", "GBP"), (150, "USD", "MXN"),
    (60, "AUD", "USD"), (90, "USD", "CHF"), (40, "EUR", "USD"), (120, "USD", "INR"),
]

COLOR_NAMES = [
    "crimson", "teal", "goldenrod", "slate gray", "coral", "navy",
    "chartreuse", "orchid", "burnt sienna", "periwinkle", "olive", "maroon",
]

DATE_FACTS = [
    "What day of the week did January 1, 2000 fall on?",
    "How many days are there in a leap year?",
    "What is the ISO 8601 format for representing dates?",
    "How many weeks are there in a standard year?",
    "How many days are in the month of February during a non-leap year?",
    "What is the next leap year after 2024?",
    "How many seconds are in an hour?",
    "What is the Unix epoch start date?",
]

EXTRACT2_FIELDS = [
    "order number and shipping date", "product name and quantity", "sender and due date",
    "customer name and total amount", "flight number and departure time",
]
EXTRACT2_CONTEXTS = [
    "Order #A4821 will ship on March 3rd.",
    "Please send 12 units of the standing desk model.",
    "Invoice from Acme Corp is due April 15th.",
    "Total for customer Maria Lopez comes to $482.10.",
    "Flight UA2291 departs at 6:45am from gate B12.",
]


def _expand(items: list[str], templates: list[str], key: str = "{c}") -> list[str]:
    return [t.replace(key, item) for item in items for t in templates]


def _simple() -> list[str]:
    prompts: list[str] = []
    prompts += _expand(US_STATES, STATE_QUESTIONS, "{s}")
    prompts += ARITHMETIC
    prompts += _expand(PROGRAMMING_TERMS, TERM_QUESTIONS, "{t}")
    prompts += [f"Convert {v} {u1} to {u2} (approximate)." for v, u1, u2 in CURRENCY_CONVERSIONS]
    prompts += [f"What is the hex code for the color {c}?" for c in COLOR_NAMES]
    prompts += DATE_FACTS
    prompts += [
        f"Extract the {field} from this text: {ctx}"
        for field in EXTRACT2_FIELDS
        for ctx in EXTRACT2_CONTEXTS
    ]
    return prompts


# ------------------------------------------------------------ moderate ---

ARTICLE_PARAGRAPHS2 = [
    "A new study found that hybrid work schedules led to a 12% increase in "
    "self-reported employee satisfaction, though managers noted a slight drop "
    "in cross-team collaboration during the first two quarters.",
    "The regional transit authority announced a pilot program offering free "
    "weekend bus service in an effort to reduce weekend traffic congestion "
    "downtown, funded by a one-year state grant.",
    "Researchers observed that a coastal wetland restoration project increased "
    "local bird species counts by nearly a third within three years, though "
    "invasive plant species remain a persistent challenge.",
    "A youth sports league reported record enrollment this spring, crediting "
    "a new scholarship fund that covers registration fees for low-income "
    "families in the district.",
    "A regional grocery chain began piloting reusable produce bags at checkout, "
    "reporting a modest drop in single-use plastic bag purchases within the "
    "first month, though customer feedback was mixed.",
    "A local school district shifted to a four-day instructional week, citing "
    "budget pressures, and is now tracking whether student attendance and "
    "test scores hold steady over the year.",
    "A manufacturing plant switched to a predictive maintenance system, "
    "cutting unplanned downtime by nearly a fifth in its first two quarters "
    "of operation.",
    "A city's public library system expanded its hours on weekends after "
    "survey data showed strong demand from working families, funded by a "
    "reallocation of the parks department budget.",
]

REVIEWS2 = [
    "The hotel room was spacious and clean, but the Wi-Fi kept dropping every hour.",
    "Fast shipping and great packaging, though the color was slightly different from the photos.",
    "Customer support was unhelpful and took three days to respond to a simple question.",
    "The laptop runs quiet and the battery easily lasts a full workday.",
    "Instructions were confusing and one of the parts arrived missing from the box.",
    "Great value for the price, though the fabric pilled after a few washes.",
]

TICKETS2 = [
    "I was charged twice for my subscription this month.",
    "The app crashes every time I try to upload a photo.",
    "I need to update the mailing address on my account.",
    "My password reset email never arrived.",
    "I want to downgrade my plan before the next billing cycle.",
    "The export button on the reports page does nothing when clicked.",
]

EMAILS2 = [
    "Subject: You've won a free cruise! Click here to claim your prize now.",
    "Subject: Your invoice #4521 from Acme Consulting is attached.",
    "Subject: Reminder - your dentist appointment is tomorrow at 2pm.",
    "Subject: URGENT - verify your account within 24 hours or lose access.",
    "Subject: Team standup notes from yesterday's meeting.",
    "Subject: Congratulations, you've been selected for a free gift card!",
]

TOPICS_FOR_PROS_CONS2 = [
    "using a monorepo for a multi-service codebase",
    "adopting a four-day work week company-wide",
    "switching a team from Jira to a lightweight kanban tool",
    "requiring code review approval from two engineers instead of one",
    "moving customer support fully to chat instead of phone",
    "offering unlimited PTO instead of an accrued-days policy",
]

PROCESSES2 = [
    "publishing a package to a public package registry",
    "onboarding a new employee at a mid-sized company",
    "renewing a passport",
    "setting up a small business LLC",
    "migrating a production database with zero downtime",
    "running a quarterly performance review cycle",
]

DATA_TRENDS2 = [
    "Monthly signups: 120, 145, 160, 210, 205, 260",
    "Weekly bug reports: 40, 35, 38, 22, 20, 15",
    "Quarterly churn rate: 5.2%, 4.8%, 4.9%, 4.1%",
    "Daily active users: 3200, 3400, 3100, 3900, 4200, 4600",
    "Support ticket response time (hours): 6.5, 5.9, 5.2, 4.8, 4.1",
    "Average order value: $42, $45, $47, $44, $51, $55",
]

COMPARE_PAIRS2 = [
    ("REST APIs", "GraphQL APIs"),
    ("unit tests", "integration tests"),
    ("a monolith architecture", "a microservices architecture"),
    ("server-side rendering", "client-side rendering"),
    ("SQL databases", "NoSQL databases"),
    ("agile sprints", "kanban-style continuous flow"),
]


def _moderate() -> list[str]:
    prompts: list[str] = []
    prompts += [f"Summarize the following in two sentences:\n\n{p}" for p in ARTICLE_PARAGRAPHS2]
    prompts += [f"What is the main takeaway of this passage in one sentence: {p}" for p in ARTICLE_PARAGRAPHS2]
    prompts += [f'Classify the sentiment of this review as positive, negative, or neutral: "{r}"' for r in REVIEWS2]
    prompts += [f'Categorize this support ticket into billing, technical, or account: "{t}"' for t in TICKETS2]
    prompts += [f'Classify this email as spam or legitimate: "{e}"' for e in EMAILS2]
    prompts += [f"List the pros and cons of {t}." for t in TOPICS_FOR_PROS_CONS2]
    prompts += [f"Outline the main steps involved in {p}." for p in PROCESSES2]
    prompts += [f"Given this data, identify the trend: {d}" for d in DATA_TRENDS2]
    prompts += [
        f"Summarize the key differences between {a} and {b} in a short paragraph."
        for a, b in COMPARE_PAIRS2
    ]
    return prompts


# ------------------------------------------------------------- complex ---

OPTION_PAIRS2 = [
    ("AWS", "GCP", "hosting a data pipeline", "cost", "vendor lock-in", "operational complexity"),
    ("Postgres", "MongoDB", "a content management system", "query flexibility", "consistency guarantees", "horizontal scaling"),
    ("Rust", "Go", "a high-throughput network service", "memory safety", "developer velocity", "concurrency model"),
    ("Kubernetes", "a managed PaaS", "deploying a small startup's backend", "operational overhead", "cost at low scale", "flexibility"),
    ("a single large model", "an ensemble of smaller models", "a production ML pipeline", "latency", "accuracy", "maintenance burden"),
]

CREATIVE_PREMISES2 = [
    "a retired astronaut who receives a radio signal from their old mission",
    "a librarian who discovers a book that rewrites itself overnight",
    "two rival chefs forced to run the same food truck for a week",
    "a clockmaker who realizes one of their clocks runs a minute into the future",
    "a small town where the streetlights start switching on in a new pattern every night",
]

PHENOMENA2 = [
    ("supply chain disruptions", "cause short-term price spikes", "over-correcting inventory afterward"),
    ("urban heat islands", "raise nighttime temperatures in cities", "increased energy demand during heat waves"),
    ("social media algorithm changes", "shift how quickly news spreads", "amplifying misinformation"),
    ("interest rate cuts", "stimulate short-term borrowing", "reigniting inflation"),
    ("droughts", "reduce hydropower output", "cascading into higher electricity prices"),
]

JUDGMENT_SCENARIOS2 = [
    ("A startup founder", "raising a large funding round with heavy dilution", "bootstrapping slower but retaining control"),
    ("A city planner", "approving a new highway expansion", "investing the same budget in public transit"),
    ("A hospital administrator", "adopting an AI diagnostic tool immediately", "waiting for a longer clinical validation period"),
    ("A nonprofit director", "accepting a large donation with restrictive strings attached", "declining it and relying on smaller unrestricted gifts"),
    ("A software team lead", "shipping a feature with known minor bugs to hit a deadline", "delaying the release to fix them first"),
]

SYSTEM_DESIGNS2 = [
    ("rate limiter", "a public API with millions of daily requests"),
    ("notification system", "a mobile app sending real-time alerts to users"),
    ("search index", "an e-commerce site with a large, frequently updated catalog"),
    ("job queue", "a service processing user-uploaded video transcoding requests"),
    ("caching layer", "a read-heavy news website with spiky traffic"),
]

PERSUASIVE_POSITIONS2 = [
    "that companies should publish their gender pay gap data annually",
    "that public libraries should expand into offering tool and equipment lending",
    "that cities should prioritize bike infrastructure over new parking capacity",
    "that schools should teach personal finance as a required course",
    "that software companies should be required to disclose known security vulnerabilities within 30 days",
]


def _complex() -> list[str]:
    prompts: list[str] = []
    for a, b, use_case, x, y, z in OPTION_PAIRS2:
        prompts.append(
            f"Compare the trade-offs between {a} and {b} for {use_case}. Analyze {x}, {y}, "
            f"and {z} step by step, then give a final recommendation. Output as JSON with "
            f"fields: options, tradeoffs, recommendation."
        )
    for premise in CREATIVE_PREMISES2:
        prompts.append(
            f"Write a short creative story about {premise}, and explain why the ending you "
            f"chose is meaningful."
        )
    for cause, effect, risk in PHENOMENA2:
        prompts.append(
            f"Why do {cause} typically {effect}? Give a nuanced, step by step explanation of "
            f"the mechanisms involved, and discuss the risks of {risk}."
        )
    for person, choice_a, choice_b in JUDGMENT_SCENARIOS2:
        prompts.append(
            f"{person} is trying to decide between {choice_a} and {choice_b}. Analyze the "
            f"ethical and practical implications of each and recommend a course of action, "
            f"explaining your reasoning."
        )
    for system_type, scenario in SYSTEM_DESIGNS2:
        prompts.append(
            f"Design a {system_type} for {scenario}, analyzing scalability, cost, and "
            f"reliability trade-offs, and justify your final design choice."
        )
    for position in PERSUASIVE_POSITIONS2:
        prompts.append(
            f"Write a persuasive essay arguing {position}, addressing likely counterarguments "
            f"and using at least three distinct supporting points."
        )
    return prompts


def main() -> None:
    labeled = json.loads(LABELED_PATH.read_text())
    base_prompts = [r["prompt"] for r in labeled]

    new_prompts = _simple() + _moderate() + _complex()

    all_prompts = list(dict.fromkeys(base_prompts + new_prompts))
    random.shuffle(all_prompts)

    records = [{"id": f"lt{i + 1}", "prompt": p} for i, p in enumerate(all_prompts)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} prompts to {OUTPUT_PATH}")
    print(f"  Base (from labeled_prompts.json): {len(base_prompts)}")
    print(f"  New: {len(new_prompts)}, after dedup: {len(all_prompts) - len(base_prompts)}")


if __name__ == "__main__":
    main()
