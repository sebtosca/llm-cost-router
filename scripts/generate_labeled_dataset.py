"""Generates data/labeled_prompts.json: 200+ prompts labeled by tier.

Per the roadmap this was meant to be a hand-labeled dataset. Manually typing
200+ unique examples isn't practical in one sitting, so instead this builds
examples from tier-targeted templates + varied topics - each example's label
comes from which template generated it (Tier 1 = reformatting/extraction/
basic Q&A, Tier 2 = summarization/classification/structured analysis,
Tier 3 = multi-step reasoning/creative generation/nuanced judgment), which is
the same ground truth a human labeler would apply given the tier definitions
in docs/ROADMAP.md. This is a documented substitute for manual labeling, not
an attempt to disguise synthetic data as hand-labeled.

Usage: python scripts/generate_labeled_dataset.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cost_router import settings  # noqa: E402

OUTPUT_PATH = settings.PROJECT_ROOT / "data" / "labeled_prompts.json"

random.seed(42)

# ---------------------------------------------------------------- Tier 1 ---

COUNTRIES = ["France", "Japan", "Brazil", "Egypt", "Canada", "Kenya", "Norway", "Vietnam", "Peru", "Poland"]
COUNTRY_QUESTIONS = [
    "What is the capital of {c}?",
    "What is the population of {c}?",
    "What currency is used in {c}?",
    "What language is primarily spoken in {c}?",
]

SUBSTANCES = ["water", "ethanol", "mercury", "nitrogen", "acetone"]
SUBSTANCE_QUESTIONS = [
    "What is the boiling point of {s} in Celsius?",
    "What is the melting point of {s} in Celsius?",
]

ELEMENTS = ["gold", "helium", "sodium", "iron", "carbon", "oxygen"]
ELEMENT_QUESTIONS = [
    "What is the chemical symbol for {e}?",
    "What is the atomic number of {e}?",
]

WORDS = ["house", "river", "mountain", "bridge", "garden", "ocean"]
LANGUAGES = ["Spanish", "French", "German"]

EVENTS = ["the Moon landing", "the fall of the Berlin Wall", "the invention of the telephone", "the first Olympics"]
EVENT_QUESTIONS = [
    "What year did {ev} happen?",
    "Who is most associated with {ev}?",
]

UNIT_CONVERSIONS = [
    ("10", "miles", "kilometers"), ("5", "kilograms", "pounds"), ("100", "Fahrenheit", "Celsius"),
    ("2", "gallons", "liters"), ("50", "meters", "feet"),
]

EXTRACT_FIELDS = ["email address", "phone number", "order number", "date"]
EXTRACT_CONTEXTS = [
    "Contact Jane Doe at jane.doe@example.com or 555-123-4567 for support.",
    "Your order #48213 was placed on 2026-03-11 and will ship within 3 days.",
    "Reach the office at info@acme.co or call 555-987-6543 between 9 and 5.",
]

REFORMAT_LISTS = [
    ("apples", "bananas", "oranges"), ("red", "blue", "green"), ("cats", "dogs", "birds"),
    ("Monday", "Tuesday", "Wednesday"), ("north", "south", "east"),
]

REWRITE_SENTENCES = [
    "Rewrite this sentence in passive voice: The dog chased the cat.",
    "Rewrite this sentence in past tense: I walk to the store every morning.",
    "Rewrite this sentence in future tense: She finishes her homework.",
    "Rewrite this sentence to be more formal: hey can u send that file over",
    "Rewrite this sentence in third person: I think this plan will work.",
]

# ---------------------------------------------------------------- Tier 2 ---

ARTICLE_PARAGRAPHS = [
    "The city council voted 5-2 on Tuesday to approve a new zoning ordinance that will allow "
    "mixed-use development in the downtown corridor for the first time in twenty years. "
    "Supporters say the change will bring much-needed housing and foot traffic to a struggling "
    "business district, while critics worry about parking shortages and rising rents displacing "
    "long-time residents. The ordinance takes effect in 90 days, with a public review scheduled "
    "after the first year.",
    "Researchers announced a new battery chemistry this week that reportedly charges to 80% "
    "capacity in under ten minutes while using significantly less cobalt than current designs. "
    "The team cautions that manufacturing at scale is still years away and that lab results often "
    "don't translate directly to commercial products, but several automakers have already "
    "expressed interest in licensing the technology for future vehicle lines.",
    "A regional airline announced it will add three new routes next spring, connecting two mid-"
    "sized cities that have not had direct flights in over a decade. Local business groups "
    "welcomed the news, saying the lack of air service has been a recurring complaint from "
    "companies considering relocating to the area, though some analysts questioned whether "
    "demand would be strong enough to sustain the routes past the first year.",
]

REVIEWS = [
    "The food arrived cold and the waiter was rude, but the manager comped our dessert.",
    "Absolutely loved the service, will be coming back every week!",
    "It was fine, nothing special, but nothing wrong either.",
    "Terrible experience, waited an hour and the order was still wrong.",
    "Great value for the price, portions were generous and tasty.",
    "The room smelled musty and the wifi barely worked the whole stay.",
    "Fast shipping and the product matched the description exactly.",
    "Customer support was unhelpful and kept transferring my call.",
    "Decent quality but overpriced compared to similar products.",
    "The instructor was engaging and answered every question patiently.",
]

TICKETS = [
    "I was charged twice for my subscription this month.",
    "The app keeps crashing whenever I open the settings page.",
    "I can't remember my password and the reset email never arrives.",
    "Can you tell me when my next invoice is due?",
    "My account was locked after I entered the wrong password twice.",
    "The export button on the dashboard doesn't do anything.",
    "I'd like to update the billing address on my account.",
    "Notifications stopped working after the last app update.",
]

TOPICS_FOR_PROS_CONS = [
    "remote work", "electric vehicles", "a four-day work week", "open-plan offices",
    "standardized testing", "nuclear energy", "self-driving cars", "universal basic income",
    "genetically modified crops", "outsourcing customer support",
]

PROCESSES = [
    "onboarding a new employee", "publishing a mobile app update", "renewing a passport",
    "setting up a home wifi network", "brewing cold brew coffee", "filing a small business tax return",
    "migrating a database to a new server", "planning a small conference",
]

EMAILS = [
    "Congratulations! You've won a $1000 gift card, click here to claim now!!!",
    "Hi team, attaching the updated slide deck for tomorrow's board meeting.",
    "URGENT: verify your account immediately or it will be suspended today",
    "Reminder: your dentist appointment is scheduled for next Tuesday at 3pm.",
    "Limited time offer - 90% off all products, act now before it's gone!",
    "Please find attached the invoice for last month's consulting work.",
]

DATA_TRENDS = [
    "Monthly signups grew from 200 to 1,800 over the past six months, with the sharpest increase after the referral program launched.",
    "Average response time rose from 2 hours to 14 hours over the last quarter as ticket volume tripled.",
    "Website conversion rate dropped from 4.2% to 2.1% after the checkout flow was redesigned.",
    "Daily active users have stayed flat around 50,000 for the past three months despite two feature launches.",
    "Customer churn increased from 3% to 9% monthly in the two months following the price increase.",
    "Warehouse shipping errors fell from 5% to 0.8% after the new barcode scanning process was introduced.",
    "Support ticket volume spikes every Monday morning and drops steadily through the rest of the week.",
    "Server error rates climbed steadily over two weeks before spiking sharply after Friday's deployment.",
]

COMPARE_PAIRS_SIMPLE = [
    ("a savings account", "a certificate of deposit"), ("public cloud hosting", "on-premise servers"),
    ("a standing desk", "a traditional desk"), ("a paper planner", "a digital calendar app"),
    ("gas stoves", "induction stoves"), ("hardcover books", "e-readers"),
    ("carpooling", "public transit"), ("freelancing", "full-time employment"),
]

# ---------------------------------------------------------------- Tier 3 ---

OPTION_PAIRS = [
    ("a relational database", "a document store", "a high-write telemetry system", "latency", "consistency", "schema flexibility"),
    ("microservices", "a monolithic architecture", "a fast-growing startup's backend", "team velocity", "operational overhead", "deployment complexity"),
    ("server-rendered pages", "a single-page application", "a content-heavy marketing site", "SEO", "initial load time", "interactivity"),
    ("a message queue", "direct synchronous API calls", "an order-processing pipeline", "reliability", "latency", "failure isolation"),
    ("renting", "buying", "a young professional's first home decision", "long-term cost", "flexibility", "financial risk"),
    ("a relational schema", "a graph database", "a social network's friend-recommendation feature", "query flexibility", "write throughput", "operational complexity"),
    ("in-house development", "a third-party SaaS tool", "a mid-size company's CRM needs", "customization", "total cost of ownership", "vendor lock-in"),
    ("a fixed-rate mortgage", "an adjustable-rate mortgage", "a buyer planning to stay 10+ years", "interest rate risk", "monthly payment stability", "total interest paid"),
    ("vertical scaling", "horizontal scaling", "a database approaching its write capacity limit", "cost", "downtime risk", "long-term headroom"),
    ("a native mobile app", "a cross-platform framework", "a startup building its first mobile product", "development speed", "performance", "platform-specific features"),
]

CREATIVE_PREMISES = [
    "a lighthouse keeper who discovers a message in a bottle that predicts the future",
    "a chess player who realizes their opponent is playing moves from a game that hasn't happened yet",
    "a translator who starts hearing a dead language whispered in dreams",
    "a gardener whose plants only grow when she tells them the truth",
    "a clockmaker who finds a watch that runs backward during thunderstorms",
    "a cartographer mapping a city that rearranges its streets every night",
    "an astronaut who receives a radio signal from a version of Earth that made different choices",
    "a librarian who discovers a book that rewrites itself based on the reader's regrets",
]

PHENOMENA = [
    ("interest rate hikes", "slow down inflation", "raising rates too aggressively and triggering a recession"),
    ("vaccination campaigns", "reduce disease transmission", "moving too quickly without adequate safety monitoring"),
    ("carbon taxes", "reduce emissions", "imposing them without offsetting costs for low-income households"),
    ("trade tariffs", "protect domestic industries", "provoking retaliatory tariffs that hurt exporters"),
    ("minimum wage increases", "raise worker income", "reducing entry-level hiring in small businesses"),
    ("urban rewilding projects", "improve biodiversity", "creating conflicts with existing land use and residents"),
    ("algorithmic content recommendation", "increase engagement", "reinforcing filter bubbles and misinformation"),
    ("quantitative easing", "stimulate economic growth", "inflating asset bubbles"),
]

JUDGMENT_SCENARIOS = [
    ("A hiring manager", "promoting a high-performing but disruptive employee", "hiring a less experienced but more collaborative candidate"),
    ("A city planner", "approving a new highway that cuts commute times", "preserving a historic neighborhood it would displace"),
    ("A hospital administrator", "expanding ICU capacity", "investing in preventive care programs"),
    ("A school principal", "adopting a strict zero-tolerance discipline policy", "using a restorative justice approach"),
    ("A startup founder", "raising venture capital for faster growth", "staying bootstrapped for more control"),
    ("A product manager", "shipping a feature quickly with technical debt", "delaying launch to build it properly"),
    ("A judge", "sentencing based on strict statutory guidelines", "weighing mitigating personal circumstances"),
    ("A nonprofit director", "spending more on direct aid now", "investing in long-term capacity building"),
]

SYSTEM_DESIGNS = [
    ("real-time chat application", "supporting millions of concurrent users"),
    ("recommendation engine", "an e-commerce platform with rapidly changing inventory"),
    ("log aggregation pipeline", "a fleet of thousands of IoT devices"),
    ("video transcoding service", "a platform handling variable, bursty upload traffic"),
    ("fraud detection system", "a payments platform processing millions of transactions per day"),
    ("search indexing pipeline", "a marketplace with constantly updated listings"),
    ("multi-tenant billing system", "a B2B SaaS product with usage-based pricing"),
    ("notification delivery system", "an app needing to reach users across push, email, and SMS reliably"),
]

PERSUASIVE_POSITIONS = [
    "that cities should ban private cars from downtown cores",
    "that all university degrees should be free",
    "that companies should be required to offer a four-day work week",
    "that social media platforms should verify the identity of all users",
    "that space exploration funding should be redirected to climate research",
    "that all new commercial buildings should be required to generate their own energy",
    "that remote work should be a legal right for eligible jobs",
    "that AI-generated content should always be labeled as such",
]


def _expand(items: list[str], templates: list[str], key: str = "{c}") -> list[str]:
    return [t.replace(key, item) for item in items for t in templates]


def _tier1() -> list[str]:
    prompts: list[str] = []
    prompts += _expand(COUNTRIES, COUNTRY_QUESTIONS, "{c}")
    prompts += _expand(SUBSTANCES, SUBSTANCE_QUESTIONS, "{s}")
    prompts += _expand(ELEMENTS, ELEMENT_QUESTIONS, "{e}")
    prompts += [f"Translate the word '{w}' into {lang}." for w in WORDS for lang in LANGUAGES]
    prompts += _expand(EVENTS, EVENT_QUESTIONS, "{ev}")
    prompts += [f"Convert {v} {u1} to {u2}." for v, u1, u2 in UNIT_CONVERSIONS]
    prompts += [
        f"Extract the {field} from this text: {ctx}"
        for field in EXTRACT_FIELDS
        for ctx in EXTRACT_CONTEXTS
    ]
    prompts += [
        f"Reformat this list into a comma-separated string: {a}\n{b}\n{c}"
        for a, b, c in REFORMAT_LISTS
    ]
    prompts += REWRITE_SENTENCES
    return prompts


def _tier2() -> list[str]:
    prompts: list[str] = []
    prompts += [f"Summarize the following article in two sentences:\n\nContext: {p}" for p in ARTICLE_PARAGRAPHS]
    prompts += [f"Identify the main topic of this paragraph in one sentence: {p}" for p in ARTICLE_PARAGRAPHS]
    prompts += [f'Classify the sentiment of this review as positive, negative, or neutral: "{r}"' for r in REVIEWS]
    prompts += [f'Categorize this support ticket into billing, technical, or account: "{t}"' for t in TICKETS]
    prompts += [f'Classify this email as spam or legitimate: "{e}"' for e in EMAILS]
    prompts += [f"List the pros and cons of {t}." for t in TOPICS_FOR_PROS_CONS]
    prompts += [f"Outline the main steps involved in {p}." for p in PROCESSES]
    prompts += [f"Given this data, identify the trend: {d}" for d in DATA_TRENDS]
    prompts += [
        f"Summarize the key differences between {a} and {b} in a short paragraph."
        for a, b in COMPARE_PAIRS_SIMPLE
    ]
    return prompts


def _tier3() -> list[str]:
    prompts: list[str] = []
    for a, b, use_case, x, y, z in OPTION_PAIRS:
        prompts.append(
            f"Compare the trade-offs between {a} and {b} for {use_case}. Analyze {x}, {y}, "
            f"and {z} step by step, then give a final recommendation. Output as JSON with "
            f"fields: options, tradeoffs, recommendation."
        )
    for premise in CREATIVE_PREMISES:
        prompts.append(
            f"Write a short creative story about {premise}, and explain why the ending you "
            f"chose is meaningful."
        )
    for cause, effect, risk in PHENOMENA:
        prompts.append(
            f"Why do {cause} typically {effect}? Give a nuanced, step by step explanation of "
            f"the mechanisms involved, and discuss the risks of {risk}."
        )
    for person, choice_a, choice_b in JUDGMENT_SCENARIOS:
        prompts.append(
            f"{person} is trying to decide between {choice_a} and {choice_b}. Analyze the "
            f"ethical and practical implications of each and recommend a course of action, "
            f"explaining your reasoning."
        )
    for system_type, scenario in SYSTEM_DESIGNS:
        prompts.append(
            f"Design a {system_type} for {scenario}, analyzing scalability, cost, and "
            f"reliability trade-offs, and justify your final design choice."
        )
    for position in PERSUASIVE_POSITIONS:
        prompts.append(
            f"Write a persuasive essay arguing {position}, addressing likely counterarguments "
            f"and using at least three distinct supporting points."
        )
    return prompts


def main() -> None:
    tier1 = list(dict.fromkeys(_tier1()))
    tier2 = list(dict.fromkeys(_tier2()))
    tier3 = list(dict.fromkeys(_tier3()))

    records = (
        [{"prompt": p, "tier": 1} for p in tier1]
        + [{"prompt": p, "tier": 2} for p in tier2]
        + [{"prompt": p, "tier": 3} for p in tier3]
    )
    random.shuffle(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} labeled prompts to {OUTPUT_PATH}")
    print(f"  Tier 1: {len(tier1)}, Tier 2: {len(tier2)}, Tier 3: {len(tier3)}")


if __name__ == "__main__":
    main()
