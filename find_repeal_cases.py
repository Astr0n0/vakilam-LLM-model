import json
from collections import defaultdict

PATH = "law-rag/data/civil-law.json"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

groups = defaultdict(list)

for item in data:
    groups[item["article"]].append(item)


def parse_date(version):
    if not version:
        return None

    parts = version.split()

    if len(parts) < 2:
        return None

    date = parts[-1]

    try:
        y, m, d = map(int, date.split("/"))
        return (y, m, d)
    except ValueError:
        return None


for article, records in groups.items():

    events = []

    for record in records:

        version = record.get("version")

        if not version:
            continue

        version_date = parse_date(version)

        if not version_date:
            continue

        if "حذف" in version:
            event_type = "repealed"

        elif "اصلاحی" in version:
            event_type = "amendment"

        elif "الحاقی" in version:
            event_type = "addition"

        elif "مصوب" in version:
            event_type = "original"

        else:
            continue

        events.append(
            (
                version_date,
                event_type,
                version,
            )
        )

    if not events:
        continue

    events.sort()

    repeal_events = [
        event
        for event in events
        if event[1] == "repealed"
    ]

    amendment_events = [
        event
        for event in events
        if event[1] in {"amendment", "addition"}
    ]

    if repeal_events and amendment_events:

        latest_event = events[-1]

        print("=" * 70)
        print(f"Article {article}")

        for event in events:
            print(
                f"  {event[0]} | "
                f"{event[1]:10} | "
                f"{event[2]}"
            )

        print(f"  LATEST EVENT: {latest_event}")
        print()