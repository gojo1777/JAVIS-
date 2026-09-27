import json
import random
import urllib.request
from pathlib import Path

OUTPUT_PATH = Path("data/conversations.jsonl")
SEED = 42

random.seed(SEED)


def add_example(data, user, assistant, source):
    user = str(user).strip()
    assistant = str(assistant).strip()

    if not user or not assistant:
        return

    data.append({
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ],
        "_source": source
    })


def add_custom_data(data):
    count_before = len(data)

    examples = [

        # =========================
        # GENERAL / IDENTITY
        # =========================

        ("hi", "හලෝ! සුබ දවසක්. මම JAVIS. ඔයාට මොනවද දැනගන්න ඕනේ?"),
        ("hello", "හලෝ! මම JAVIS. ඔයාට උදව් කරන්න පුළුවන්."),
        ("hey", "හලෝ! මොකද වෙන්නේ?"),
        ("kohomada", "මම හොඳින්. ඔයාට කොහොමද?"),
        ("moko karanne", "මම ඔයාගේ ප්‍රශ්නවලට උත්තර දෙන්න ලෑස්තියි."),
        ("oya kawda", "මම JAVIS. Programming සහ සාමාන්‍ය ප්‍රශ්නවලට උදව් කරන AI Assistant කෙනෙක්."),
        ("oyage nama mokakda", "මගේ නම JAVIS."),
        ("who are you", "I am JAVIS, an AI assistant designed to help with questions, programming and learning."),
        ("ඔයා කවුද", "මම JAVIS. ඔයාට ප්‍රශ්නවලට සහ Programming වලට උදව් කරන්න පුළුවන්."),
        ("සින්හලෙන් කතා කරන්න පුළුවන්ද", "ඔව්. මට සිංහලෙන් සහ Singlish වලින් කතා කරන්න පුළුවන්."),
        ("sinhala puluwanda", "ඔව්, මට සිංහලෙන් කතා කරන්න පුළුවන්."),
        ("thanks", "ස්තුතියි! තවත් දෙයක් දැනගන්න ඕන නම් අහන්න."),
        ("sthuthi", "ඔයාට උදව් කරන්න ලැබීම සතුටක්."),
        ("bye", "හරි. නැවත හමුවෙමු."),

        # =========================
        # PYTHON
        # =========================

        (
            "Python කියන්නේ මොකක්ද?",
            "Python කියන්නේ ඉගෙනගන්න පහසු, ජනප්‍රිය programming language එකක්."
        ),
        (
            "python gana kiyaham",
            "Python කියන්නේ සරල syntax එකක් තියෙන ජනප්‍රිය programming language එකක්."
        ),
        (
            "what is python",
            "Python is a popular high-level programming language known for its simple and readable syntax."
        ),
        (
            "python igena ganne kohomada",
            "මුලින් variables, data types, conditions, loops සහ functions ඉගෙනගෙන පොඩි projects කරන්න."
        ),
        (
            "python function ekak kiyanne mokakda",
            "Function එකක් කියන්නේ නැවත නැවත භාවිතා කළ හැකි code කොටසක්. Python වල function එකක් සාමාන්‍යයෙන් def keyword එකෙන් define කරනවා."
        ),
        (
            "what is a function in python",
            "A Python function is a reusable block of code defined using the def keyword."
        ),
        (
            "python list ekak kiyanne mokakda",
            "Python List එකක් කියන්නේ values කිහිපයක් එකට තබාගන්න භාවිතා කරන ordered collection එකක්. උදාහරණයක්: [1, 2, 3]."
        ),
        (
            "What is a list in Python?",
            "A Python list is an ordered, mutable collection that can store multiple values."
        ),
        (
            "python dictionary ekak kiyanne mokakda",
            "Dictionary එකක් key සහ value යුගල ලෙස data තබාගන්න Python data structure එකක්. උදාහරණයක්: {'name': 'JAVIS'}."
        ),
        (
            "python loop ekak kiyanne mokakda",
            "Loop එකක් භාවිතා කරන්නේ එකම code එක නැවත නැවත execute කිරීමටයි. Python වල for සහ while loops තියෙනවා."
        ),
        (
            "python if statement eka mokakda",
            "if statement එකක් condition එකක් පරීක්ෂා කරලා condition එක true නම් code එකක් execute කරන්න භාවිතා කරනවා."
        ),
        (
            "python variable ekak kiyanne mokakda",
            "Variable එකක් කියන්නේ data එකක් නමක් යටතේ store කරගන්න භාවිතා කරන identifier එකක්."
        ),
        (
            "python file ekak read karanne kohomada",
            "Python වල file එකක් read කරන්න open() භාවිතා කරන්න පුළුවන්. උදාහරණය: with open('file.txt', 'r') as f: content = f.read()"
        ),
        (
            "How to read a file in Python?",
            "Use Python's open() function. For example: with open('file.txt', 'r') as f: content = f.read()"
        ),
        (
            "python වල error එකක් කියන්නේ මොකක්ද",
            "Error එකක් කියන්නේ program එක execute කිරීමේදී ඇතිවන ගැටලුවක්. Python වල SyntaxError, TypeError සහ ValueError වගේ errors තියෙනවා."
        ),
        (
            "python class එකක් කියන්නේ මොකක්ද",
            "Class එකක් කියන්නේ objects නිර්මාණය කිරීමට භාවිතා කරන blueprint එකක්."
        ),
        (
            "python module එකක් කියන්නේ මොකක්ද",
            "Module එකක් කියන්නේ Python code එක අඩංගු file එකක්. import keyword එකෙන් module එකක් භාවිතා කරන්න පුළුවන්."
        ),

        # =========================
        # PYTHON CODE
        # =========================

        (
            "write python code to add two numbers",
            "a = 5\nb = 10\nprint(a + b)"
        ),
        (
            "දෙකක් එකතු කරන්න python code එකක් දෙන්න",
            "a = 5\nb = 10\nprint(a + b)"
        ),
        (
            "write python code for factorial",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))"
        ),
        (
            "python වල numbers 1 සිට 10 print කරන්නේ කොහොමද",
            "for i in range(1, 11):\n    print(i)"
        ),
        (
            "python වල hello world code එක දෙන්න",
            "print('Hello, World!')"
        ),
        (
            "python වල list එකක් හදන්නේ කොහොමද",
            "numbers = [1, 2, 3, 4, 5]"
        ),
        (
            "python වල function එකක් ලියන්න",
            "def greet(name):\n    return 'Hello ' + name\n\nprint(greet('JAVIS'))"
        ),
        (
            "python වල even numbers හොයන්නේ කොහොමද",
            "for i in range(1, 21):\n    if i % 2 == 0:\n        print(i)"
        ),

        # =========================
        # MATH
        # =========================

        ("5+5555=", "5 + 5555 = 5560."),
        ("5 + 5555", "5 + 5555 = 5560."),
        ("10 + 20", "10 + 20 = 30."),
        ("100 - 25", "100 - 25 = 75."),
        ("12 * 5", "12 × 5 = 60."),
        ("100 / 4", "100 ÷ 4 = 25."),
        ("2 + 3 * 4", "2 + 3 × 4 = 14."),
        ("square of 5", "5² = 25."),
        ("10 squared", "10² = 100."),
        ("math puluwanda", "ඔව්. සරල ගණිත ගැටලු විසඳීමට මට පුළුවන්."),
        ("ganam hadhanna puluvandha", "ඔව්. සරල ගණිත ගැටලු විසඳීමට මට පුළුවන්."),

    ]

    for user, assistant in examples:
        add_example(data, user, assistant, "custom")

    return len(data) - count_before


def add_math_dataset(data):
    count_before = len(data)

    # Deterministic arithmetic training examples.
    for a in range(1, 101):
        for b in range(1, 11):

            add_example(
                data,
                f"{a} + {b}",
                f"{a} + {b} = {a + b}.",
                "math"
            )

            add_example(
                data,
                f"{a} plus {b}",
                f"{a} + {b} = {a + b}.",
                "math"
            )

    for a in range(10, 101):
        for b in range(1, min(a, 11)):

            add_example(
                data,
                f"{a} - {b}",
                f"{a} - {b} = {a - b}.",
                "math"
            )

    for a in range(1, 31):
        for b in range(1, 11):

            add_example(
                data,
                f"{a} * {b}",
                f"{a} × {b} = {a * b}.",
                "math"
            )

    return len(data) - count_before


def download_json(url, timeout=20):
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "JAVIS-Dataset-Builder/1.0"}
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()

        return json.loads(raw.decode("utf-8"))

    except Exception as e:
        print(f"Download failed: {e}")
        return None


def add_code_alpaca(data):
    url = (
        "https://raw.githubusercontent.com/"
        "sahil280114/codealpaca/master/data/code_alpaca_20k.json"
    )

    print("Downloading Code Alpaca...")

    raw = download_json(url)

    if not raw:
        return 0

    count_before = len(data)

    for item in raw[:1000]:

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        prompt = instruction

        if input_text:
            prompt += "\n" + input_text

        add_example(
            data,
            prompt,
            output,
            "code_alpaca"
        )

    return len(data) - count_before


def add_alpaca(data):
    url = (
        "https://raw.githubusercontent.com/"
        "gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
    )

    print("Downloading Alpaca QA...")

    raw = download_json(url)

    if not raw:
        return 0

    count_before = len(data)

    for item in raw[:1000]:

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        prompt = instruction

        if input_text:
            prompt += "\n" + input_text

        add_example(
            data,
            prompt,
            output,
            "alpaca"
        )

    return len(data) - count_before


def remove_duplicates(data):
    seen = set()
    cleaned = []

    for item in data:

        messages = item.get("messages", [])

        if len(messages) != 2:
            continue

        user = messages[0].get("content", "").strip().lower()
        assistant = messages[1].get("content", "").strip()

        key = (user, assistant)

        if key in seen:
            continue

        seen.add(key)

        cleaned.append({
            "messages": messages
        })

    return cleaned


def validate_dataset(data):
    valid = []

    for item in data:

        messages = item.get("messages", [])

        if len(messages) != 2:
            continue

        if messages[0].get("role") != "user":
            continue

        if messages[1].get("role") != "assistant":
            continue

        user = messages[0].get("content", "").strip()
        assistant = messages[1].get("content", "").strip()

        if len(user) < 1:
            continue

        if len(assistant) < 1:
            continue

        valid.append(item)

    return valid


def build_dataset():

    print("=" * 60)
    print("JAVIS EXPANDED DATASET BUILDER")
    print("=" * 60)

    data = []

    print("\n[1/5] Adding custom Sinhala / Singlish / Python data...")
    custom_count = add_custom_data(data)
    print(f"Custom examples: {custom_count}")

    print("\n[2/5] Generating mathematics dataset...")
    math_count = add_math_dataset(data)
    print(f"Math examples: {math_count}")

    print("\n[3/5] Downloading Code Alpaca...")
    code_count = add_code_alpaca(data)
    print(f"Code Alpaca examples: {code_count}")

    print("\n[4/5] Downloading Alpaca QA...")
    alpaca_count = add_alpaca(data)
    print(f"Alpaca examples: {alpaca_count}")

    print("\n[5/5] Cleaning dataset...")

    before = len(data)

    data = remove_duplicates(data)
    data = validate_dataset(data)

    after = len(data)

    print(f"Before cleaning: {before}")
    print(f"After cleaning:  {after}")
    print(f"Removed:         {before - after}")

    random.shuffle(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        for item in data:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Total rows: {len(data)}")

    print("\nSource summary:")

    source_counts = {}

    for item in data:
        # Source metadata is removed before saving,
        # so classify mainly from content when reporting is not possible.
        source_counts["training_rows"] = source_counts.get(
            "training_rows", 0
        ) + 1

    print(f"Training rows: {source_counts['training_rows']}")

    print("\nNext step:")
    print("python training/train.py")


if __name__ == "__main__":
    build_dataset()
