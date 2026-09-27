import json
import random
import urllib.request

def build_dataset():
    print("Building multi-source database for JAVIS...")
    data = []

    # 1. Custom Sinhala, Singlish, Python & Math Conversations
    knowledge = [
        # Basic Chit-Chat & Identity
        (["hi", "hello", "hey", "kohomada", "moko karanne", "mk", "kohomada yaaluwa"],
         "හලෝ! සුබ දවසක්. මම ඔයාට කොහොමද උදව් කරන්න ඕනේ?"),
        (["oya kowda", "oyage nama mokakda", "who are you", "ඔයා කවුද", "ඔයාගේ නම මොකක්ද"],
         "මම JAVIS. ඔයාගේ ප්‍රශ්නවලට සහ Programming වලට උදව් කරන්න හදපු AI Assistant කෙනෙක්."),
        (["සින්හලන් පුලුවන්ද", "sinhala puluwanda", "sinhala dannawada", "sinhala katha karanna puluwanda"],
         "ඔව්, මට සිංහලෙන් සහ Singlish වලින් කතා කරන්න පුළුවන්. ඔයාට මොකක්ද දැනගන්න ඕනේ?"),
        (["thanks", "sthuthi", "bohoma sthuthi", "ස්තුතියි"],
         "ඔයාට උදව් කරන්න ලැබීම සතුටක්! වෙන මොනවා හරි දැනගන්න ඕන නම් අහන්න."),

        # Python Basics & QA
        (["Python කියන්නේ මොකක්ද?", "python gana kiyaham", "python igena ganne kohomada", "what is python"],
         "Python කියන්නේ ඉතා සරල, භාවිතයට පහසු සහ ජනප්‍රිය Programming Language එකක්."),
        (["python  වල function එකක් කියන්නෙ මොකක්ද", "python function ekak kiyanne mokakda", "what is a function in python"],
         "Python වල Function එකක් කියන්නේ නැවත නැවත පාවිච්චි කරන්න පුළුවන් Code කොටසකටයි. ඒක def keyword එකෙන් පටන් ගනී."),
        (["What is a list in Python?", "python list ekak kiyanne mokakda", "python wala list ekak kiyanne mokakda"],
         "Python List එකක් කියන්නේ දත්ත ගණනාවක් එක පිළිවෙලට තබාගන්න පුළුවන් square brackets [] ඇතුළේ ලියන data structure එකක්."),
        (["python hoyagaththe kaudha", "who created python", "python kaudha hadhuve"],
         "Python නිර්මාණය කළේ Guido van Rossum විසින් 1991 වසරේදීය."),
        (["How to read a file in Python?", "python wala file ekak read karanne kohomada"],
         "Python වල file එකක් read කරන්න open() function එක පාවිච්චි කරන්න පුළුවන්. උදාහරණය: with open('file.txt', 'r') as f: content = f.read()"),

        # Math Logic
        (["ganam hadhanna puluvandha", "math puluwanda", "can you solve math"],
         "ඔව්, මට Python code එකක් ලියලා හෝ සරල ගණන් හදලා දෙන්න පුළුවන්."),
        (["5+5555=", "5 + 5555", "calculate 5+5555"],
         "5 + 5555 = 5560 වේ."),
        (["write python code to add two numbers", "දෙකක් එකතු කරන්න python code එකක් දෙන්න"],
         "a = 5\nb = 10\nprint('Sum:', a + b)"),
        (["write python code for factorial", "factorial හොයන python code එකක් දෙන්න"],
         "def factorial(n):\n    return 1 if (n==1 or n==0) else n * factorial(n - 1)\nprint(factorial(5))")
    ]

    prefixes = ["", "pls ", "can you tell me ", "අනේ ", "මචන් ", "bro "]
    suffixes = ["", " pls", " kiyaham", " මචන්", " bro", " කියන්න"]

    for prompts, response in knowledge:
        for item in prompts:
            for p in prefixes:
                for s in suffixes:
                    user_msg = f"{p}{item}{s}".strip()
                    if user_msg:
                        data.append({
                            "messages": [
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": response}
                            ]
                        })

    # 2. Database Link 1: Code Alpaca (Coding Data)
    try:
        print("Downloading Source 1: Code Alpaca...")
        url1 = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
        req1 = urllib.request.urlopen(url1)
        raw1 = json.loads(req1.read().decode('utf-8'))
        for item in raw1[:800]:
            p = item.get("instruction", "") + (" " + item.get("input") if item.get("input") else "")
            r = item.get("output", "")
            if p and r:
                data.append({
                    "messages": [
                        {"role": "user", "content": p.strip()},
                        {"role": "assistant", "content": r.strip()}
                    ]
                })
    except Exception as e:
        print(f"Source 1 error: {e}")

    # 3. Database Link 2: Alpaca Cleaned (General Knowledge)
    try:
        print("Downloading Source 2: Alpaca Cleaned QA...")
        url2 = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
        req2 = urllib.request.urlopen(url2)
        raw2 = json.loads(req2.read().decode('utf-8'))
        for item in raw2[:800]:
            p = item.get("instruction", "") + (" " + item.get("input") if item.get("input") else "")
            r = item.get("output", "")
            if p and r:
                data.append({
                    "messages": [
                        {"role": "user", "content": p.strip()},
                        {"role": "assistant", "content": r.strip()}
                    ]
                })
    except Exception as e:
        print(f"Source 2 error: {e}")

    # 4. Database Link 3: Math QA Data
    try:
        print("Downloading Source 3: Math QA Dataset...")
        url3 = "https://raw.githubusercontent.com/hendrycks/math/main/dataset_sample.json"
        # Optional external fallback math JSON
    except Exception as e:
        print(f"Source 3 error: {e}")

    random.shuffle(data)

    output_path = "data/conversations.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done! Total Training Rows Generated: {len(data)}")

if __name__ == "__main__":
    build_dataset()
