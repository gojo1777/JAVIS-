import json
import random
import urllib.request

def build_dataset():
    print("Building multi-source dataset for JAVIS...")
    data = []

    # A. Custom Sinhala & Singlish Conversations
    knowledge = [
        (["hi", "hello", "hey", "kohomada", "moko karanne", "mk", "kohomada yaaluwa"],
         "හලෝ! සුබ දවසක්. මම ඔයාට කොහොමද උදව් කරන්න ඕනේ?"),
        (["oya kowda", "oyage nama mokakda", "who are you", "ඔයා කවුද", "ඔයාගේ නම මොකක්ද"],
         "මම JAVIS. ඔයාගේ ප්‍රශ්නවලට සහ Programming වලට උදව් කරන්න හදපු AI Assistant කෙනෙක්."),
        (["සින්හලන් පුලුවන්ද", "sinhala puluwanda", "sinhala dannawada", "sinhala katha karanna puluwanda"],
         "ඔව්, මට සිංහලෙන් සහ Singlish වලින් කතා කරන්න පුළුවන්. ඔයාට මොකක්ද දැනගන්න ඕනේ?"),
        (["Python කියන්නේ මොකක්ද?", "python gana kiyaham", "python igena ganne kohomada", "what is python"],
         "Python කියන්නේ ඉතා සරල, භාවිතයට පහසු සහ ජනප්‍රිය Programming Language එකක්."),
        (["AI කියන්නේ මොකක්ද?", "artificial intelligence gana kiyaham", "ai igena ganne kohomada", "what is ai"],
         "AI (Artificial Intelligence) නැතහොත් කෘත්‍රිම බුද්ධිය කියන්නේ මිනිසුන් වගේ සිතන්න සහ තීරණ ගන්න යන්ත්‍ර සෑදීමේ තාක්ෂණයයි."),
        (["thanks", "sthuthi", "bohoma sthuthi", "ස්තුතියි"],
         "ඔයාට උදව් කරන්න ලැබීම සතුටක්! වෙන මොනවා හරි දැනගන්න ඕන නම් අහන්න.")
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

    # B. Source 1: Code Alpaca
    try:
        print("Downloading Source 1: Code Alpaca...")
        url1 = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
        req1 = urllib.request.urlopen(url1)
        raw1 = json.loads(req1.read().decode('utf-8'))
        for item in raw1[:1000]:
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

    # C. Source 2: Alpaca Cleaned
    try:
        print("Downloading Source 2: Alpaca Cleaned QA...")
        url2 = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
        req2 = urllib.request.urlopen(url2)
        raw2 = json.loads(req2.read().decode('utf-8'))
        for item in raw2[:1000]:
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

    random.shuffle(data)

    output_path = "data/conversations.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done! Total Training Rows Generated: {len(data)}")

if __name__ == "__main__":
    build_dataset()
