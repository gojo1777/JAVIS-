import json
import random
import urllib.request

def build_dataset():
    print("Dataset එක සාදමින් පවතී...")
    data = []

    # 1. සිංහල සහ Singlish ප්‍රශ්න සහ උත්තර
    topics = {
        "python": [
            "Python කියන්නේ මොකක්ද?", "python gana kiyaham", "python igena ganne kohomada", 
            "python igena ganna hoda විදිය මොකක්ද", "what is python"
        ],
        "ai": [
            "AI කියන්නේ මොකක්ද?", "artificial intelligence gana kiyaham", "ai igena ganne kohomada", 
            "ai walin mokada wenne", "what is ai"
        ],
        "machine_learning": [
            "Machine Learning කියන්නේ මොකක්ද?", "ml gana kiyaham", "machine learning igena ganne kohomada"
        ],
        "greetings": [
            "hi", "hello", "hey", "kohomada", "moko karanne", "mk", "kohomada yaaluwa"
        ],
        "about": [
            "oya kowda", "oyage nama mokakda", "who are you", "oyata mowa කරන්න පුලුවන්ද"
        ],
        "thanks": [
            "thanks", "sthuthi", "bohoma sthuthi", "elakiri", "super"
        ],
        "sinhala": [
            "සින්හලන් පුලුවන්ද", "sinhala puluwanda", "sinhala dannawada", "sinhala katha karanna puluwanda"
        ]
    }

    responses = {
        "python": "Python කියන්නේ ඉතා සරල, භාවිතයට පහසු සහ ජනප්‍රිය Programming Language එකක්. AI, Data Science සහ Web Development වලට මේක ගොඩක් භාවිතා කරනවා.",
        "ai": "AI (Artificial Intelligence) නැතහොත් කෘත්‍රිම බුද්ධිය කියන්නේ මිනිසුන් වගේ සිතන්න සහ තීරණ ගන්න යන්ත්‍ර සෑදීමේ තාක්ෂණයයි.",
        "machine_learning": "Machine Learning යනු AI හි කොටසකි. එහිදී පරිගණකය දත්ත (data) භාවිතයෙන් තනිවම ඉගෙන ගනී.",
        "greetings": "හලෝ! සුබ දවසක්. මම ඔයාට කොහොමද උදව් කරන්න ඕනේ?",
        "about": "මම JAVIS. ඔයාගේ ප්‍රශ්නවලට සහ Programming වලට උදව් කරන්න හදපු AI Assistant කෙනෙක්.",
        "thanks": "ඔයාට උදව් කරන්න ලැබීම සතුටක්! වෙන මොනවා හරි දැනගන්න ඕන නම් අහන්න.",
        "sinhala": "ඔව්, මට සිංහලෙන් සහ Singlish වලින් කතා කරන්න පුළුවන්. ඔයාට මොකක්ද දැනගන්න ඕනේ?"
    }

    prefixes = ["", "pls ", "can you tell me ", "අනේ ", "මචන් ", "bro "]
    suffixes = ["", " pls", " kiyaham", " මචන්", " bro", " ශෝට් එකෙන් කියන්න"]

    # වචන වෙනස් කරමින් combinations සෑදීම
    for key in topics:
        for item in topics[key]:
            resp = responses[key]
            for p in prefixes:
                for s in suffixes:
                    prompt_text = f"{p}{item}{s}".strip()
                    data.append({"prompt": prompt_text, "response": resp})

    # 2. Open-source coding dataset එකක් Internet එකෙන් Download කිරීම
    print("Internet එකෙන් අමතර Data ලබා ගනිමින් පවතී...")
    dataset_url = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"

    try:
        req = urllib.request.urlopen(dataset_url)
        raw_data = json.loads(req.read().decode('utf-8'))
        
        for item in raw_data[:1000]:
            prompt_text = item.get("instruction", "")
            if item.get("input"):
                prompt_text += " " + item.get("input")
                
            response_text = item.get("output", "")
            if prompt_text and response_text:
                data.append({"prompt": prompt_text, "response": response_text})

    except Exception as e:
        print(f"Online dataset ලබා ගැනීමට නොහැකි විය: {e}")

    random.shuffle(data)

    # data/conversations.jsonl එකට ලියයි
    output_path = "data/conversations.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"සාර්ථකයි! Dataset එක සෑදුවා. මුළු ප්‍රමාණය: {len(data)}")

if __name__ == "__main__":
    build_dataset()
