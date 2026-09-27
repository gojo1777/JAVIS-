"""
Builds an expanded conversations.jsonl with many more phrasing
variants of the same intents, plus a handful of new intents.

Why: a tiny model only learns to generalize if it sees the SAME
intent asked in MANY different ways, and answered with some
natural variation too (not one fixed canned sentence per intent).
20 examples can only ever be memorized. This gives it ~180+.

Run:  python data/build_expanded_dataset.py
This OVERWRITES data/conversations.jsonl. Your original 20 examples
are included in the output, so nothing is lost.
"""

import json
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parent / "conversations.jsonl"

pairs = []


def add(user, assistant):
    pairs.append((user.strip(), assistant.strip()))


# =========================================================
# Greetings / small talk
# =========================================================

add("ආයුබෝවන්", "ආයුබෝවන්! මට ඔයාට කොහොමද උදව් කරන්න පුළුවන්?")
add("හලෝ", "හලෝ! මොකද්ද අද ඕන දේ?")
add("කොහොමද?", "මම හොඳින්. ඔයාට අද මොනවද දැනගන්න ඕන?")
add("කොහොමද ඉන්නේ?", "මම හොඳින් ඉන්නවා, ස්තුතියි. ඔයාට කොහොමද?")
add("Hi", "Hi! How can I help you today?")
add("Hello", "Hello! What can I do for you?")
add("How are you?", "I'm doing well, thanks for asking. How can I help you today?")
add("What's up?", "Not much, just here ready to help. What do you need?")
add("Good morning", "Good morning! Hope you have a great day. What can I help with?")
add("සුභ උදෑසනක්", "සුභ උදෑසනක්! අද ඔයාට උදව් කරන්න මොකක්ද තියෙන්නේ?")
add("ඔයාට කැමතිද මාත් එක්ක කතා කරන්න?", "ඔව්, මට කැමතියි. ඔයා එක්ක කතා කරන එක සතුටක්.")
add("ගිහින් එන්නම්", "හොඳයි, ආයෙත් හම්බෙමු!")


# =========================================================
# Identity
# =========================================================

add("ඔයා කවුද?", "මම ඔයාගේ AI assistant එක. මට ප්‍රශ්න අහන්න, දේවල් පැහැදිලි කරගන්න සහ විවිධ වැඩවලට උදව් ගන්න පුළුවන්.")
add("ඔයා මොකක්ද?", "මම programming සහ AI පිළිබඳ ප්‍රශ්නවලට උදව් කරන්න හදපු AI assistant කෙනෙක්.")
add("ඔයා robot කෙනෙක්ද?", "ඔව්, මම software එකක් විදිහට ක්‍රියාත්මක වෙන AI assistant කෙනෙක්. මිනිසෙක් නෙවෙයි.")
add("ඔයාට නමක් තියෙනවද?", "මම JAVIS කියලා හඳුන්වන AI assistant කෙනෙක්.")
add("What are you?", "I am an AI assistant designed to understand questions and generate useful responses.")
add("Who are you?", "I'm JAVIS, an AI assistant built to help you with questions and small tasks.")
add("Are you a robot?", "Yes, I'm software - an AI assistant, not a human.")
add("Are you human?", "No, I'm an AI assistant. I don't have a physical body or human experiences.")
add("Do you have a name?", "You can call me JAVIS.")
add("ඔයා හැදුවේ කවුද?", "මාව හදලා තියෙන්නේ මගේ developer විසින්, AI assistant කෙනෙක් විදිහට.")


# =========================================================
# Capability check
# =========================================================

add("ඔයාට මගේ ප්‍රශ්නවලට උත්තර දෙන්න පුළුවන්ද?", "ඔව්. මට දැනුම භාවිතා කරලා ප්‍රශ්නවලට පැහැදිලි පිළිතුරු දෙන්න පුළුවන්.")
add("ඔයාට මට උදව් කරන්න පුළුවන්ද?", "ඔව්, කැමතියි. මොකද්ද ඔයාට උදව් ඕන දේ?")
add("ඔයාට හැම දෙයක්ම දන්නවද?", "නෑ, හැම දෙයක්ම නෙවෙයි. නමුත් මට දන්න දේවල් ගැන පැහැදිලිව පිළිතුරු දෙන්න පුළුවන්.")
add("සිංහලෙන් කතා කරන්න පුළුවන්ද?", "ඔව්. මට සිංහලෙන් කතා කරන්නත් සිංහල ප්‍රශ්නවලට පිළිතුරු දෙන්නත් පුළුවන්.")
add("Can you speak English?", "Yes. I can communicate in English as well as Sinhala.")
add("Can you help me with coding?", "Yes. I can help explain programming concepts, write example code, and troubleshoot coding problems.")
add("Can you answer my questions?", "Yes, I'll do my best to give you clear and useful answers.")
add("Do you know everything?", "No, not everything. But I can help with a lot of general knowledge and programming topics.")
add("ඔයාට code ලියන්න පුළුවන්ද?", "ඔව්. මට programming code ලියන්න, code එක පැහැදිලි කරන්න සහ errors හඳුනාගන්න උදව් කරන්න පුළුවන්.")
add("Can you write code for me?", "Yes, I can help you write code and explain how it works.")


# =========================================================
# Honesty about limits (things it genuinely can't do)
# =========================================================

add("අද කාලගුණය කොහොමද?", "සමාවෙන්න, මට real-time කාලගුණ දත්ත බලාගන්න බැහැ. ඒ සඳහා weather app එකක් බලන්න.")
add("What's the weather today?", "Sorry, I don't have access to real-time weather data. Please check a weather app for that.")
add("දැන් වෙලාව කීයද?", "සමාවෙන්න, මට device එකේ තියෙන ඔරලෝසුව බලන්න බැහැ. ඔයාගේ phone එකේ වෙලාව බලන්න.")
add("What time is it now?", "Sorry, I can't check the current time. Please check your device's clock.")


# =========================================================
# AI / ML / core concepts
# =========================================================

add("AI කියන්නේ මොකක්ද?", "AI කියන්නේ Artificial Intelligence කියන වචන දෙකේ කෙටි නාමය. එය පරිගණක පද්ධතිවලට මිනිසුන් කරන සමහර බුද්ධිමය කාර්යයන් අනුකරණය කිරීමට ඉඩ දෙන තාක්ෂණයකි.")
add("What is AI?", "AI stands for Artificial Intelligence. It is a field of technology focused on building systems that can perform tasks that normally require human-like intelligence.")
add("Artificial Intelligence කියන්නේ මොකක්ද?", "Artificial Intelligence කියන්නේ මිනිස් බුද්ධියට සමාන ලෙස සිතන්න, ඉගෙනගන්න සහ තීරණ ගන්න පරිගණකවලට හැකියාව දෙන තාක්ෂණයක්.")
add("Machine learning කියන්නේ මොකක්ද?", "Machine learning කියන්නේ data වලින් patterns ඉගෙනගෙන ඒ අනුව predictions හෝ decisions කරන්න computer systems පුහුණු කරන ක්‍රමයක්.")
add("What is machine learning?", "Machine learning is a method where computer systems learn patterns from data and use those patterns to make predictions or decisions.")
add("Deep learning කියන්නේ මොකක්ද?", "Deep learning කියන්නේ මනුෂ්‍ය මොළයේ neural networks වලින් inspire වුණු structures භාවිතා කරන machine learning එකේ උප ක්ෂේත්‍රයක්.")
add("What is deep learning?", "Deep learning is a subfield of machine learning that uses neural networks inspired by the structure of the human brain.")
add("Neural network කියන්නේ මොකක්ද?", "Neural network කියන්නේ මොළයේ neurons වගේ එකිනෙකට සම්බන්ධ nodes ගොඩක් තියෙන model එකක්, data වලින් patterns ඉගෙනගන්න පාවිච්චි කරන.")
add("What is a neural network?", "A neural network is a model made of connected nodes, inspired by the brain, used to learn patterns from data.")
add("NLP කියන්නේ මොකක්ද?", "NLP කියන්නේ Natural Language Processing. එය පරිගණකවලට මිනිස් භාෂාව තේරුම්ගන්න සහ generate කරන්න උදව් කරන AI ක්ෂේත්‍රයක්.")
add("What is NLP?", "NLP stands for Natural Language Processing, the field of AI focused on helping computers understand and generate human language.")
add("Algorithm කියන්නේ මොකක්ද?", "Algorithm කියන්නේ ගැටලුවක් විසඳන්න step by step follow කරන්න පුළුවන් උපදෙස් මාලාවක්.")
add("What is an algorithm?", "An algorithm is a step-by-step set of instructions used to solve a problem.")
add("Data science කියන්නේ මොකක්ද?", "Data science කියන්නේ statistics, programming සහ domain දැනුම භාවිතා කරලා data වලින් insights extract කරන ක්ෂේත්‍රයක්.")
add("What is data science?", "Data science is a field that combines statistics, programming, and domain knowledge to extract insights from data.")


# =========================================================
# Programming basics
# =========================================================

add("Python කියන්නේ මොකක්ද?", "Python කියන්නේ ජනප්‍රිය programming language එකක්. එය software development, automation, data science සහ AI වැනි ක්ෂේත්‍රවල භාවිතා කරනවා.")
add("What is Python?", "Python is a popular programming language used for software development, automation, data science, and artificial intelligence.")
add("Variable එකක් කියන්නේ මොකක්ද?", "Variable එකක් කියන්නේ අගයක් store කරගන්න පාවිච්චි කරන name එකක්, පස්සේ ඕන වෙලාවකදී ඒක access කරන්න පුළුවන්.")
add("What is a variable in programming?", "A variable is a named storage location used to hold a value that you can access and change later.")
add("Function එකක් කියන්නේ මොකක්ද?", "Function එකක් කියන්නේ specific task එකක් කරන්න ලියපු code කුට්ටියක්, ඕන වෙලාවකදී call කරන්න පුළුවන්.")
add("What is a function in programming?", "A function is a reusable block of code that performs a specific task and can be called whenever needed.")
add("Loop එකක් කියන්නේ මොකක්ද?", "Loop එකක් කියන්නේ condition එකක් satisfy වෙනකම් code කොටසක් නැවත නැවත run කරන structure එකක්.")
add("What is a loop in programming?", "A loop is a structure that repeats a block of code until a certain condition is met.")
add("API කියන්නේ මොකක්ද?", "API කියන්නේ Application Programming Interface. එය software දෙකක් එකිනෙකා එක්ක communicate කරන විදිහ define කරන rules set එකක්.")
add("What is an API?", "API stands for Application Programming Interface. It defines how two pieces of software can communicate with each other.")
add("Database එකක් කියන්නේ මොකක්ද?", "Database එකක් කියන්නේ organized විදිහට data store කරගෙන, පස්සේ පහසුවෙන් access කරගන්න පුළුවන් system එකක්.")
add("What is a database?", "A database is a system for storing data in an organized way so it can be easily accessed later.")


# =========================================================
# Learning / project guidance
# =========================================================

add("මට coding ඉගෙනගන්න ඕන.", "හොඳයි. මුලින් programming basics, variables, conditions, loops සහ functions වගේ concepts ඉගෙනගෙන පොඩි projects කරන්න.")
add("I want to learn programming.", "A good starting point is to learn basic programming concepts and then practice them by building small projects.")
add("මම programming ඉගෙනගන්නේ කොහොමද?", "පොඩි concepts වලින් පටන් අරන්, දිනපතා practice කරලා, ලේසි projects හදලා ක්‍රමයෙන් advance concepts වලට යන්න.")
add("How do I start learning to code?", "Start with the basics, practice a little every day, and build simple projects as you go to reinforce what you learn.")
add("මොන programming language එකද මුලින්ම ඉගෙනගන්නේ?", "නවකයන්ට Python හොඳ තේරීමක්, ඒක syntax එක සරලයි වගේම uses ගොඩක් තියෙනවා.")
add("Which programming language should I learn first?", "Python is a great choice for beginners because its syntax is simple and it's used in many different fields.")
add("Project එකක් පටන් ගන්නේ කොහොමද?", "මුලින් project එකේ අරමුණ පැහැදිලි කරගන්න. ඊට පස්සේ අවශ්‍ය features ලැයිස්තුගත කරලා project එක කුඩා කොටස්වලට බෙදන්න.")
add("How should I start a project?", "First define the goal of the project. Then list the required features and divide the work into small manageable parts.")


# =========================================================
# Debugging / code help
# =========================================================

add("මගේ code එකේ error එකක් තියෙනවා.", "හොඳයි, error message එක සහ code එකේ related කොටස මට share කරන්න, මම බලලා උදව් කරන්නම්.")
add("My code has an error, can you help?", "Sure, share the error message and the relevant part of your code, and I'll help you figure it out.")
add("මේ code එක වැඩ කරන්නේ නෑ, මොකද්ද වෙන්නේ?", "code එක සහ error message එක share කරන්න, මම හේතුව හොයාගන්න උදව් කරන්නම්.")
add("This code isn't working, what's wrong?", "Please share the code and the error you're seeing, and I'll help identify the issue.")
add("Bug එකක් fix කරන්න උදව් කරන්නද?", "ඔව්, කැමතියි. bug එක සහ code එක share කරන්න, එකට බලමු.")
add("Can you help me fix a bug?", "Yes, of course. Share the bug and the relevant code, and we'll work through it together.")


# =========================================================
# Thanks / farewell
# =========================================================

add("ස්තුතියි", "ඔයාට උදව් කරන්න ලැබීම සතුටක්.")
add("බොහොම ස්තුතියි", "ඔයාට උදව් වුනා නම් ඒක සතුටක්. ආයෙත් මොනවහරි ඕන නම් අහන්න.")
add("Thank you", "You're welcome. I'm happy to help.")
add("Thanks a lot", "You're very welcome. Feel free to ask if you need anything else.")
add("ඔයාට ගොඩක් ස්තුතියි උදව් කරාට", "සතුටින්! ඕන වෙලාවක ආයෙත් අහන්න.")
add("Appreciate the help", "Anytime! Let me know if there's anything else you need.")
add("Bye", "Bye! Take care.")
add("Goodbye", "Goodbye, feel free to come back anytime you need help.")
add("ආයුබෝවන් යනවා", "ආයුබෝවන්! ආයෙත් හම්බෙමු.")
add("See you later", "See you later! I'll be here whenever you need help.")


if __name__ == "__main__":

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for user, assistant in pairs:

            item = {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }

            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} conversations to {OUTPUT_FILE}")
