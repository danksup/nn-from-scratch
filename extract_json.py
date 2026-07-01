
import json

def clean_answer(a):
    if isinstance(a, str) and a.startswith("https://en.wikipedia.org/wiki/"):
        a = a.rsplit("/", 1)[-1].replace("_", " ")
    return str(a)

def extract_json_to_txt(filepath, query, answer):
    with open(filepath, "r") as f:
        data = json.load(f)

    txt = ""

    for item in data:
        queries = item[query]
        answers = item[answer]

        if not isinstance(queries, list):
            queries = [queries]

        if isinstance(answers, list):
            answers = ", ".join(clean_answer(a) for a in answers)
        else:
            answers = clean_answer(answers)

        for q in queries:
            txt += f"Question: {q}\n"
            txt += f"Answer: {answers}\n\n"

    return txt

def save_txt(filename, txt):
    with open(f"data/{filename}.txt", "w") as f:
        f.write(txt)
    print(f"saved {filename}.txt")

save_txt("qoac",extract_json_to_txt("data/json/qoac.json", "questions", "answers"))