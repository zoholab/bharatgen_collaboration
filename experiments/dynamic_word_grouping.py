import sys

# ------------- CONFIGURABLE PATTERNS -----------------
word_group_count=0
total_tokens=0
# 1. Specific multiword expressions to be grouped as a unit
RULE1_PHRASES = [
    ["दे", "दिया"],
    ["मिला", "दें"],
    ["मुकर", "जाएं"],
    ["सम्मिलित", "करना"],
    ["हाल", "ही", "में"],
    ["कर", "दी"],
    ["दे", "दो"],
    ["दे", "दें"],
    ["दी", "थी"]
]

# 2. Words that must attach to the word on their left with "_"
ATTACH_TO_LEFT = {
    "से", "में", "का", "के", "की", "को", "पर", "ने", "भी", "ही",
    "द्वारा", "वाला", "वाली", "वाले", "जी", "सी", "तरह", "किया", "किए"
    "दी_थी",  # in case this already appears as such in the data
}

# 3. Multiwords that should first be grouped as a unit,
#    and then attach to the word on their left.
RULE3_MULTIWORDS = [
    ["रहे", "हैं"],
    ["रहा", "है"],
    ["रही", "है"],
    ["सकता", "है"],
    ["सकती", "है"],
    ["सकते", "हैं"],
    ["हो", "गयी"],
    ["हो", "गया"],
    ["के", "लिए"],
    ["के", "बाद"],
    ["के", "साथ"],
    ["के", "बीच"],
    ["के", "दौरान"],
    ["के", "खिलाफ़"],
    ["के", "प्रति"],
    ["की", "ओर"],
    ["बारे", "में"],
    ["के", "मुताबिक़"],
    ["के", "मुताबिक"],
    ["के", "तहत"],
    ["ओर", "से"],
    ["के", "कारण"],
    ["ने", "भी"],
    ["में", "ही"],
    ["ही", "में"],
]

# From RULE3_MULTIWORDS we’ll derive the grouped forms that should attach to the left.
ATTACH_MULTI_TO_LEFT = {"##".join(p) for p in RULE3_MULTIWORDS}

# 5. Rule 5: endings + auxiliaries
# (implemented as patterns, not data lists)
A_ENDING = "ा"
EE_ENDING = "ी"
E_ENDING = "े"

AUX_AFTER_EE = {"गई", "जाएगी", "जायेगी"}
AUX_AFTER_A = {"गया", "जाएगा", "जायेगा"}

# ------------- CORE FUNCTIONS -----------------

def is_number_token(w: str) -> bool:
    """
    Returns True if the token looks like a number.
    Handles:
    - Western digits with commas/decimals: 80,000  3.14
    - Hindi digits: ५, १०, १०००, etc. 
    """
    global word_group_count
    # Remove common number formatting chars
    cleaned = w.replace(",", "").replace(".", "")
    return cleaned.isdigit()


def apply_phrase_grouping(tokens):
    """
    - Group RULE1_PHRASES and RULE3_MULTIWORDS into single tokens joined by "##".
    - Apply Rule 5 character-ending + auxiliary patterns.
    """
    global word_group_count
    # Combine all phrase patterns and sort by length (longest first)
    all_phrases = RULE1_PHRASES + RULE3_MULTIWORDS
    all_phrases = sorted(all_phrases, key=len, reverse=True)

    out = []
    i = 0
    n = len(tokens)

    while i < n:
        # Try phrase-based patterns first (Rule 1 + Rule 3)
        matched = False
        for phrase in all_phrases:
            L = len(phrase)
            if i + L <= n and tokens[i:i + L] == phrase:
                out.append("##".join(phrase))
                word_group_count+=1
                i += L
                matched = True
                break
        if matched:
            continue

        # Rule 5 patterns (ending char + auxiliary)
        w = tokens[i]

        # word ending with "ी" followed by certain auxiliaries
        if w.endswith(EE_ENDING) and i + 1 < n and tokens[i + 1] in AUX_AFTER_EE:
            out.append(w + "##" + tokens[i + 1])
            word_group_count+=1
            i += 2
            continue

        # word ending with "ा" followed by certain auxiliaries
        if w.endswith(A_ENDING) and i + 1 < n and tokens[i + 1] in AUX_AFTER_A:
            out.append(w + "##" + tokens[i + 1])
            word_group_count+=1
            i += 2
            continue

        # word ending with "ा"/"ी" + "चाहिए"
        if i + 1 < n and tokens[i + 1] == "चाहिए":
            if w.endswith(A_ENDING) or w.endswith(EE_ENDING):
                out.append(w + "_चाहिए")
                word_group_count+=1
                i += 2
                continue

        # word ending with "े" + "लगता/लगती है"
        if w.endswith(E_ENDING) and i + 2 < n and tokens[i + 1] in {"लगता", "लगती"} and tokens[i + 2] == "है":
            out.append(w + "##" + tokens[i + 1] + "##" + tokens[i + 2])
            word_group_count+=1
            i += 3
            continue

        # default: no special grouping
        out.append(w)
        i += 1

    return out


def apply_right_attachment(tokens):
    """
    Rule 4
    - "नहीं" and numbers should be grouped with the word to their right.
    Examples: "नहीं करना" -> "नहीं_करना", "10 ग्राम" -> "10_ग्राम"
    """
    out = []
    i = 0
    n = len(tokens)
    global word_group_count
    while i < n:
        w = tokens[i]
        # If "नहीं" or numeric token and there is a right neighbor
        if (w == "नहीं" or is_number_token(w)) and i + 1 < n:
            out.append(w + "##" + tokens[i + 1])
            word_group_count+=1
            i += 2
        else:
            out.append(w)
            i += 1

    return out


def apply_left_attachment(tokens):
    """
    Rule 2 + left-attachment part of Rule 3
    - Tokens in ATTACH_TO_LEFT attach to the previous token.
    - Tokens in ATTACH_MULTI_TO_LEFT (e.g., "के_लिए") also attach to the previous token.
    """
    global word_group_count
    out = []
    for tok in tokens:
        if tok in ATTACH_TO_LEFT or tok in ATTACH_MULTI_TO_LEFT:
            if out:
                out[-1] = out[-1] + "##" + tok
                word_group_count+=1
            else:
                # No left word; just keep as is
                out.append(tok)
        else:
            out.append(tok)
    return out


def group_sentence(sentence):
    global total_tokens
    #Apply all grouping rules to one sentence (string).
    
    tokens = sentence.strip().split()
    if not tokens:
        return ""

    tokens = apply_phrase_grouping(tokens)
    tokens = apply_right_attachment(tokens)
    tokens = apply_left_attachment(tokens)
    length=len(tokens)
    total_tokens+=length
    return " ".join(tokens)


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.strip() == "":
                fout.write("\n")
                continue
            grouped = group_sentence(line)
            fout.write(grouped + "\n")


# ------------- MAIN -----------------


def main():
    if len(sys.argv) != 3:
        print("Usage: python hi_word_grouping.py <input.txt> <output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_file(input_path, output_path)
    print(word_group_count)
    print(total_tokens)

if __name__ == "__main__":
    main()