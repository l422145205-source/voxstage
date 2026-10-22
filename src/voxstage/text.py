"""Content fidelity metrics; ASR never receives these reference strings."""
import unicodedata


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def compare(reference, hypothesis):
    a,b = normalize(reference),normalize(hypothesis)
    if not a:
        raise ValueError("empty normalized reference")
    previous = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(x!=y)))
        previous=row
    errors = previous[-1]
    # CER can exceed 1 when the recognizer inserts many characters.
    return {"cer": errors/len(a), "edit_distance": errors, "reference_length": len(a),
            "normalized_reference": a, "normalized_hypothesis": b}
