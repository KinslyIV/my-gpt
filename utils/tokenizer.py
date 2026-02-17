
def char_encode(text: str) -> list[int]:
    return [ord(c) for c in text]

def char_decode(tokens: list[int]) -> str:
    if not tokens:
        return ""
    
    text = ""
    for token in tokens:
        text += chr(token)

    return text