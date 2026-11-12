"""Text and value rules shared by the training environments.

    parse_response(text)      one model step -> dict(kind in {call, ask, tell, none}, calls, text, think, reason)
    canon(args, keep)         {argument path: canonical scalar} of a call's arguments
    same_value(a, b, arg)     equivalence of a generated argument value and a recorded one
    is_ask(text)              the visible text asks the user for information
    is_placeholder(value)     a placeholder instead of a real value ("<user_id>", "your_token", "TBD", ...)

A step is parsed from the text after its closing </think>: <tool_call> blocks with one JSON object each are calls;
otherwise the visible text is an ask or a plain answer ("tell").
"""
import json
import re


# ---------------------------------------------------------------- canonical scalars


QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        " ": " ", "–": "-", "—": "-", "−": "-"})


WS = re.compile(r"\s+")


def norm(s):
    return WS.sub(" ", str(s).translate(QUOTES).lower()).strip()


def words_of_name(name):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if w]


NUM_STR = re.compile(r"^[-+]?\$?\d[\d,]*(\.\d+)?%?$")


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def fmt_num(x):
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.10g}"


def canon_scalar(v):
    """Canonical string of a scalar for equality checks."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return fmt_num(float(v))
    s = norm(v).strip(" .,;:!?'\"`")
    if NUM_STR.match(s):
        try:
            return fmt_num(float(s.replace(",", "").replace("$", "").replace("%", "")))
        except ValueError:
            pass
    return s


def leaves_of(obj, path=""):
    if isinstance(obj, dict):
        if not obj:
            yield path, obj
        for k, v in obj.items():
            yield from leaves_of(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        if not obj:
            yield path, obj
        for i, v in enumerate(obj):
            yield from leaves_of(v, f"{path}[{i}]")
    else:
        yield path, obj


def top_arg(path):
    return re.split(r"[.\[]", path, 1)[0]


# ---------------------------------------------------------------- questions, offers, placeholders


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


ASK_REQUEST = re.compile(r"""
  \b(?:could|can|would|will)\s+you\s+(?:please\s+|kindly\s+)?
      (?:provide|specify|share|tell|confirm|clarify|give|send|supply|indicate|let\s+me\s+know|enter|name|identify|advise)\b
 |\bplease\s+(?:provide|specify|share|tell\s+me|confirm|clarify|give\s+me|send\s+me|supply|indicate|enter|
      let\s+me\s+know\s+(?:the|which|what|your|how|whether|who|where|when|if))\b
 |\b(?:i|we)\s*(?:would\s+|will\s+|'ll\s+|’ll\s+|still\s+)?(?:need|require)\s+(?:\*\*)?
      (?:the|your|to\s+know|more|you\s+to|some|additional|further|clarification|confirmation|details|information|specific)\b
 |\b(?:let\s+me\s+know|tell\s+me)\s+(?:the|which|what|your|how\s+many|how\s+much|whether|who|where|when)\b
""", re.I | re.X)


OFFER = re.compile(r"""
  \b(?:would|do)\s+you\s+(?:like|want|prefer|wish)\b|\b(?:shall|should)\s+i\b|\banything\s+else\b
 |\bhow\s+(?:else\s+)?(?:can|may)\s+i\s+(?:help|assist)\b|\bis\s+there\s+(?:anything|something)\b
 |\b(?:further|more|other|additional)\s+(?:help|assistance)\b
""", re.I | re.X)


TOOLISH = re.compile(r"\b(?:tool|function|api|capabilit)", re.I)


def sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def is_ask(text):
    """Explicit request for information addressed to the user."""
    for s in sentences(text):
        if ASK_REQUEST.search(s) and not (re.search(r"\bneed|\brequire", s, re.I) and TOOLISH.search(s)
                                          and not re.search(r"\byou\b", s, re.I)):
            return True
        if s.rstrip("*_ )\"'`").endswith("?") and not OFFER.search(s):
            return True
    return False


PLACEHOLDER_EXACT = re.compile(r"""^(?:unknown|tbd|tba|todo|placeholder|n/?a|none|null|nil|undefined|\?+|\.{3,}|…
  |to\s+be\s+(?:determined|confirmed|provided|decided)|not\s+(?:provided|specified|available|known|given)
  |unspecified|example|dummy|\{\}|\[\])$""", re.I | re.X)


PLACEHOLDER_ANY = re.compile(r"""x{3,}|^[#*_\-]{3,}$|^<[^<>]{1,60}>$|^\{\{?[^{}]{1,60}\}?\}$|^\$\{?[A-Za-z_]\w*\}?$
  |\[\s*insert|\binsert[_\s]|^your_\w+$|^your\s+\w+(?:\s+\w+)?$|<\s*(?:your|insert|result|output|id|booking|file|name|value)
  |\b(?:result|output|value|id|response)[_\s](?:of|from)[_\s](?:the[_\s])?(?:previous|above|prior|last|step|first|earlier)
  |\bprevious[_\s](?:result|output|step|call)|\bfrom[_\s]previous""", re.I | re.X)


def is_placeholder_basic(v):
    if isinstance(v, dict):
        return len(v) == 0
    if isinstance(v, str):
        s = v.strip()
        return s == "" or bool(PLACEHOLDER_EXACT.match(s)) or bool(PLACEHOLDER_ANY.search(s))
    return False


# ---------------------------------------------------------------- one model step


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_calls(text):
    calls = []
    for m in TOOL_CALL_RE.finditer(text or ""):
        try:
            c = json.loads(m.group(1))
        except Exception:
            continue
        if isinstance(c, dict) and c.get("name"):
            calls.append({"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments") or {}}})
    return calls


def parse_response(text):
    """-> dict(kind in {call, ask, tell, none}, calls, text, think, reason)."""
    if "</think>" not in (text or ""):
        return {"kind": "none", "reason": "unclosed_think", "calls": [], "text": "", "think": text or ""}
    think, rest = text.split("</think>", 1)
    calls = parse_calls(rest)
    if "<tool_call>" in rest and not calls:
        return {"kind": "none", "reason": "bad_json", "calls": [], "text": rest, "think": think}
    visible = TOOL_CALL_RE.sub("", rest).replace("<|im_end|>", "").strip()
    if calls:
        return {"kind": "call", "calls": calls, "text": visible, "think": think, "reason": None}
    if not visible:
        return {"kind": "none", "reason": "empty", "calls": [], "text": "", "think": think}
    if is_ask(visible):
        return {"kind": "ask", "calls": [], "text": visible, "think": think, "reason": None}
    return {"kind": "tell", "calls": [], "text": visible, "think": think, "reason": None}


# ---------------------------------------------------------------- value equivalence


_MON = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}


_RX_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[t ].*)?$")


_RX_NUM = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$")


_RX_MD = re.compile(r"^([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?$")


_RX_DM = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?(?:,?\s+(\d{4}))?$")


def date_key(s):
    """(year or None, month, day) for common date spellings, else None. Only whole-string dates are parsed."""
    s = str(s).strip().lower()
    y = mo = d = None
    m = _RX_ISO.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = _RX_NUM.match(s)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y = y + 2000 if y < 100 else y
            mo, d = (b, a) if a > 12 else (a, b)
        else:
            m = _RX_MD.match(s)
            if m and m.group(1)[:3] in _MON:
                mo, d, y = _MON[m.group(1)[:3]], int(m.group(2)), int(m.group(3)) if m.group(3) else None
            else:
                m = _RX_DM.match(s)
                if m and m.group(2)[:3] in _MON:
                    d, mo, y = int(m.group(1)), _MON[m.group(2)[:3]], int(m.group(3)) if m.group(3) else None
    if mo is None or not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return (y, mo, d)


def _same_value_basic(model_v, target_v):
    """Canonical equality (case, whitespace, number format already normalised by canon_scalar), plus equal dates
    written differently ("October 15th" = "2023-10-15"; the year is compared only when both give one)."""
    if model_v == target_v:
        return True
    a, b = date_key(model_v), date_key(target_v)
    return a is not None and b is not None and a[1:] == b[1:] and (a[0] is None or b[0] is None or a[0] == b[0])


def canon(args, keep=None):
    out = {}
    for path, v in leaves_of(args if isinstance(args, dict) else {}):
        if not path:
            continue
        if keep is not None and top_arg(path) not in keep:
            continue
        out[path] = canon_scalar(v)
    return out


def _fmt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _tool_names(tools):
    names = set()
    for t in tools or []:
        fn = (t.get("function") or t) if isinstance(t, dict) else {}
        if fn.get("name"):
            names.add(fn["name"])
    return names


READ_VERBS = {"retrieve", "retrieves", "fetch", "fetches", "get", "gets", "list", "lists", "search", "searches",
              "return", "returns", "look", "lookup", "check", "checks", "find", "finds", "query", "queries", "provide",
              "provides", "obtain", "obtains", "show", "shows", "decode", "decodes", "calculate", "calculates",
              "compute", "computes", "convert", "converts", "estimate", "estimates", "predict", "predicts", "analyze",
              "analyzes", "translate", "translates", "view", "read", "reads", "extract", "extracts", "validate",
              "validates", "verify", "verifies", "detect", "detects"}


FREE_ARGS = {"query", "q", "keyword", "keywords", "search", "searchterm", "search_term", "search_query", "term", "text",
             "prompt", "question", "topic"}


STOP = set("the a an of for to in on at by with and or is are be this that which from as it its given specified "
           "specific provide provided using use api tool data information details based".split())


PLACEHOLDER_EXTRA = re.compile(r"^<[^<>]{1,60}>$|^\[[^\[\]]{1,60}\]$|^(your|my)[_ ]\w+|_here$|^(provided|sample|example)_"
                               r"|[_ ](from|of)[_ ](the[_ ])?(previous|prior|search|result|response|above|step|last|first|earlier)", re.I)


def stem(w):
    for suf in ("ing", "ers", "er", "es", "s", "e"):
        if len(w) > 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def toks(s):
    return {stem(w) for w in words_of_name(str(s)) if w not in STOP and len(w) >= 2}


def norm_key(k):
    k = re.sub(r"[^a-z0-9]", "", str(k).lower())
    return k[2:] if k.startswith("is") and len(k) > 3 else k


NAME_FILLER = {"get", "api", "the", "a", "of", "by", "v1", "v2", "v3", "endpoint"}


def norm_name(n):
    return " ".join(stem(w) for w in words_of_name(n) if w not in NAME_FILLER)  # order kept: cmyk-to-hex != hex_to_cmyk


def sep_norm(s):
    return re.sub(r"[\s_\-]+", " ", str(s)).strip()


_ACTIVE = {"ctx": "", "enum_args": set()}  # set by Episode.step() before any matching (single-threaded use)


_IDENT = re.compile(r"[0-9@]|://|\.(com|org|net|io)\b|[/\\]")


def _descriptive(s):
    s = str(s).strip()
    if not s or _IDENT.search(s):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
    return bool(words) and not any(w[0].isupper() for w in words[1:])


def _span_rule(mv, tv, arg):
    if not (isinstance(mv, str) and isinstance(tv, str)) or norm_key(arg) in _ACTIVE["enum_args"]:
        return False
    if not (_descriptive(mv) and _descriptive(tv)):
        return False
    a, b = toks(mv), toks(tv)
    if not a or not b or max(len(a), len(b)) < 3 or not (a <= b or b <= a):
        return False
    ctx = _ACTIVE["ctx"]
    return sep_norm(mv).lower() in ctx


def same_value(mv, tv, arg=""):
    if _same_value_basic(mv, tv):
        return True
    if isinstance(mv, str) and isinstance(tv, str):
        if sep_norm(mv) == sep_norm(tv):
            return True
        a, b = toks(mv), toks(tv)
        if len(b) >= 2 and a == b:  # word order / punctuation / plural: 'Paris 10th arrondissement' = '10th arrondissement, Paris'
            return True
        if norm_key(arg) in {norm_key(a) for a in FREE_ARGS}:
            a, b = toks(mv), toks(tv)
            if a and b and len(a & b) / len(a | b) >= 0.5:
                return True
        if _span_rule(mv, tv, arg):
            return True
    return False


def is_placeholder(v):
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False  # empty strings are legitimate 'no value' for optional args (e.g. cursor '')
    return is_placeholder_basic(s) or bool(PLACEHOLDER_EXTRA.search(s))
