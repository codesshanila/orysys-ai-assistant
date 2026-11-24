"""Guardrails: deterministic checks before and after the LLM.

Layered defence (no single layer is trusted on its own):

  INPUT     validate_user_input   - size/charset limits, instruction-override, exfiltration
                                    and tool-abuse patterns -> block or flag.
  RETRIEVAL sanitize_retrieved    - retrieved text is DATA. Lines that look like
                                    instructions are neutralised and the chunk is wrapped
                                    in <document> tags the prompts tell the model to treat
                                    as untrusted.
  TOOLS     (rbac.require_tool + pydantic schemas in tools/registry.py)
  OUTPUT    validate_answer       - every [citation] must exist in the evidence the user
                                    was allowed to see; no secrets/PII; brand rules
                                    (no financial advice or promises, no disparaging
                                    language, stays in the Commercial Bank persona).

Pattern checks are intentionally simple and explainable. They run in microseconds, are
unit-tested, and every decision is emitted to the activity panel and LangSmith.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 4000

INJECTION_PATTERNS: dict[str, str] = {
    "instruction_override": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|system|earlier)\b.{0,20}\b(instructions?|rules?|prompts?|guidelines?)",
    "role_hijack": r"\b(you are now|act as|pretend to be|from now on you)\b.{0,40}\b(admin|administrator|developer|dan|jailbreak|unrestricted|root)\b",
    "mode_switch": r"\b(admin|developer|god|debug|jailbreak)\s+mode\b",
    "prompt_extraction": r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}\b(system prompt|your instructions|hidden prompt|initial prompt|your rules)\b",
}
EXFILTRATION_PATTERNS: dict[str, str] = {
    "send_externally": r"\b(send|email|post|upload|forward|exfiltrate|transmit)\b.{0,60}(https?://|@[\w-]+\.[a-z]{2,}|webhook|pastebin)",
    "bulk_dump": r"\b(dump|export|list|give me|reveal)\b.{0,20}\b(all|every|entire)\b.{0,30}\b(restricted|passwords?|secrets?|credentials?|api keys?)\b",
    "encoded_payload": r"(base64|rot13|hex)\s*(encode|decode|encoded)",
}
TOOL_ABUSE_PATTERNS: dict[str, str] = {
    "code_injection": r"(__import__|\bos\.system|subprocess|\beval\(|\bexec\(|\bopen\(|rm\s+-rf|;\s*drop\s+table)",
    "privilege_escalation": r"\b(grant|give|make)\b.{0,20}\b(me|my account)\b.{0,20}\b(admin|administrator|superuser|full access)\b",
    "chaos_request": r"\b(disable|turn off|bypass)\b.{0,30}\b(guardrails?|safety|authori[sz]ation|rate limit)",
}

SECRET_PATTERNS = {
    "api_key": r"\b(sk-[A-Za-z0-9_\-]{16,}|pcsk_[A-Za-z0-9_]{16,}|lsv2_[A-Za-z0-9_]{16,})\b",
    "password_assignment": r"\bpassword\s*[:=]\s*\S+",
    "card_number": r"\b(?:\d[ -]?){13,16}\b",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}

# Brand guardrail: the assistant speaks for Commercial Bank.
BRAND_PATTERNS = {
    "financial_advice": r"\b(you should (buy|sell|invest)|guaranteed returns?|i recommend investing)\b",
    "unauthorised_commitment": r"\b(we guarantee|the bank guarantees|we promise|i promise)\b",
    "disparaging": r"\b(stupid|idiot|useless|incompetent|garbage)\b",
}

INSTRUCTION_LINE = re.compile(
    "|".join(f"(?:{p})" for p in list(INJECTION_PATTERNS.values()) + list(EXFILTRATION_PATTERNS.values())),
    re.I,
)
CITATION_RE = re.compile(r"\[([A-Z]{2,5}-\d{3,4}(?:-\d{3})?#\d{2})\]")


@dataclass
class GuardResult:
    allowed: bool
    findings: list[str] = field(default_factory=list)
    sanitized: str = ""
    reason: str = ""

    def as_event(self) -> dict:
        return {"allowed": self.allowed, "findings": self.findings, "reason": self.reason}


def _scan(text: str, patterns: dict[str, str]) -> list[str]:
    return [name for name, pat in patterns.items() if re.search(pat, text, re.I | re.S)]


def validate_user_input(text: str) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, ["empty_input"], reason="The question is empty.")
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, ["too_long"], reason=f"Questions are limited to {MAX_INPUT_CHARS} characters.")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        return GuardResult(False, ["control_characters"], reason="The question contains invalid characters.")

    findings = (
        [f"injection:{n}" for n in _scan(text, INJECTION_PATTERNS)]
        + [f"exfiltration:{n}" for n in _scan(text, EXFILTRATION_PATTERNS)]
        + [f"tool_abuse:{n}" for n in _scan(text, TOOL_ABUSE_PATTERNS)]
    )
    if findings:
        return GuardResult(
            False, findings, sanitized=text,
            reason="This request looks like an attempt to override the assistant's rules or move data outside approved channels.",
        )
    return GuardResult(True, [], sanitized=text.strip())


def sanitize_retrieved(text: str) -> tuple[str, list[str]]:
    """Neutralise instruction-like lines inside retrieved documents."""
    findings: list[str] = []
    clean_lines = []
    for line in text.splitlines() or [text]:
        if INSTRUCTION_LINE.search(line):
            findings.append("retrieved_instruction_neutralised")
            clean_lines.append("[removed: instruction-like text found in source document]")
        else:
            clean_lines.append(line)
    # Within single-line chunk texts, neutralise sentence by sentence.
    out = "\n".join(clean_lines)
    if not findings and INSTRUCTION_LINE.search(out):
        parts = re.split(r"(?<=[.!?])\s+", out)
        out = " ".join("[removed: instruction-like text]" if INSTRUCTION_LINE.search(p) else p for p in parts)
        findings.append("retrieved_instruction_neutralised")
    return out, findings


def wrap_evidence(chunk_id: str, title: str, text: str) -> str:
    safe = text.replace("<document", "&lt;document").replace("</document", "&lt;/document")
    return f'<document id="{chunk_id}" title="{title}">\n{safe}\n</document>'


def validate_tool_args_text(value: str, max_len: int = 500) -> str:
    """Shared check for free-text tool parameters (queries, names)."""
    if len(value) > max_len:
        raise ValueError(f"parameter longer than {max_len} characters")
    if _scan(value, TOOL_ABUSE_PATTERNS) or _scan(value, EXFILTRATION_PATTERNS):
        raise ValueError("parameter contains disallowed content")
    return value.strip()


def redact_secrets(text: str) -> tuple[str, list[str]]:
    findings = []
    for name, pat in SECRET_PATTERNS.items():
        if re.search(pat, text, re.I):
            findings.append(f"redacted:{name}")
            text = re.sub(pat, "[REDACTED]", text, flags=re.I)
    return text, findings


@dataclass
class AnswerValidation:
    valid: bool
    citations: list[str]
    invalid_citations: list[str]
    issues: list[str]
    text: str

    def as_event(self) -> dict:
        return {"valid": self.valid, "citations": self.citations,
                "invalid_citations": self.invalid_citations, "issues": self.issues}


def validate_answer(answer: str, evidence_ids: set[str], require_citations: bool) -> AnswerValidation:
    issues: list[str] = []
    text, redactions = redact_secrets(answer)
    issues.extend(redactions)

    cited = list(dict.fromkeys(CITATION_RE.findall(text)))
    invalid = [c for c in cited if c not in evidence_ids]
    if invalid:
        issues.append("hallucinated_citations")
    if require_citations and evidence_ids and not cited:
        issues.append("missing_citations")
    issues.extend(f"brand:{n}" for n in _scan(text, BRAND_PATTERNS))
    if len(text.strip()) < 2:
        issues.append("empty_answer")

    blocking = [i for i in issues if not i.startswith("redacted:")]
    return AnswerValidation(not blocking, cited, invalid, issues, text)


def strip_invalid_citations(text: str, invalid: list[str]) -> str:
    for c in invalid:
        text = text.replace(f"[{c}]", "")
    return text
