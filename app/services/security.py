import re
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class InputSanitizer:
    INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directions)", re.IGNORECASE),
        re.compile(r"(reveal|disclose|show|expose|output|print|display|leak)\s+(your|the|its)\s*(system|internal|secret|hidden|original)\s*(prompt|instructions|directives|prompt|message)", re.IGNORECASE),
        re.compile(r"(system|prompt|instruction|role|persona)\s*(:|is|was|as)\s*(.*)", re.IGNORECASE),
        re.compile(r"you\s+are\s+(now|henceforth)\s+(\w+)", re.IGNORECASE),
        re.compile(r"jailbreak|security\s*bypass|bypass\s*restrictions|dan\s*(mode)?", re.IGNORECASE),
        re.compile(r"role\s*play|roleplay", re.IGNORECASE),
        re.compile(r"no\s*(restrictions|limits|boundaries|filtering)", re.IGNORECASE),
        re.compile(r"new\s*rule", re.IGNORECASE),
        re.compile(r"you're\s+an?\s+ai\s+(assistant|chatbot)", re.IGNORECASE),
        re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    ]

    CLEAN_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"---.*?---", re.DOTALL), ""),
        (re.compile(r"\{\{.*?\}\}", re.DOTALL), ""),
        (re.compile(r"<\|.*?\|>", re.DOTALL), ""),
    ]

    def check(self, text: str) -> tuple[bool, str | None]:
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return False, f"Prompt injection detected: '{match.group()[:60]}'"
        return True, None

    def clean(self, text: str) -> str:
        result = text
        for pattern, replacement in self.CLEAN_PATTERNS:
            result = pattern.sub(replacement, result)
        return result.strip()


class PIIDetector:
    PII_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "email"),
        (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "phone"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
        (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "credit_card"),
    ]

    def detect(self, text: str) -> list[str]:
        found = []
        for pattern, label in self.PII_PATTERNS:
            if pattern.search(text):
                found.append(label)
        return found

    def mask(self, text: str) -> str:
        result = text
        for pattern, label in self.PII_PATTERNS:
            result = pattern.sub(f"[{label.upper()} REDACTED]", result)
        return result


class OutputValidator:
    BLOCKED_PATTERNS: list[re.Pattern] = [
        re.compile(r"hack|exploit|vulnerability|malware", re.IGNORECASE),
    ]

    def __init__(self):
        self._pii = PIIDetector()

    def validate(self, text: str) -> tuple[str, list[str]]:
        warnings: list[str] = []

        masked = self._pii.mask(text)
        if masked != text:
            warnings.append("PII detected in output and redacted")

        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(masked):
                masked = "This response was blocked by content safety filters."
                warnings.append("Harmful content detected in output")
                break

        return masked, warnings


class SecurityPipeline:
    def __init__(self):
        self._sanitizer = InputSanitizer()
        self._pii = PIIDetector()
        self._output_validator = OutputValidator()

    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        notes: list[str] = []

        cleaned = self._sanitizer.clean(text)
        if cleaned != text:
            notes.append("Input cleaned (delimiters/templates removed)")

        is_safe, reason = self._sanitizer.check(cleaned)
        if not is_safe:
            notes.append(f"Injection blocked: {reason}")
            return False, cleaned, notes

        pii_found = self._pii.detect(cleaned)
        if pii_found:
            notes.append(f"PII detected in input: {', '.join(pii_found)}")
            cleaned = self._pii.mask(cleaned)

        return True, cleaned, notes

    def check_output(self, text: str) -> tuple[str, list[str]]:
        return self._output_validator.validate(text)
