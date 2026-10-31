"""Stage 5 [SMALL LLM — Minimal Cost]: Constrained response generation with static fallback."""

from __future__ import annotations

from typing import Dict, Optional

from src.config import DEFAULT_CONFIG
from src.schemas import CleanedPayload, DraftOutput, RuleEvaluationResult
from src.slm_client import LocalSLMClient, SLMClientError

STATIC_FALLBACK_TEMPLATES: Dict[str, str] = {
    "TPL_BILLING_01": (
        "Thank you for contacting Billing Support regarding your account. "
        "Your ticket ({ticket_id}) has been logged with {priority} priority. "
        "A specialist will respond within our {sla_hours}-hour SLA window. "
        "You can review billing FAQs at https://billing.example.com/help."
    ),
    "TPL_TECH_01": (
        "We have received your technical support request under ticket {ticket_id}. "
        "Our engineering support team is reviewing the issue ({priority} priority) "
        "and will provide a technical update within {sla_hours} hours. "
        "Check system status anytime at https://status.example.com."
    ),
    "TPL_FEAT_01": (
        "Thank you for sharing your product feedback! We have logged your feature request "
        "under reference {ticket_id}. Our product team reviews all submissions within "
        "{sla_hours} hours. Visit https://docs.example.com for current release notes."
    ),
}

DEFAULT_FALLBACK_BODY = (
    "Thank you for reaching out. Your request has been logged under ticket {ticket_id} "
    "and our team will respond within {sla_hours} hours."
)


def _build_ticket_id(data: CleanedPayload) -> str:
    """Create a deterministic short ticket reference from the event UUID."""
    short_hex = str(data.event_id).split("-")[0].upper()
    return f"TKT-{short_hex}"


def render_static_fallback_draft(
    data: CleanedPayload,
    rules: RuleEvaluationResult,
) -> DraftOutput:
    """Populate a deterministic static template via Python `.format()` when SLM drafting fails."""
    ticket_id = _build_ticket_id(data)
    template = STATIC_FALLBACK_TEMPLATES.get(rules.response_template_id, DEFAULT_FALLBACK_BODY)
    body = template.format(
        ticket_id=ticket_id,
        sla_hours=rules.sla_hours,
        priority=rules.priority.value,
        queue=rules.assigned_queue,
    )[:600]

    base_subject = data.clean_subject if data.clean_subject else "Support Request"
    subject_line = f"Re: {base_subject} [{ticket_id}]"[:120]

    return DraftOutput(
        subject_line=subject_line,
        body_text=body,
        referenced_ticket_id=ticket_id,
    )


def draft_response(
    data: CleanedPayload,
    rules: RuleEvaluationResult,
    slm_client: LocalSLMClient,
    model: str = DEFAULT_CONFIG.draft_model,
) -> Optional[DraftOutput]:
    """Generate a constrained reply using a small local LLM, or return None if drafting is disabled.

    - Returns `None` immediately (0 tokens) if `not rules.should_draft_reply`.
    - Passes only sanitized metadata, `allowed_tone`, `response_template_id`, and `sla_hours`.
    - Enforces `DraftOutput.model_json_schema()` with `num_predict=200` and `temperature=0.0`.
    - Falls back to a deterministic `.format()` template if SLM call or validation fails.
    """
    if not rules.should_draft_reply:
        return None

    ticket_id = _build_ticket_id(data)

    system_prompt = (
        "You are a constrained automated support responder. "
        "Generate a concise reply strictly adhering to the provided metadata and tone. "
        "Do NOT include any HTML tags, external non-example.com URLs, personal data, "
        "credit card numbers, or refund/legal guarantees. "
        "Output ONLY valid JSON matching the DraftOutput schema."
    )

    user_prompt = (
        f"Reference Ticket ID: {ticket_id}\n"
        f"Template ID: {rules.response_template_id}\n"
        f"Allowed Tone: {rules.allowed_tone}\n"
        f"Priority: {rules.priority.value}\n"
        f"Target Queue: {rules.assigned_queue}\n"
        f"SLA Hours: {rules.sla_hours}\n"
        f"Sanitized Subject Summary: {data.clean_subject[:100]}\n"
    )

    try:
        return slm_client.generate_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=DraftOutput,
            num_predict=DEFAULT_CONFIG.draft_max_tokens,
            temperature=DEFAULT_CONFIG.temperature,
        )
    except (SLMClientError, Exception):
        return render_static_fallback_draft(data, rules)

