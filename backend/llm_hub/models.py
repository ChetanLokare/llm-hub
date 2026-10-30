from dataclasses import dataclass, field

@dataclass
class ModelInfo:
    provider: str
    model_id: str
    display_name: str
    context_window: int | None = None
    capabilities: list[str] = field(default_factory=lambda: ["text"])
    metadata_source: str = "starter catalog"
    billing: str = "unknown"  # "free" | "paid" | "unknown"; filled in by model_registry
    # What the PROVIDER says this model accepts as input, e.g. {"image": True, "audio": False}.
    # None = the provider's model list says nothing. Never filled in from the model's name.
    modality_support: dict | None = None

@dataclass
class UsageInfo:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    source: str = "estimated"
    estimated_input_tokens: int | None = None
