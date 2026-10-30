export type Billing = 'free' | 'paid' | 'unknown';
export type InputStatus = 'yes' | 'no' | 'unlikely' | 'unknown';
export type InputSupport = Record<string, { status: InputStatus; source: string; evidence?: string; checked_at?: string }>;
export type ProbeInfo = Record<string, { status: 'yes' | 'no' | 'unknown'; evidence: string; checked_at: string }>;

export type Model = {
  model_id: string; display_name: string; capabilities: string[]; context_window?: number | null;
  metadata_source?: string; catalog_source?: 'live' | 'fallback'; billing?: Billing; billing_source?: string; custom?: boolean;
  input_support?: InputSupport; probes?: ProbeInfo;
  image_output?: { status: 'verified' | 'failed' | 'declared' | 'untested' | 'unsupported_adapter'; detail?: string };
};
export type Provider = { provider_id: string; display_name: string; models: Model[]; catalog_source?: 'live' | 'fallback'; catalog_message?: string; last_checked?: string | null };

export type RoutePreview = {
  provider_id: string | null; provider_name?: string; model_id?: string; reason: string;
  alternatives?: Array<{ provider_id: string; provider_name: string; model_id: string; billing: Billing; input_support?: Record<string, string> }>;
};

export type GeneratedImage = { mime_type: string; data: string };
export type StoredFile = { id: number; name: string; kind: string; mime: string; size: number; stored: boolean };
export type LocalFile = { name: string; mime_type: string; size: number; previewUrl?: string };
export type Attached = LocalFile & { data: string };
export type ChatMessage = {
  id: number | null; role: 'user' | 'assistant'; content: string;
  provider?: string | null; model?: string | null; note?: string | null;
  images?: GeneratedImage[] | null; files?: StoredFile[]; localFiles?: LocalFile[]; pending?: boolean;
};
export type ChatSummary = { id: string; title: string; updated_at: string; message_count: number };
export type ChatDetail = { id: string; title: string; context: string; messages: ChatMessage[] };

export type OutputMode = 'auto' | 'text' | 'image';
export type Prefs = { providerId: string; modelId: string; autoPick: boolean; preferFree: boolean; safeFallback: boolean; output: OutputMode };

export type ProbeKind = 'text' | 'code' | 'image' | 'pdf' | 'audio' | 'video' | 'image_output';
export type ProbeResult = { kind: ProbeKind; status: 'pass' | 'fail' | 'inconclusive' | 'skipped'; detail: string; ms: number | null; recorded: boolean };

export type DashboardData = {
  generated_at: string;
  summary: { requests: number; tokens: number; estimated_cost_usd: number; today_requests: number; today_tokens: number; today_estimated_cost_usd: number };
  calibration: { total_requests: number; provider_reported_requests: number; note: string };
  providers: Array<{ provider_id: string; display_name: string; key_configured: boolean; requests: number; tokens: number; estimated_cost_usd: number; reported_usage_requests: number; quota_status: string; models: Model[] }>;
  recent_events: Array<{ ts: string; provider_id: string; model_id: string; task: string; total_tokens: number | null; usage_source: string; estimated_cost_usd: number | null }>;
};
