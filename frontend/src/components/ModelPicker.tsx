import { ChevronDown, Sparkles } from 'lucide-react';
import { useState } from 'react';
import type { Billing, Prefs, Provider, RoutePreview } from '../types';

const GROUP: Record<Billing, string> = { free: 'Free models', paid: 'Paid / billed models', unknown: 'Not classified' };

export function ModelPicker({ providers, prefs, setPrefs, preview, hasAttachments }: {
  providers: Provider[]; prefs: Prefs; setPrefs: (p: Partial<Prefs>) => void; preview: RoutePreview | null; hasAttachments: boolean;
}) {
  const [open, setOpen] = useState(false);
  const provider = providers.find(p => p.provider_id === prefs.providerId);
  const label = prefs.autoPick
    ? `Auto${preview?.model_id ? ` · ${preview.model_id}` : ''}`
    : `${provider?.display_name ?? 'Choose model'}${prefs.modelId ? ` · ${prefs.modelId}` : ''}`;
  return (
    <div className="model-picker">
      <button className="picker-button" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        {prefs.autoPick && <Sparkles size={13} />}<span>{label}</span><ChevronDown size={13} />
      </button>
      {open && <>
        <div className="picker-overlay" onClick={() => setOpen(false)} />
        <div className="picker-panel" role="dialog" aria-label="Model settings">
          <label className="picker-row"><input type="checkbox" checked={prefs.autoPick} onChange={e => setPrefs({ autoPick: e.target.checked })} />
            <span><strong>Auto-pick model</strong><small>Chooses among every model your connected providers offer, using what they report and what Model Lab and past requests proved. No vendor or model names are favoured.</small></span></label>
          <label className="picker-row"><input type="checkbox" checked={prefs.preferFree} onChange={e => setPrefs({ preferFree: e.target.checked })} />
            <span><strong>Prefer free models</strong><small>Free first, then unclassified, then paid. Nothing is excluded.</small></span></label>
          <label className="picker-row"><input type="checkbox" checked={prefs.safeFallback} onChange={e => setPrefs({ safeFallback: e.target.checked })} />
            <span><strong>Safe fallback</strong><small>If a model fails for any reason, try the next capable one until an answer arrives. A retry after a timeout can occasionally bill twice.</small></span></label>
          {prefs.autoPick && preview && <div className="picker-preview"><strong>Next pick{hasAttachments ? ' (for your attachments)' : ''}</strong><span>{preview.reason}</span>
            {(preview.alternatives || []).slice(1, 4).map(a => <small key={`${a.provider_id}/${a.model_id}`}>then {a.provider_name} · {a.model_id}{a.billing !== 'unknown' ? ` (${a.billing})` : ''}</small>)}</div>}
          <div className={`picker-manual ${prefs.autoPick ? 'disabled' : ''}`}>
            <label>Provider<select disabled={prefs.autoPick} value={prefs.providerId} onChange={e => {
              const p = providers.find(x => x.provider_id === e.target.value); setPrefs({ providerId: e.target.value, modelId: p?.models[0]?.model_id || '' });
            }}>{providers.map(p => <option key={p.provider_id} value={p.provider_id}>{p.display_name}</option>)}</select></label>
            <label>Model<select disabled={prefs.autoPick} value={prefs.modelId} onChange={e => setPrefs({ modelId: e.target.value })}>
              {(['free', 'paid', 'unknown'] as Billing[]).map(b => {
                const list = (provider?.models ?? []).filter(m => (m.billing || 'unknown') === b);
                return list.length ? <optgroup key={b} label={GROUP[b]}>{list.map(m => <option key={m.model_id} value={m.model_id}>{m.display_name}</option>)}</optgroup> : null;
              })}
            </select></label>
          </div>
        </div>
      </>}
    </div>
  );
}
