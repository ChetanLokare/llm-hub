import { FlaskConical, Play, Square } from 'lucide-react';
import { useRef, useState } from 'react';
import { json, request } from '../api';
import type { Model, ProbeKind, ProbeResult, Provider } from '../types';
import { readAsBase64 } from '../util';

const KINDS: Array<{ key: ProbeKind; label: string; hint: string; defaultOn: boolean }> = [
  { key: 'text', label: 'Text', hint: 'Does the model answer at all?', defaultOn: true },
  { key: 'code', label: 'Code', hint: 'Can it write a small function?', defaultOn: true },
  { key: 'image', label: 'Image in', hint: 'Sends a coloured square; the model must name the colour', defaultOn: true },
  { key: 'pdf', label: 'PDF in', hint: 'Sends a PDF with a secret code; the model must repeat it', defaultOn: true },
  { key: 'audio', label: 'Audio in', hint: 'Sends a 1-second tone (acceptance is checked, what it hears is not)', defaultOn: false },
  { key: 'video', label: 'Video in', hint: 'Needs ffmpeg on this computer, or upload a short sample below', defaultOn: false },
  { key: 'image_output', label: 'Image out', hint: 'Asks for a picture (Gemini only in this app)', defaultOn: false },
];
const MARK: Record<string, string> = { yes: '✓', no: '✗', unlikely: '!', unknown: '?', pass: '✓', fail: '✗', inconclusive: '…', skipped: '–' };
const LIMITS = [3, 5, 10, 25, 1000];

type RunState = { running?: boolean; results?: ProbeResult[] };

const knowledge = (m: Model): Array<{ label: string; status: string; tip: string }> => {
  const inp = m.input_support || {}; const pr = m.probes || {};
  const tip = (e?: string, at?: string, src?: string) => [e, at ? `tested ${at.slice(0, 10)}` : '', src].filter(Boolean).join(' · ');
  return [
    { label: 'text', status: pr.text?.status || 'unknown', tip: tip(pr.text?.evidence, pr.text?.checked_at) },
    { label: 'code', status: pr.code?.status || 'unknown', tip: tip(pr.code?.evidence, pr.code?.checked_at) },
    ...(['image', 'pdf', 'audio', 'video'] as const).map(k => ({ label: `${k} in`, status: inp[k]?.status || 'unknown', tip: tip(inp[k]?.evidence, inp[k]?.checked_at, inp[k]?.source) })),
    { label: 'image out', status: pr.image_output?.status || 'unknown', tip: tip(pr.image_output?.evidence, pr.image_output?.checked_at) },
  ];
};

export function ModelLab({ providers, keyed, onChanged }: { providers: Provider[]; keyed: Set<string>; onChanged: () => void }) {
  const [kinds, setKinds] = useState<ProbeKind[]>(KINDS.filter(k => k.defaultOn).map(k => k.key));
  const [limit, setLimit] = useState(5);
  const [onlyUntested, setOnlyUntested] = useState(true);
  const [runs, setRuns] = useState<Record<string, RunState>>({});
  const [batch, setBatch] = useState<string | null>(null);
  const [sample, setSample] = useState<{ name: string; mime_type: string; data: string } | null>(null);
  const [showAll, setShowAll] = useState<Record<string, boolean>>({});
  const [error, setError] = useState('');
  const stop = useRef(false);

  const toggle = (k: ProbeKind) => setKinds(prev => prev.includes(k) ? prev.filter(x => x !== k) : [...prev, k]);
  const confirmRun = () => window.confirm('Model Lab sends small real requests to the provider (a few tokens each; a picture request for Image out). Continue?');

  const runModel = async (pid: string, mid: string): Promise<boolean> => {
    const key = `${pid}/${mid}`;
    setRuns(r => ({ ...r, [key]: { running: true, results: r[key]?.results } }));
    try {
      const data = await request<{ results: ProbeResult[] }>('/probe', json('POST', { provider_id: pid, model_id: mid, kinds, video_sample: kinds.includes('video') ? sample : null }));
      setRuns(r => ({ ...r, [key]: { results: data.results } }));
      return true;
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'The test could not run';
      setRuns(r => ({ ...r, [key]: { results: [{ kind: 'text', status: 'inconclusive', detail: msg, ms: null, recorded: false }] } }));
      setError(msg);
      return !/key|budget|limit/i.test(msg);
    }
  };

  const single = async (pid: string, mid: string) => {
    if (!kinds.length || !confirmRun()) return;
    setError(''); await runModel(pid, mid); onChanged();
  };

  const runProvider = async (p: Provider) => {
    if (!kinds.length || !confirmRun()) return;
    setError(''); stop.current = false; setBatch(p.provider_id);
    const todo = p.models.filter(m => !onlyUntested || (m.probes?.text?.status ?? 'unknown') === 'unknown').slice(0, limit);
    for (const m of todo) {
      if (stop.current) break;
      if (!(await runModel(p.provider_id, m.model_id))) break;   // stop on a rejected key / spending limit
    }
    setBatch(null); onChanged();
  };

  const pickSample = async (f?: File) => {
    if (!f) return;
    if (f.size > 20 * 1048576) { setError('Sample video must be under 20 MB.'); return; }
    setSample({ name: f.name, mime_type: f.type || 'video/mp4', data: await readAsBase64(f) });
  };

  const usable = providers.filter(p => keyed.has(p.provider_id));
  return (
    <section className="dash-panel lab-panel">
      <div className="panel-heading"><div><h2><FlaskConical size={17} /> Model Lab</h2>
        <p>Send different small requests to each connected API and model to find out what really works: text, code, image / PDF / audio / video in, images out. Results feed Auto-pick and Safe fallback. Only providers with a saved key can be tested.</p></div></div>
      <div className="lab-controls">
        <div className="lab-kinds">{KINDS.map(k => <label key={k.key} title={k.hint}><input type="checkbox" checked={kinds.includes(k.key)} onChange={() => toggle(k.key)} /> {k.label}</label>)}</div>
        <div className="lab-options">
          <label>Models per run <select value={limit} onChange={e => setLimit(Number(e.target.value))}>{LIMITS.map(n => <option key={n} value={n}>{n === 1000 ? 'All' : n}</option>)}</select></label>
          <label><input type="checkbox" checked={onlyUntested} onChange={e => setOnlyUntested(e.target.checked)} /> Skip models already tested</label>
          {kinds.includes('video') && <label className="lab-sample">Video sample <input type="file" accept="video/*" onChange={e => void pickSample(e.target.files?.[0])} />{sample && <small>{sample.name}</small>}</label>}
        </div>
      </div>
      {error && <div className="notice-box"><span>{error}</span></div>}
      {!usable.length && <p className="empty-note">No provider with a saved API key yet. Add one in Settings, then come back to test its models.</p>}
      {usable.map(p => {
        const shown = showAll[p.provider_id] ? p.models : p.models.slice(0, 12);
        return (
          <div className="lab-provider" key={p.provider_id}>
            <div className="lab-provider-head">
              <div><strong>{p.display_name}</strong><small>{p.models.length} models</small></div>
              {batch === p.provider_id
                ? <button className="settings-button" onClick={() => { stop.current = true; }}><Square size={13} /> Stop</button>
                : <button className="settings-button" disabled={batch !== null || !kinds.length} onClick={() => void runProvider(p)}><Play size={13} /> Test {limit === 1000 ? 'all' : `up to ${limit}`} models</button>}
            </div>
            {shown.map(m => {
              const run = runs[`${p.provider_id}/${m.model_id}`];
              return (
                <div className="lab-model" key={m.model_id}>
                  <div className="lab-model-name"><span title={m.model_id}>{m.display_name}</span>
                    <button className="settings-button" disabled={Boolean(run?.running) || batch !== null || !kinds.length} onClick={() => void single(p.provider_id, m.model_id)}>{run?.running ? 'Testing…' : 'Test'}</button></div>
                  <div className="lab-badges">{knowledge(m).map(b => <span key={b.label} className={`input-badge ${b.status}`} title={b.tip || 'Not tested yet'}>{b.label} {MARK[b.status] || '?'}</span>)}</div>
                  {run?.results && <div className="lab-results">{run.results.map(r => <span key={r.kind} className={`lab-result ${r.status}`} title={r.detail}>{r.kind.replace('_', ' ')} {MARK[r.status]}{r.ms != null ? ` ${r.ms} ms` : ''} <em>{r.detail}</em></span>)}</div>}
                </div>
              );
            })}
            {p.models.length > 12 && <button className="link-button" onClick={() => setShowAll(s => ({ ...s, [p.provider_id]: !s[p.provider_id] }))}>{showAll[p.provider_id] ? 'Show fewer' : `Show all ${p.models.length} models`}</button>}
          </div>
        );
      })}
      <p className="lab-note">✓ confirmed · ✗ not supported · ! failed recently (tried last, never excluded) · ? untested. A rate limit or server error is shown as “…” and is never held against a model.</p>
    </section>
  );
}
