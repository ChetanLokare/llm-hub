import { Image as ImageIcon, CircleAlert, Code2, Download, FileText, Paperclip, Pencil, RefreshCw, Send, Sparkles, Type, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { json, newChatId, request } from '../api';
import type { Attached, ChatDetail, ChatMessage, OutputMode, Prefs, Provider, RoutePreview } from '../types';
import { copyText, filesToAttached, fmtSize, kindOf } from '../util';
import { MessageView } from './MessageView';
import { ModelPicker } from './ModelPicker';

type Props = {
  providers: Provider[]; prefs: Prefs; setPrefs: (p: Partial<Prefs>) => void;
  chatId: string | null; onChatId: (id: string | null) => void; onChatsChanged: () => void;
  reloadCatalog: () => void; catalogVersion: number; connected: boolean;
};

const STARTERS = [
  { icon: Code2, title: 'Write or fix code', text: 'Write a Python function that checks whether a number is prime, with tests.' },
  { icon: ImageIcon, title: 'Create an image', text: 'Generate an image of a cozy reading nook with warm evening light.' },
  { icon: FileText, title: 'Read my files', text: 'Summarise the attached files and list the key points.' },
  { icon: Pencil, title: 'Improve my writing', text: 'Help me improve this writing: ' },
];
const OUTPUTS: Array<{ key: OutputMode; label: string; icon: typeof Type; hint: string }> = [
  { key: 'auto', label: 'Auto', icon: Sparkles, hint: 'Make an image when you ask for one, otherwise answer in text' },
  { key: 'text', label: 'Text', icon: Type, hint: 'Always answer in text' },
  { key: 'image', label: 'Image', icon: ImageIcon, hint: 'Always try to create an image' },
];

export function Chat({ providers, prefs, setPrefs, chatId, onChatId, onChatsChanged, reloadCatalog, catalogVersion, connected }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [title, setTitle] = useState('New chat');
  const [context, setContext] = useState('');
  const [contextOpen, setContextOpen] = useState(false);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attached[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<RoutePreview | null>(null);
  const loadedId = useRef<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const scroller = useRef<HTMLElement>(null);

  // Returns the stored messages, or null when the chat does not exist on the server (yet).
  const loadChat = useCallback(async (id: string): Promise<ChatMessage[] | null> => {
    try {
      const d = await request<ChatDetail>(`/chats/${id}`);
      loadedId.current = id; setMessages(d.messages); setTitle(d.title); setContext(d.context || '');
      return d.messages;
    } catch { return null; }
  }, []);

  // switching chats from the sidebar
  useEffect(() => {
    if (chatId === loadedId.current) return;
    setError(''); setInput(''); setAttachments([]);
    if (!chatId) { loadedId.current = null; setMessages([]); setTitle('New chat'); setContext(''); return; }
    void loadChat(chatId);
  }, [chatId, loadChat]);

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' }); }, [messages, busy]);
  useEffect(() => { const el = area.current; if (el) { el.style.height = 'auto'; el.style.height = `${Math.min(220, el.scrollHeight)}px`; } }, [input]);

  // what Auto-pick would try first, given the kinds of file attached
  const modalities = [...new Set(attachments.map(kindOf).filter(Boolean))].join(',');
  useEffect(() => {
    if (!prefs.autoPick) { setPreview(null); return; }
    const ctrl = new AbortController();
    request<RoutePreview>(`/route-preview?prefer_free=${prefs.preferFree}&modalities=${encodeURIComponent(modalities)}`, { signal: ctrl.signal })
      .then(setPreview).catch(() => { /* preview is optional */ });
    return () => ctrl.abort();
  }, [prefs.autoPick, prefs.preferFree, modalities, catalogVersion]);

  const addFiles = async (list: File[]) => {
    if (!list.length) return;
    const { added, errors } = await filesToAttached(list, attachments);
    if (errors.length) setError(errors.join(' '));
    if (added.length) setAttachments(prev => [...prev, ...added]);
  };
  const removeAttachment = (i: number) => setAttachments(prev => { const gone = prev[i]; if (gone?.previewUrl) URL.revokeObjectURL(gone.previewUrl); return prev.filter((_, j) => j !== i); });

  const send = async (opts: { regenerate?: boolean; editId?: number; text?: string; keepFiles?: boolean } = {}) => {
    if (busy) return;
    const regenerate = Boolean(opts.regenerate);
    const editing = opts.editId !== undefined;
    const text = (opts.text ?? input).trim();
    const files = regenerate || editing ? [] : attachments;
    if (!regenerate && !editing && !text && !files.length) return;
    const id = chatId ?? newChatId();
    const knownIds = new Set(messages.map(m => m.id));
    setError(''); setBusy(true);
    loadedId.current = id;
    if (!chatId) onChatId(id);

    const userMsg: ChatMessage = { id: null, role: 'user', content: text || 'Please look at the attached file(s).', pending: true,
      localFiles: files.map(f => ({ name: f.name, mime_type: f.mime_type, size: f.size, previewUrl: f.previewUrl })) };
    if (regenerate) setMessages(prev => { let end = prev.length; while (end > 0 && prev[end - 1].role === 'assistant') end--; return prev.slice(0, end); });
    else if (editing) setMessages(prev => { const at = prev.findIndex(m => m.id === opts.editId); return [...prev.slice(0, at < 0 ? prev.length : at), { ...userMsg, files: prev[at]?.files }]; });
    else { setMessages(prev => [...prev, userMsg]); setInput(''); setAttachments([]); }

    try {
      const data = await request<any>('/chat', json('POST', {
        chat_id: id, message: regenerate ? '' : text, attachments: files.map(({ name, mime_type, data }) => ({ name, mime_type, data })),
        edit_message_id: editing ? opts.editId : null, regenerate, keep_files: opts.keepFiles ?? true, output: prefs.output,
        provider_id: prefs.providerId, model_id: prefs.modelId, auto: prefs.autoPick, prefer_free: prefs.preferFree,
        safe_fallback: prefs.safeFallback, context: '',
      }));
      await loadChat(data.chat_id);
      onChatsChanged();
      if (data.images?.length) reloadCatalog();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
      // The server stores your message before asking a model, so after a failure it is usually already saved and
      // "Retry" re-asks it. If it was not saved (e.g. an invalid attachment) put the draft back instead.
      const server = await loadChat(id);
      const stored = Boolean(server?.some(m => m.role === 'user' && !knownIds.has(m.id)));
      if (!server) { setMessages(prev => prev.filter(m => !m.pending)); if (!chatId) { loadedId.current = null; onChatId(null); } }
      if (!regenerate && !editing && !stored) { setMessages(prev => prev.filter(m => !m.pending)); setInput(prev => prev || text); setAttachments(prev => (prev.length ? prev : files)); }
      onChatsChanged();
    } finally { setBusy(false); }
  };

  const newContextChat = async (value: string) => {
    const id = chatId ?? newChatId();
    if (!chatId) { await request('/chats', json('POST', { id, context: value })); loadedId.current = id; onChatId(id); }
    else await request(`/chats/${id}`, json('PUT', { context: value }));
    setContext(value); setContextOpen(false);
  };
  const rename = async () => {
    if (!chatId) return;
    const t = window.prompt('Rename this chat', title);
    if (t && t.trim()) { await request(`/chats/${chatId}`, json('PUT', { title: t.trim() })); setTitle(t.trim()); onChatsChanged(); }
  };
  const exportChat = () => {
    const body = messages.map(m => `## ${m.role === 'user' ? 'You' : `Assistant${m.model ? ` (${m.model})` : ''}`}\n\n${m.content}`).join('\n\n---\n\n');
    const url = URL.createObjectURL(new Blob([body], { type: 'text/markdown;charset=utf-8' }));
    const a = document.createElement('a'); a.href = url; a.download = (title || 'llm-hub-chat').replace(/[^a-z0-9-_ ]/gi, '').trim().replace(/\s+/g, '-') + '.md'; a.click(); URL.revokeObjectURL(url);
  };
  const editMessage = (id: number, text: string, keepFiles: boolean) => void send({ editId: id, text, keepFiles });
  const lastIdx = messages.length - 1;
  const lastIsUser = messages.length > 0 && messages[lastIdx].role === 'user';

  return (
    <>
      <header className="chat-head">
        <div className="chat-title"><strong title={title}>{title}</strong>{chatId && <button className="icon-btn" onClick={() => void rename()} title="Rename chat"><Pencil size={13} /></button>}</div>
        <div className="chat-head-actions">
          <ModelPicker providers={providers} prefs={prefs} setPrefs={setPrefs} preview={preview} hasAttachments={attachments.length > 0} />
          <button className="settings-button" onClick={() => setContextOpen(true)} title="Notes kept on this device and sent to every model in this chat"><Pencil size={14} /> Context{context ? ' ●' : ''}</button>
          <button className="settings-button" onClick={exportChat} disabled={!messages.length}><Download size={14} /> Export</button>
        </div>
      </header>

      <section className="chat-scroll" id="chat-scroll" ref={scroller}>
        {!messages.length ? (
          <div className="welcome">
            <div className="welcome-icon"><Sparkles size={27} /></div>
            <div className="eyebrow">YOUR AI WORKSPACE</div>
            <h1>What are we working on<br /><span>today?</span></h1>
            <p>One chat for everything: text, code, images, documents, zip files, audio and video. Switch models any time — your history, files and notes stay on this device and go with you.</p>
            <div className="prompt-grid">{STARTERS.map(c => { const Icon = c.icon; return (
              <button key={c.title} className="prompt-card" onClick={() => { setInput(c.text); area.current?.focus(); }}>
                <span className="prompt-icon"><Icon size={17} /></span><span><strong>{c.title}</strong><small>{c.text.length > 54 ? c.text.slice(0, 54) + '…' : c.text}</small></span>
              </button>); })}</div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((m, i) => <MessageView key={m.id ?? `p${i}`} msg={m} chatId={chatId} isLast={i === lastIdx} busy={busy} onEdit={editMessage} onRegenerate={() => void send({ regenerate: true })} />)}
            {busy && <div className="message-row assistant"><div className="message-avatar assistant-avatar"><Sparkles size={16} /></div><div className="message-content"><div className="message-meta"><strong>LLM Hub</strong><span>Working…</span></div><div className="bubble assistant-bubble typing"><i /><i /><i /></div></div></div>}
          </div>
        )}
        {!connected && <div className="error-banner">Backend is not connected. Start the FastAPI server using the README instructions.</div>}
        {error && <div className="error-banner"><span>{error}</span>{lastIsUser && !busy && <button className="settings-button" onClick={() => void send({ regenerate: true })}><RefreshCw size={13} /> Retry</button>}<button className="icon-btn" aria-label="Dismiss" onClick={() => setError('')}><X size={14} /></button></div>}
      </section>

      <footer className="composer-area">
        <div className={`composer${dragOver ? ' drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); void addFiles(Array.from(e.dataTransfer.files as FileList)); }}>
          {attachments.length > 0 && <div className="attach-chips">{attachments.map((a, k) => (
            <span key={k} className={`attach-chip${a.previewUrl ? ' with-thumb' : ''}`} title={kindOf(a) ? `${kindOf(a)} input: Auto-pick looks for a model that accepts it` : 'Read as text where possible; zip files are opened'}>
              {a.previewUrl ? <img src={a.previewUrl} alt="" /> : <Paperclip size={11} />}{a.name} · {fmtSize(a.size)}
              <button aria-label={`Remove ${a.name}`} onClick={() => removeAttachment(k)}><X size={11} /></button>
            </span>))}</div>}
          <textarea ref={area} value={input} rows={1} onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); } }}
            onPaste={e => { const files = Array.from(e.clipboardData.files as FileList); if (files.length) { e.preventDefault(); void addFiles(files); } }}
            placeholder="Message LLM Hub…  (paste or drop images, video, audio, PDFs, documents, code, zip)" />
          <div className="composer-bottom">
            <div className="composer-left">
              <input ref={fileInput} type="file" multiple hidden onChange={e => { if (e.target.files) void addFiles(Array.from(e.target.files as FileList)); e.target.value = ''; }} />
              <button className="settings-button attach-btn" title="Attach images, video, audio, PDFs, documents or zip files" onClick={() => fileInput.current?.click()}><Paperclip size={14} /> Attach</button>
              <div className="seg" role="group" aria-label="Output type">{OUTPUTS.map(o => { const Icon = o.icon; return (
                <button key={o.key} className={prefs.output === o.key ? 'active' : ''} title={o.hint} onClick={() => setPrefs({ output: o.key })}><Icon size={12} /> {o.label}</button>); })}</div>
            </div>
            <button className="send-button" disabled={(!input.trim() && !attachments.length) || busy} onClick={() => void send()} title="Send (Enter)"><Send size={16} /></button>
          </div>
        </div>
        <div className="composer-note"><CircleAlert size={12} /> AI can make mistakes. Verify important information. API usage may incur provider charges. Replies arrive all at once (no streaming yet).</div>
      </footer>

      {contextOpen && <ContextModal value={context} onClose={() => setContextOpen(false)} onSave={v => void newContextChat(v)} />}
    </>
  );
}

function ContextModal({ value, onClose, onSave }: { value: string; onClose: () => void; onSave: (v: string) => void }) {
  const [text, setText] = useState(value);
  return (
    <div className="modal-backdrop" onClick={onClose}><section className="settings-modal context-modal" onClick={e => e.stopPropagation()}>
      <div className="modal-head"><div><span className="eyebrow">CONVERSATION MEMORY</span><h2>Chat context</h2></div><button className="icon-btn" onClick={onClose}><X size={18} /></button></div>
      <p className="modal-desc">Facts and instructions you want every model to remember in this chat. Saved on this device and sent to whichever model answers, including after a switch or an automatic fallback. Your messages, attached files and recent images are remembered automatically.</p>
      <label>Instructions and facts to preserve
        <textarea className="context-editor" rows={8} value={text} onChange={e => setText(e.target.value)} placeholder="Example: I am building a FastAPI + React app. Prefer Windows PowerShell commands and explain changes step by step." /></label>
      <div className="modal-actions"><button className="settings-button" onClick={() => setText('')}>Clear</button><button className="primary-wide" onClick={() => onSave(text)}>Save context</button></div>
    </section></div>
  );
}
