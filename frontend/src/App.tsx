import { ChevronDown, LayoutDashboard, Menu, MessageSquarePlus, PanelLeftClose, Pencil, Plus, Search, Settings2, Sparkles, Sun, Trash2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { API, json, request } from './api';
import { Chat } from './components/Chat';
import { Dashboard } from './Dashboard';
import type { Billing, ChatSummary, DashboardData, Prefs, Provider } from './types';

const PREFS_KEY = 'llmhub-prefs-v2';
const LEGACY_CHATS_KEY = 'llmhub-chats-v1';
const DEFAULT_PREFS: Prefs = { providerId: '', modelId: '', autoPick: true, preferFree: false, safeFallback: true, output: 'auto' };
const loadPrefs = (): Prefs => { try { return { ...DEFAULT_PREFS, ...JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') }; } catch { return DEFAULT_PREFS; } };
const legacyChats = (): Array<{ title: string; context?: string; messages: Array<{ role: string; content: string; provider?: string; model?: string }> }> => {
  try { return JSON.parse(localStorage.getItem(LEGACY_CHATS_KEY) || '[]'); } catch { return []; }
};

function App() {
  const [theme, setTheme] = useState<'dark' | 'light' | 'system'>(() => (localStorage.getItem('llmhub-theme') as 'dark' | 'light' | 'system') || 'dark');
  const [providers, setProviders] = useState<Provider[]>([]);
  const [catalog, setCatalog] = useState<Provider[]>([]);
  const [catalogRefreshing, setCatalogRefreshing] = useState(false);
  const [catalogVersion, setCatalogVersion] = useState(0);
  const [prefs, setPrefsState] = useState<Prefs>(loadPrefs);
  const [connected, setConnected] = useState(true);
  const [error, setError] = useState('');
  const [testingModel, setTestingModel] = useState<string | null>(null);
  const [addModelOpen, setAddModelOpen] = useState(false);
  const [addModelError, setAddModelError] = useState('');
  const [newModel, setNewModel] = useState<{ provider_id: string; model_id: string; display_name: string; billing: Billing; context_window: string; capabilities: string[] }>({ provider_id: '', model_id: '', display_name: '', billing: 'unknown', context_window: '', capabilities: ['text'] });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [keyProvider, setKeyProvider] = useState('gemini');
  const [apiKey, setApiKey] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [view, setView] = useState<'chat' | 'dashboard'>('chat');
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [detection, setDetection] = useState<any>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [search, setSearch] = useState('');
  const [chatId, setChatId] = useState<string | null>(null);
  const [legacyCount, setLegacyCount] = useState(() => legacyChats().length);

  const setPrefs = useCallback((p: Partial<Prefs>) => setPrefsState(prev => ({ ...prev, ...p })), []);
  const providerId = prefs.providerId, modelId = prefs.modelId;
  const setProviderId = (v: string) => setPrefs({ providerId: v });
  const setModelId = (v: string) => setPrefs({ modelId: v });

  useEffect(() => { localStorage.setItem('llmhub-theme', theme); }, [theme]);
  useEffect(() => { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); }, [prefs]);

  const reloadCatalog = useCallback(async () => {
    try {
      const r = await fetch(`${API}/providers`); if (!r.ok) return;
      const list: Provider[] = await r.json(); setCatalog(list); setProviders(list); setConnected(true); setCatalogVersion(v => v + 1);
      setPrefsState(prev => {
        const p = list.find(x => x.provider_id === prev.providerId) ?? list[0];
        if (!p) return prev;
        const m = p.models.some(x => x.model_id === prev.modelId) ? prev.modelId : p.models[0]?.model_id || '';
        return p.provider_id === prev.providerId && m === prev.modelId ? prev : { ...prev, providerId: p.provider_id, modelId: m };
      });
    } catch { setConnected(false); }
  }, []);
  useEffect(() => { void reloadCatalog(); }, [reloadCatalog]);

  const refreshChats = useCallback(async () => {
    try { setChats(await request<ChatSummary[]>(`/chats?q=${encodeURIComponent(search)}`)); } catch { /* backend offline: banner shows in chat */ }
  }, [search]);
  useEffect(() => { const t = window.setTimeout(() => void refreshChats(), 200); return () => window.clearTimeout(t); }, [refreshChats]);

  const importLegacy = async () => {
    for (const c of legacyChats()) {
      try { await request('/chats/import', json('POST', { title: c.title, context: c.context || '', messages: c.messages })); } catch { /* skip a bad chat */ }
    }
    localStorage.removeItem(LEGACY_CHATS_KEY); setLegacyCount(0); await refreshChats();
  };
  const newChat = () => { setChatId(null); setView('chat'); };
  const openChat = (id: string) => { setChatId(id); setView('chat'); };
  const deleteChat = async (c: ChatSummary) => {
    if (!window.confirm(`Delete "${c.title}"? Its messages and stored files are removed from this device.`)) return;
    await request(`/chats/${c.id}`, { method: 'DELETE' });
    if (c.id === chatId) setChatId(null);
    await refreshChats();
  };
  const renameChat = async (c: ChatSummary) => {
    const t = window.prompt('Rename this chat', c.title);
    if (t && t.trim()) { await request(`/chats/${c.id}`, json('PUT', { title: t.trim() })); await refreshChats(); }
  };

  const loadDashboard = async () => { setDashboardLoading(true); try { const r=await fetch(`${API}/dashboard`); if(!r.ok) throw new Error('Dashboard API failed'); setDashboardData(await r.json()); } catch(e) { setError(e instanceof Error?e.message:'Could not load dashboard'); } finally { setDashboardLoading(false); } };
    const refreshCatalog = async () => { setCatalogRefreshing(true); try { const response=await fetch(`${API}/catalog/refresh`,{method:'POST'}); const result=await response.json(); if(!response.ok) throw new Error(result.detail||'Could not refresh model catalog'); const refreshed:Provider[]=result.providers||[]; setCatalog(refreshed); setProviders(refreshed); if(!refreshed.some(p=>p.provider_id===providerId)){setProviderId(refreshed[0]?.provider_id||'');setModelId(refreshed[0]?.models[0]?.model_id||'');} else if(!refreshed.find(p=>p.provider_id===providerId)?.models.some(m=>m.model_id===modelId)){setModelId(refreshed.find(p=>p.provider_id===providerId)?.models[0]?.model_id||'');} } catch(e) { setError(e instanceof Error?e.message:'Could not refresh model catalog'); } finally {setCatalogRefreshing(false);} };
    const setBilling = async (pid:string,mid:string,billing:'free'|'paid'|'auto') => { try { const r=await fetch(`${API}/models/billing`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider_id:pid,model_id:mid,billing})}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Could not update billing'); await reloadCatalog(); } catch(e) { setError(e instanceof Error?e.message:'Could not update billing'); } };
    const deleteModel = async (pid:string,mid:string) => { try { const path=mid.split('/').map(encodeURIComponent).join('/'); const r=await fetch(`${API}/models/custom/${encodeURIComponent(pid)}/${path}`,{method:'DELETE'}); if(!r.ok) throw new Error('Could not remove the model'); await reloadCatalog(); if(pid===providerId&&mid===modelId){ const first=providers.find(p=>p.provider_id===pid)?.models.find(m=>m.model_id!==mid); setModelId(first?.model_id||''); } } catch(e) { setError(e instanceof Error?e.message:'Could not remove the model'); } };
    const testImage = async (pid:string,mid:string) => { if(!window.confirm('This sends one small image request to the provider and may use a few tokens. Continue?')) return; setTestingModel(`${pid}/${mid}`); try { const r=await fetch(`${API}/models/test-image`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider_id:pid,model_id:mid})}); const d=await r.json(); if(!r.ok) throw new Error(typeof d.detail==='string'?d.detail:'The test could not be run'); window.alert(d.message+(d.detail?`\n\n${d.detail}`:'')); await reloadCatalog(); } catch(e) { setError(e instanceof Error?e.message:'The test could not be run'); } finally { setTestingModel(null); } };
    const openAddModel = () => { setAddModelError(''); setNewModel(m=>({...m,provider_id:m.provider_id||providers[0]?.provider_id||''})); setAddModelOpen(true); };
    const submitNewModel = async () => { setAddModelError(''); try { const body={provider_id:newModel.provider_id,model_id:newModel.model_id.trim(),display_name:newModel.display_name.trim(),capabilities:newModel.capabilities,billing:newModel.billing,context_window:newModel.context_window?parseInt(newModel.context_window,10):null}; const r=await fetch(`${API}/models/custom`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const d=await r.json(); if(!r.ok) throw new Error(typeof d.detail==='string'?d.detail:'Could not add the model. Check the fields.'); await reloadCatalog(); setProviderId(body.provider_id); setModelId(body.model_id); setAddModelOpen(false); setNewModel(m=>({...m,model_id:'',display_name:'',context_window:''})); } catch(e) { setAddModelError(e instanceof Error?e.message:'Could not add the model'); } };
    const detectApis = async () => { try { const r=await fetch(`${API}/detect`); if(!r.ok) throw new Error('API detection failed'); setDetection(await r.json()); } catch(e) { setError(e instanceof Error?e.message:'Detection failed'); } };
    const saveKey=async () => {
      setError('');
      try {
        const res=await fetch(`${API}/keys`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider_id:keyProvider,api_key:apiKey})});
        const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Could not save key');
        setApiKey(''); setSettingsOpen(false);
      } catch(e) {setError(e instanceof Error?e.message:'Could not save key');}
    };
  useEffect(() => { if (view === 'dashboard') void loadDashboard(); }, [view]);
  const providersForKey = useMemo(() => providers.map(p => ({ id: p.provider_id, name: p.display_name })), [providers]);
  const visibleChats = chats.filter(c => c.message_count > 0);

  return <div className={`app theme-${theme}`}>
    {sidebarOpen && <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Sparkles size={20} /></div><div><strong>LLM Hub</strong><span>One chat, every model</span></div><button className="icon-btn sidebar-collapse" onClick={() => setSidebarOpen(false)} title="Collapse sidebar"><PanelLeftClose size={17} /></button></div>
      <button className="new-chat" onClick={newChat}><MessageSquarePlus size={17} /> New chat</button>
      <button className={`task-item dashboard-nav ${view === 'dashboard' ? 'active' : ''}`} onClick={() => setView('dashboard')}><LayoutDashboard size={17} /><span>Dashboard &amp; Model Lab</span>{view === 'dashboard' && <span className="active-dot" />}</button>
      <div className="chat-search"><Search size={13} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search chats" aria-label="Search chats" />{search && <button className="icon-btn" aria-label="Clear search" onClick={() => setSearch('')}><X size={12} /></button>}</div>
      <div className="side-label history-label">CHATS <button className="tiny-side-action" onClick={newChat} title="New chat"><Plus size={13} /></button></div>
      {legacyCount > 0 && <button className="import-legacy" onClick={() => void importLegacy()}>Import {legacyCount} chat{legacyCount > 1 ? 's' : ''} saved by the earlier version</button>}
      <div className="history-list">
        {visibleChats.map(c => <div className={`history-item ${c.id === chatId && view === 'chat' ? 'selected' : ''}`} key={c.id}>
          <button className="history-open" onClick={() => openChat(c.id)} title={c.title}><MessageSquarePlus size={14} /><span>{c.title}</span></button>
          <span className="history-actions"><button className="icon-btn" aria-label="Rename" onClick={() => void renameChat(c)}><Pencil size={12} /></button><button className="icon-btn" aria-label="Delete" onClick={() => void deleteChat(c)}><Trash2 size={12} /></button></span>
        </div>)}
        {!visibleChats.length && <small className="history-empty">{search ? 'No chats match.' : 'Your conversations are saved on this device.'}</small>}
      </div>
      <div className="sidebar-bottom"><div className="profile-avatar">AI</div><div className="profile-copy"><strong>Local workspace</strong><span>Chats stay on this device</span></div><button className="icon-btn" onClick={() => setSettingsOpen(true)} title="Settings"><Settings2 size={17} /></button></div>
    </aside>}
    <main className="main">
      <header className="topbar">
        <div className="topbar-left">{!sidebarOpen && <button className="icon-btn" onClick={() => setSidebarOpen(true)}><Menu size={18} /></button>}<div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{view === 'chat' ? 'Chat' : 'Dashboard & Model Lab'}</strong></div><span className={`status-pill ${connected ? '' : 'offline'}`}><i /> {connected ? 'Ready' : 'Backend offline'}</span></div>
        <div className="topbar-right">
          <div className="theme-picker"><Sun size={15} /><select aria-label="Theme" value={theme} onChange={e => setTheme(e.target.value as typeof theme)}><option value="light">Light theme</option><option value="dark">Dark theme</option><option value="system">System default</option></select><ChevronDown size={13} /></div>
          <button className="settings-button" onClick={() => setSettingsOpen(true)}><Settings2 size={16} /> Settings</button>
        </div>
      </header>
      {view === 'dashboard'
        ? <section className="dashboard-scroll"><Dashboard data={dashboardData} loading={dashboardLoading} onRefresh={() => void loadDashboard()} onDetect={() => void detectApis()} detection={detection} catalog={catalog} onRefreshCatalog={() => void refreshCatalog()} catalogRefreshing={catalogRefreshing} onAddModel={openAddModel} onSetBilling={(p, m, b) => void setBilling(p, m, b)} onDeleteModel={(p, m) => void deleteModel(p, m)} onTestImage={(p, m) => void testImage(p, m)} testingModel={testingModel} onRefreshCatalogQuiet={() => void reloadCatalog()} />{error && <div className="error-banner">{error}</div>}</section>
        : <Chat providers={providers} prefs={prefs} setPrefs={setPrefs} chatId={chatId} onChatId={setChatId} onChatsChanged={() => void refreshChats()} reloadCatalog={() => void reloadCatalog()} catalogVersion={catalogVersion} connected={connected} />}
    </main>
    {settingsOpen&&<div className="modal-backdrop" onClick={()=>setSettingsOpen(false)}><section className="settings-modal" onClick={e=>e.stopPropagation()}><div className="modal-head"><div><span className="eyebrow">PREFERENCES</span><h2>Settings</h2></div><button className="icon-btn" onClick={()=>setSettingsOpen(false)}>×</button></div><p className="modal-desc">Manage API credentials. Keys are sent only to your local backend, which stores them using the OS credential store.</p><label>Provider<select value={keyProvider} onChange={e=>setKeyProvider(e.target.value)}>{providersForKey.map(p=><option value={p.id} key={p.id}>{p.name}</option>)}</select></label><label>API key<input type="password" value={apiKey} onChange={e=>setApiKey(e.target.value)} placeholder="Paste provider API key"/></label><button className="primary-wide" disabled={!apiKey.trim()} onClick={()=>void saveKey()}>Save API key securely</button><div className="security-note"><Settings2 size={16}/> Never share your API keys or commit them to source control.</div></section></div>}
    {addModelOpen&&<div className="modal-backdrop" onClick={()=>setAddModelOpen(false)}><section className="settings-modal" onClick={e=>e.stopPropagation()}><div className="modal-head"><div><span className="eyebrow">CATALOG</span><h2>Add a model</h2></div><button className="icon-btn" onClick={()=>setAddModelOpen(false)}><X size={18}/></button></div><p className="modal-desc">Use this for a model the provider list does not show yet, such as a newly released one. It is stored on this device only and is used by the dropdowns and Auto-pick.</p><label>Provider<select value={newModel.provider_id} onChange={e=>setNewModel(m=>({...m,provider_id:e.target.value}))}>{providersForKey.map(p=><option value={p.id} key={p.id}>{p.name}</option>)}</select></label><label>Model ID (exactly as the provider documents it)<input value={newModel.model_id} onChange={e=>setNewModel(m=>({...m,model_id:e.target.value}))} placeholder="e.g. a-new-model-id"/></label><label>Display name (optional)<input value={newModel.display_name} onChange={e=>setNewModel(m=>({...m,display_name:e.target.value}))} placeholder="Shown in menus"/></label><label>Billing<select value={newModel.billing} onChange={e=>setNewModel(m=>({...m,billing:e.target.value as Billing}))}><option value="unknown">Not sure</option><option value="free">Free</option><option value="paid">Paid / billed</option></select></label><label>Context window in tokens (optional)<input type="number" min={1000} value={newModel.context_window} onChange={e=>setNewModel(m=>({...m,context_window:e.target.value}))} placeholder="e.g. 200000"/></label><div className="cap-grid" role="group" aria-label="Capabilities">{['text','coding','analysis','vision','fast','image_generation'].map(c=><label key={c} title={c==='image_generation'?'Tick only for a model that returns images in chat, for example a Gemini image model':''}><input type="checkbox" checked={newModel.capabilities.includes(c)} onChange={e=>setNewModel(m=>({...m,capabilities:e.target.checked?[...m.capabilities,c]:m.capabilities.filter(x=>x!==c)}))}/> {c==='image_generation'?'image output':c}</label>)}</div>{addModelError&&<div className="form-error">{addModelError}</div>}<button className="primary-wide" disabled={!newModel.model_id.trim()||!newModel.provider_id} onClick={()=>void submitNewModel()}>Add model</button></section></div>}
  </div>;
}

export default App;
