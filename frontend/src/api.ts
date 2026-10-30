export const API: string = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

export const errorText = (data: any): string =>
  typeof data?.detail === 'string' ? data.detail : data?.detail?.message || (data?.detail ? JSON.stringify(data.detail) : 'Request failed');

export async function request<T = any>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try { res = await fetch(`${API}${path}`, init); }
  catch { throw new Error('Backend is not connected. Start the FastAPI server using the README instructions.'); }
  let data: any = null;
  try { data = await res.json(); } catch { /* empty or non-JSON body */ }
  if (!res.ok) throw new Error(errorText(data));
  return data as T;
}

export const json = (method: string, body: unknown): RequestInit =>
  ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const fileUrl = (chatId: string, fileId: number) => `${API}/chats/${encodeURIComponent(chatId)}/files/${fileId}`;
export const newChatId = () => (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '') : String(Date.now()) + Math.random().toString(16).slice(2));
