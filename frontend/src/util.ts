import type { Attached } from './types';

export const MAX_FILE_MB = 25;
export const MAX_TOTAL_MB = 60;

export const readAsBase64 = (file: File) => new Promise<string>((resolve, reject) => {
  const r = new FileReader();
  r.onload = () => resolve(String(r.result).split(',')[1] || '');
  r.onerror = () => reject(new Error(`Could not read ${file.name}`));
  r.readAsDataURL(file);
});

export const kindOf = (a: { name: string; mime_type: string }) =>
  a.mime_type.startsWith('image/') ? 'image'
  : a.mime_type === 'application/pdf' || /\.pdf$/i.test(a.name) ? 'pdf'
  : a.mime_type.startsWith('audio/') ? 'audio'
  : a.mime_type.startsWith('video/') ? 'video' : '';

export const fmtSize = (n: number) => n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;

export async function copyText(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { return false; }
}

export async function filesToAttached(list: File[], existing: Attached[]): Promise<{ added: Attached[]; errors: string[] }> {
  const added: Attached[] = []; const errors: string[] = [];
  let total = existing.reduce((n, a) => n + a.size, 0);
  for (const f of list) {
    if (f.size > MAX_FILE_MB * 1048576) { errors.push(`${f.name} is larger than ${MAX_FILE_MB} MB.`); continue; }
    if (total + f.size > MAX_TOTAL_MB * 1048576) { errors.push(`Attachments can total at most ${MAX_TOTAL_MB} MB.`); break; }
    try {
      const name = f.name || (f.type.startsWith('image/') ? `pasted-image.${f.type.split('/')[1] || 'png'}` : 'attachment');
      added.push({ name, mime_type: f.type, size: f.size, data: await readAsBase64(f), previewUrl: f.type.startsWith('image/') ? URL.createObjectURL(f) : undefined });
      total += f.size;
    } catch (e) { errors.push(e instanceof Error ? e.message : 'Could not read file'); }
  }
  return { added, errors };
}
