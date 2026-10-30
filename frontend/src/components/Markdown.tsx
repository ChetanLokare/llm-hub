import { Check, Copy } from 'lucide-react';
import { Children, isValidElement, useRef, useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { copyText } from '../util';

function CodeBlock({ children }: { children?: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const child = Children.toArray(children)[0];
  const lang = isValidElement(child) ? /language-(\S+)/.exec(String((child.props as { className?: string }).className || ''))?.[1] : undefined;
  const copy = async () => {
    const code = ref.current?.querySelector('code')?.textContent || '';
    if (await copyText(code)) { setCopied(true); window.setTimeout(() => setCopied(false), 1500); }
  };
  return (
    <div className="code-block">
      <div className="code-head"><span>{lang || 'code'}</span>
        <button className="copy-code-btn" onClick={() => void copy()} title="Copy code">{copied ? <><Check size={13} /> Copied</> : <><Copy size={13} /> Copy code</>}</button>
      </div>
      <pre ref={ref}>{children}</pre>
    </div>
  );
}

export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
      pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
      a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
    }}>{text}</ReactMarkdown>
  );
}
