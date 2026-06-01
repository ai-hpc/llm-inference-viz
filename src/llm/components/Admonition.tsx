import React from 'react';
import clsx from 'clsx';

// Clean documentation-style callout boxes (Tip / Note / Warning), in the spirit of the
// Qwen / Read-the-Docs admonitions: a colored left rule, a soft tint, an icon, a label.

type Kind = 'tip' | 'note' | 'warning' | 'key';

const STYLES: Record<Kind, { wrap: string; label: string; body: string; name: string }> = {
    tip: { wrap: 'border-emerald-400 bg-emerald-50/70', label: 'text-emerald-700', body: 'text-emerald-950/80', name: 'Tip' },
    note: { wrap: 'border-sky-400 bg-sky-50/70', label: 'text-sky-700', body: 'text-sky-950/80', name: 'Note' },
    warning: { wrap: 'border-amber-400 bg-amber-50/70', label: 'text-amber-700', body: 'text-amber-950/80', name: 'Warning' },
    key: { wrap: 'border-violet-400 bg-violet-50/70', label: 'text-violet-700', body: 'text-violet-950/80', name: 'Key idea' },
};

function Icon({ kind, className }: { kind: Kind; className?: string }) {
    const common = { width: 12, height: 12, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2.2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, className };
    if (kind === 'tip') return <svg {...common}><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V18h6v-1.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2Z" /></svg>;
    if (kind === 'warning') return <svg {...common}><path d="M10.3 3.5 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.5a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4M12 17h.01" /></svg>;
    if (kind === 'key') return <svg {...common}><circle cx="7.5" cy="15.5" r="5.5" /><path d="m21 2-9.6 9.6M15.5 7.5l3 3" /></svg>;
    return <svg {...common}><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></svg>; // note / info
}

export const Admonition: React.FC<{ kind: Kind; title?: string; children: React.ReactNode }> = ({ kind, title, children }) => {
    const s = STYLES[kind];
    return <div className={clsx('rounded-r-md border-l-[3px] px-2.5 py-2', s.wrap)}>
        <div className={clsx('mb-1 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider', s.label)}>
            <Icon kind={kind} />
            {title ?? s.name}
        </div>
        <div className={clsx('text-[11px] leading-relaxed', s.body)}>{children}</div>
    </div>;
};
