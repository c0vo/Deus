"use client";

import { marked } from "marked";

// Configure marked for safe GitHub-Flavored Markdown rendering.
marked.setOptions({
  gfm: true,
  breaks: true, // Preserve single newlines as line breaks <br />
});

/**
 * Convert lightweight Markdown to HTML using `marked`.
 * Performs safe, non-destructive normalization (CRLF conversion, spacing ATX headers).
 */
export function formatGeneratedText(value?: string | null): string {
  if (!value) return "";

  let text = value.replace(/\r\n/g, "\n");

  // 1. Separate "Executive Summary" / "Analyst Conclusion" fused to body text ("Executive SummarySK Hynix" → "### **Executive Summary**\n\nSK Hynix")
  text = text.replace(
    /(^|\n|\s*)(Executive Summary|Analyst Summary|Analyst Conclusion|Conclusion)([A-Z0-9])/g,
    "$1\n\n### **$2**\n\n$3"
  );

  // 2. Separate numbered headers stuck to body text ("1. Title NameBody Text" → "\n\n### **1. Title Name**\n\nBody Text")
  text = text.replace(
    /(^|\n|\s*)(\d+\.\s+[A-Z0-9\s,&'/:-]{5,60}?)([a-z0-9])([A-Z][a-z][^\n]*)/g,
    (match, p1, p2, p3, p4) => {
      const title = (p2 + p3).trim();
      return `${p1}\n\n### **${title}**\n\n${p4}`;
    }
  );

  // 3. Separate fused mid-line bullets ("sentence.- Subtitle:" → "sentence.\n\n- **Subtitle:**")
  // ONLY match if stuck mid-line without a preceding newline and not already a bullet item
  text = text.replace(/([a-zA-Z0-9.\)])\s*-\s*([A-Z][A-Za-z0-9\s,/&':-]{2,40}?:)/g, "$1\n\n- **$2**");

  // 4. Ensure space after ATX heading hashes if missing ("###Header" → "### Header")
  text = text.replace(/(^|\n)(#{1,6})([^\s#])/g, "$1$2 $3");

  // 5. Ensure blank line before mid-text headings ("text. ### Header" → "text.\n\n### Header")
  text = text.replace(/([^\n#])\s*(#{1,6}\s)/g, "$1\n\n$2");

  // 6. Ensure blank line AFTER heading line before paragraph text
  text = text.replace(/(^|\n)(#{1,6}\s+[^\n]+)\n(?!\n)([^\n])/g, "$1$2\n\n$3");

  // ── Parse with marked ──
  return marked.parse(text, { async: false }) as string;
}

export default function FormattedText({
  text,
  className = "",
}: {
  text?: string | null;
  className?: string;
}) {
  return (
    <div
      className={`formatted-text ${className}`}
      dangerouslySetInnerHTML={{ __html: formatGeneratedText(text) }}
    />
  );
}

