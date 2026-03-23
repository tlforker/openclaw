import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { AssistantMessage } from "@mariozechner/pi-ai";
import { extractTextFromChatContent } from "../shared/chat-content.js";
import { stripReasoningTagsFromText } from "../shared/text/reasoning-tags.js";
import { sanitizeUserFacingText } from "./pi-embedded-helpers.js";
import { formatToolDetail, resolveToolDisplay } from "./tool-display.js";

export function isAssistantMessage(msg: AgentMessage | undefined): msg is AssistantMessage {
  return msg?.role === "assistant";
}

/**
 * Strip malformed Minimax tool invocations that leak into text content.
 * Minimax sometimes embeds tool calls as XML in text blocks instead of
 * proper structured tool calls. This removes:
 * - <invoke name="...">...</invoke> blocks
 * - </minimax:tool_call> closing tags
 */
export function stripMinimaxToolCallXml(text: string): string {
  if (!text) {
    return text;
  }
  if (!/minimax:tool_call/i.test(text)) {
    return text;
  }

  // Remove <invoke ...>...</invoke> blocks (non-greedy to handle multiple).
  let cleaned = text.replace(/<invoke\b[^>]*>[\s\S]*?<\/invoke>/gi, "");

  // Remove stray minimax tool tags.
  cleaned = cleaned.replace(/<\/?minimax:tool_call>/gi, "");

  return cleaned;
}

/**
 * Strip DeepSeek DSML tool call XML that leaks into text content.
 * DeepSeek models (via NVIDIA NIM) sometimes embed tool calls using DSML
 * format in text blocks instead of proper structured tool calls. This removes:
 * - <｜DSML｜function_calls>...</｜DSML｜function_calls> blocks
 * - Stray remaining DSML tags
 * The ｜ character is U+FF5C (FULLWIDTH VERTICAL LINE).
 */
export function stripDeepSeekDsmlToolCallXml(text: string): string {
  if (!text) {
    return text;
  }
  // Fast-path: DSML tags always contain the U+FF5C fullwidth vertical bar
  if (!text.includes("\uFF5CDSML\uFF5C")) {
    return text;
  }

  // Remove complete <｜DSML｜function_calls>...</｜DSML｜function_calls> blocks.
  let cleaned = text.replace(
    /<\uFF5CDSML\uFF5Cfunction_calls>[\s\S]*?<\/\uFF5CDSML\uFF5Cfunction_calls>/gi,
    "",
  );

  // Remove any remaining stray DSML open/close tags.
  cleaned = cleaned.replace(/<\/?\uFF5CDSML\uFF5C[^>]*>/g, "");

  return cleaned;
}

/**
 * Extract actual text from NIM-serialized content blocks.
 * NVIDIA NIM-hosted models (kimi-k2.5, deepseek-v3.2) sometimes emit their
 * response as a Python-formatted content array rather than plain text, e.g.:
 *   [{'type': 'text', 'text': 'actual response here'}]
 * This detects that pattern and extracts the text values from it iteratively
 * by performing in-place replacement, handling multiple levels of nesting,
 * escaping (including escaped keys), and preserving surrounding text.
 */
export function extractFromNimSerializedContent(text: string): string {
  if (!text || typeof text !== "string") {
    return text;
  }
  let current = text;
  let iterations = 0;

  // Function to find the end of a quoted string handling backslash escaping
  const findValueEnd = (str: string, startIdx: number, quoteChar: string): number => {
    let escaped = false;
    for (let i = startIdx; i < str.length; i++) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (str[i] === "\\") {
        escaped = true;
        continue;
      }
      if (str[i] === quoteChar) {
        return i;
      }
    }
    return -1;
  };

  // Continuously unwrap as long as we find NIM-serialized blocks
  while (iterations < 10) {
    // Flexible regex to find the 'text' key even if it has leading backslashes (escaped in nested layers)
    const textKeyRe = /\\*['"]text\\*['"]\s*:\s*(['"])/g;
    let match: RegExpExecArray | null;
    let found = false;

    // Use a temporary list of replacements to perform them all at once at the end of the pass
    const replacements: Array<{ start: number; end: number; content: string }> = [];

    while ((match = textKeyRe.exec(current)) !== null) {
      const quoteChar = match[1];
      const valueStartIdx = match.index + match[0].length;
      const valueEndIdx = findValueEnd(current, valueStartIdx, quoteChar);
      if (valueEndIdx === -1) {
        continue;
      }

      // Verify this 'text' key is part of a NIM block by looking for 'type': 'text' in the containing dict
      const beforeMatch = current.substring(0, match.index);
      const lastOpenBrace = beforeMatch.lastIndexOf("{");
      if (lastOpenBrace === -1) {
        continue;
      }

      const dictContent = current.substring(lastOpenBrace, valueEndIdx + 1);
      // NIM blocks must contain a type: text field (also potentially escaped)
      if (!/\\*['"]type\\*['"]\s*:\s*\\*['"]text\\*['"]/.test(dictContent)) {
        continue;
      }

      // Extract raw value and unescape one level
      const rawValue = current.substring(valueStartIdx, valueEndIdx);
      const unescaped = rawValue.replace(/\\(.)/g, (m, char) => {
        if (char === "\\") {
          return "\\";
        }
        if (char === "'") {
          return "'";
        }
        if (char === '"') {
          return '"';
        }
        if (char === "n") {
          return "\n";
        }
        if (char === "r") {
          return "\r";
        }
        if (char === "t") {
          return "\t";
        }
        return char;
      });

      // Determine the boundaries of the containing block ([{...}]) to replace
      let blockStart = lastOpenBrace;
      const prefix = current.substring(0, blockStart);
      if (prefix.trim().endsWith("[")) {
        blockStart = prefix.lastIndexOf("[");
      }

      let blockEnd = valueEndIdx + 1;
      const suffix = current.substring(blockEnd);
      const firstCloseBrace = suffix.indexOf("}");
      if (firstCloseBrace !== -1) {
        blockEnd = valueEndIdx + 1 + firstCloseBrace + 1;
        const remaining = current.substring(blockEnd);
        if (remaining.trim().startsWith("]")) {
          blockEnd += remaining.indexOf("]") + 1;
        }
      }

      replacements.push({ start: blockStart, end: blockEnd, content: unescaped });
      found = true;
      // Advance regex to skip the rest of this block
      textKeyRe.lastIndex = blockEnd;
    }

    if (!found) {
      break;
    }

    // Apply replacements in reverse order to keep indices valid
    let nextCurrent = current;
    for (let i = replacements.length - 1; i >= 0; i--) {
      const { start, end, content: unescaped } = replacements[i];
      nextCurrent = nextCurrent.substring(0, start) + unescaped + nextCurrent.substring(end);
    }
    current = nextCurrent;
    iterations++;
  }

  return current;
}

/**
 * Strip downgraded tool call text representations that leak into text content.
 * When replaying history to Gemini, tool calls without `thought_signature` are
 * downgraded to text blocks like `[Tool Call: name (ID: ...)]`. These should
 * not be shown to users.
 */
export function stripDowngradedToolCallText(text: string): string {
  if (!text) {
    return text;
  }
  if (!/\[Tool (?:Call|Result)/i.test(text) && !/\[Historical context/i.test(text)) {
    return text;
  }

  const consumeJsonish = (
    input: string,
    start: number,
    options?: { allowLeadingNewlines?: boolean },
  ): number | null => {
    const { allowLeadingNewlines = false } = options ?? {};
    let index = start;
    while (index < input.length) {
      const ch = input[index];
      if (ch === " " || ch === "\t") {
        index += 1;
        continue;
      }
      if (allowLeadingNewlines && (ch === "\n" || ch === "\r")) {
        index += 1;
        continue;
      }
      break;
    }
    if (index >= input.length) {
      return null;
    }

    const startChar = input[index];
    if (startChar === "{" || startChar === "[") {
      let depth = 0;
      let inString = false;
      let escape = false;
      for (let i = index; i < input.length; i += 1) {
        const ch = input[i];
        if (inString) {
          if (escape) {
            escape = false;
          } else if (ch === "\\") {
            escape = true;
          } else if (ch === '"') {
            inString = false;
          }
          continue;
        }
        if (ch === '"') {
          inString = true;
          continue;
        }
        if (ch === "{" || ch === "[") {
          depth += 1;
          continue;
        }
        if (ch === "}" || ch === "]") {
          depth -= 1;
          if (depth === 0) {
            return i + 1;
          }
        }
      }
      return null;
    }

    if (startChar === '"') {
      let escape = false;
      for (let i = index + 1; i < input.length; i += 1) {
        const ch = input[i];
        if (escape) {
          escape = false;
          continue;
        }
        if (ch === "\\") {
          escape = true;
          continue;
        }
        if (ch === '"') {
          return i + 1;
        }
      }
      return null;
    }

    let end = index;
    while (end < input.length && input[end] !== "\n" && input[end] !== "\r") {
      end += 1;
    }
    return end;
  };

  const stripToolCalls = (input: string): string => {
    const markerRe = /\[Tool Call:[^\]]*\]/gi;
    let result = "";
    let cursor = 0;
    for (const match of input.matchAll(markerRe)) {
      const start = match.index ?? 0;
      if (start < cursor) {
        continue;
      }
      result += input.slice(cursor, start);
      let index = start + match[0].length;
      while (index < input.length && (input[index] === " " || input[index] === "\t")) {
        index += 1;
      }
      if (input[index] === "\r") {
        index += 1;
        if (input[index] === "\n") {
          index += 1;
        }
      } else if (input[index] === "\n") {
        index += 1;
      }
      while (index < input.length && (input[index] === " " || input[index] === "\t")) {
        index += 1;
      }
      if (input.slice(index, index + 9).toLowerCase() === "arguments") {
        index += 9;
        if (input[index] === ":") {
          index += 1;
        }
        if (input[index] === " ") {
          index += 1;
        }
        const end = consumeJsonish(input, index, { allowLeadingNewlines: true });
        if (end !== null) {
          index = end;
        }
      }
      if (
        (input[index] === "\n" || input[index] === "\r") &&
        (result.endsWith("\n") || result.endsWith("\r") || result.length === 0)
      ) {
        if (input[index] === "\r") {
          index += 1;
        }
        if (input[index] === "\n") {
          index += 1;
        }
      }
      cursor = index;
    }
    result += input.slice(cursor);
    return result;
  };

  // Remove [Tool Call: name (ID: ...)] blocks and their Arguments.
  let cleaned = stripToolCalls(text);

  // Remove [Tool Result for ID ...] blocks and their content.
  cleaned = cleaned.replace(/\[Tool Result for ID[^\]]*\]\n?[\s\S]*?(?=\n*\[Tool |\n*$)/gi, "");

  // Remove [Historical context: ...] markers (self-contained within brackets).
  cleaned = cleaned.replace(/\[Historical context:[^\]]*\]\n?/gi, "");

  return cleaned.trim();
}

/**
 * Strip thinking tags and their content from text.
 * This is a safety net for cases where the model outputs <think> tags
 * that slip through other filtering mechanisms.
 */
export function stripThinkingTagsFromText(text: string): string {
  return stripReasoningTagsFromText(text, { mode: "strict", trim: "both" });
}

export function extractAssistantText(msg: AssistantMessage): string {
  const extracted =
    extractTextFromChatContent(msg.content, {
      sanitizeText: (text) =>
        stripThinkingTagsFromText(
          stripDowngradedToolCallText(
            stripDeepSeekDsmlToolCallXml(
              stripMinimaxToolCallXml(extractFromNimSerializedContent(text)),
            ),
          ),
        ).trim(),
      joinWith: "\n",
      normalizeText: (text) => text.trim(),
    }) ?? "";
  // Only apply keyword-based error rewrites when the assistant message is actually an error.
  // Otherwise normal prose that *mentions* errors (e.g. "context overflow") can get clobbered.
  const errorContext = msg.stopReason === "error" || Boolean(msg.errorMessage?.trim());
  return sanitizeUserFacingText(extracted, { errorContext });
}

export function extractAssistantThinking(msg: AssistantMessage): string {
  if (!Array.isArray(msg.content)) {
    return "";
  }
  const blocks = msg.content
    .map((block) => {
      if (!block || typeof block !== "object") {
        return "";
      }
      const record = block as unknown as Record<string, unknown>;
      if (record.type === "thinking" && typeof record.thinking === "string") {
        return record.thinking.trim();
      }
      return "";
    })
    .filter(Boolean);
  return blocks.join("\n").trim();
}

export function formatReasoningMessage(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) {
    return "";
  }
  // Show reasoning in italics (cursive) for markdown-friendly surfaces (Discord, etc.).
  // Keep the plain "Reasoning:" prefix so existing parsing/detection keeps working.
  // Note: Underscore markdown cannot span multiple lines on Telegram, so we wrap
  // each non-empty line separately.
  const italicLines = trimmed
    .split("\n")
    .map((line) => (line ? `_${line}_` : line))
    .join("\n");
  return `Reasoning:\n${italicLines}`;
}

type ThinkTaggedSplitBlock =
  | { type: "thinking"; thinking: string }
  | { type: "text"; text: string };

export function splitThinkingTaggedText(text: string): ThinkTaggedSplitBlock[] | null {
  const trimmedStart = text.trimStart();
  // Avoid false positives: only treat it as structured thinking when it begins
  // with a think tag (common for local/OpenAI-compat providers that emulate
  // reasoning blocks via tags).
  if (!trimmedStart.startsWith("<")) {
    return null;
  }
  const openRe = /<\s*(?:think(?:ing)?|thought|antthinking)\s*>/i;
  const closeRe = /<\s*\/\s*(?:think(?:ing)?|thought|antthinking)\s*>/i;
  if (!openRe.test(trimmedStart)) {
    return null;
  }
  if (!closeRe.test(text)) {
    return null;
  }

  const scanRe = /<\s*(\/?)\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  let inThinking = false;
  let cursor = 0;
  let thinkingStart = 0;
  const blocks: ThinkTaggedSplitBlock[] = [];

  const pushText = (value: string) => {
    if (!value) {
      return;
    }
    blocks.push({ type: "text", text: value });
  };
  const pushThinking = (value: string) => {
    const cleaned = value.trim();
    if (!cleaned) {
      return;
    }
    blocks.push({ type: "thinking", thinking: cleaned });
  };

  for (const match of text.matchAll(scanRe)) {
    const index = match.index ?? 0;
    const isClose = Boolean(match[1]?.includes("/"));

    if (!inThinking && !isClose) {
      pushText(text.slice(cursor, index));
      thinkingStart = index + match[0].length;
      inThinking = true;
      continue;
    }

    if (inThinking && isClose) {
      pushThinking(text.slice(thinkingStart, index));
      cursor = index + match[0].length;
      inThinking = false;
    }
  }

  if (inThinking) {
    return null;
  }
  pushText(text.slice(cursor));

  const hasThinking = blocks.some((b) => b.type === "thinking");
  if (!hasThinking) {
    return null;
  }
  return blocks;
}

export function promoteThinkingTagsToBlocks(message: AssistantMessage): void {
  if (!Array.isArray(message.content)) {
    return;
  }
  const hasThinkingBlock = message.content.some(
    (block) => block && typeof block === "object" && block.type === "thinking",
  );
  if (hasThinkingBlock) {
    return;
  }

  const next: AssistantMessage["content"] = [];
  let changed = false;

  for (const block of message.content) {
    if (!block || typeof block !== "object" || !("type" in block)) {
      next.push(block);
      continue;
    }
    if (block.type !== "text") {
      next.push(block);
      continue;
    }
    const split = splitThinkingTaggedText(block.text);
    if (!split) {
      next.push(block);
      continue;
    }
    changed = true;
    for (const part of split) {
      if (part.type === "thinking") {
        next.push({ type: "thinking", thinking: part.thinking });
      } else if (part.type === "text") {
        const cleaned = part.text.trimStart();
        if (cleaned) {
          next.push({ type: "text", text: cleaned });
        }
      }
    }
  }

  if (!changed) {
    return;
  }
  message.content = next;
}

export function extractThinkingFromTaggedText(text: string): string {
  if (!text) {
    return "";
  }
  const scanRe = /<\s*(\/?)\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  let result = "";
  let lastIndex = 0;
  let inThinking = false;
  for (const match of text.matchAll(scanRe)) {
    const idx = match.index ?? 0;
    if (inThinking) {
      result += text.slice(lastIndex, idx);
    }
    const isClose = match[1] === "/";
    inThinking = !isClose;
    lastIndex = idx + match[0].length;
  }
  return result.trim();
}

export function extractThinkingFromTaggedStream(text: string): string {
  if (!text) {
    return "";
  }
  const closed = extractThinkingFromTaggedText(text);
  if (closed) {
    return closed;
  }

  const openRe = /<\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  const closeRe = /<\s*\/\s*(?:think(?:ing)?|thought|antthinking)\s*>/gi;
  const openMatches = [...text.matchAll(openRe)];
  if (openMatches.length === 0) {
    return "";
  }
  const closeMatches = [...text.matchAll(closeRe)];
  const lastOpen = openMatches[openMatches.length - 1];
  const lastClose = closeMatches[closeMatches.length - 1];
  if (lastClose && (lastClose.index ?? -1) > (lastOpen.index ?? -1)) {
    return closed;
  }
  const start = (lastOpen.index ?? 0) + lastOpen[0].length;
  return text.slice(start).trim();
}

export function inferToolMetaFromArgs(toolName: string, args: unknown): string | undefined {
  const display = resolveToolDisplay({ name: toolName, args });
  return formatToolDetail(display);
}
