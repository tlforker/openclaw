import { describe, expect, it } from "vitest";
import { extractFromNimSerializedContent } from "./pi-embedded-utils.js";

describe("extractFromNimSerializedContent", () => {
  it("extracts text from simple single-quoted NIM serialization", () => {
    const input = "[{'type': 'text', 'text': 'Hello world'}]";
    expect(extractFromNimSerializedContent(input)).toBe("Hello world");
  });

  it("extracts text from single-quoted NIM serialization with escaped quotes", () => {
    const input = "[{'type': 'text', 'text': 'It\\'s a test'}]";
    expect(extractFromNimSerializedContent(input)).toBe("It's a test");
  });

  it("extracts and joins multiple text blocks", () => {
    const input = "[{'type': 'text', 'text': 'Part 1'}, {'type': 'text', 'text': 'Part 2'}]";
    expect(extractFromNimSerializedContent(input)).toBe("Part 1, Part 2");
  });

  it("handles triple nesting with escaped keys (User Reported Case)", () => {
    // Exact structure reported by user where even keys are escaped
    const input =
      "[{'type': 'text', 'text': \"[{'type': 'text', 'text': '[{\\\\\\'type\\\\\\': \\\\\\'text\\\\\\', \\\\\\'text\\\\\\': \\\\\\'Status Report\\\\\\'}]'}]\"}]";
    const result = extractFromNimSerializedContent(input);
    expect(result).toBe("Status Report");
  });

  it("preserves text outside the blocks", () => {
    const input = "Prefix [{'type': 'text', 'text': 'Content'}] Suffix";
    expect(extractFromNimSerializedContent(input)).toBe("Prefix Content Suffix");
  });
});
