export interface TextSelectionOffsets {
  text: string;
  start: number;
  end: number;
}

export function getSelectionOffsets(
  container: HTMLElement,
  selection: Selection | null,
): TextSelectionOffsets | null {
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return null;
  }
  const range = selection.getRangeAt(0);
  const ancestor = range.commonAncestorContainer;
  if (ancestor !== container && !container.contains(ancestor)) {
    return null;
  }

  const prefix = range.cloneRange();
  prefix.selectNodeContents(container);
  prefix.setEnd(range.startContainer, range.startOffset);
  // Python validates offsets as Unicode code points; Array.from keeps browser
  // offsets consistent even if a passage contains non-BMP characters.
  const start = Array.from(prefix.toString()).length;
  const text = range.toString();
  if (!text.trim()) {
    return null;
  }
  return { text, start, end: start + Array.from(text).length };
}
