import { loadArtifacts } from './model/artifacts';
import { scoreText } from './model/inference';

type Artifacts = Awaited<ReturnType<typeof loadArtifacts>>;

// Settings interface
interface Settings {
  enabled: boolean;
  showPlaceholders: boolean;
  highlightOnly: boolean;
}

// Page statistics
interface PageStats {
  scannedCount: number;
  toxicCount: number;
  labelCounts: { [key: string]: number };
}

// Global state
const processedNodes = new WeakSet<Node>();
let observer: MutationObserver | undefined;
let scheduled = false;
let artifactsPromise: Promise<Artifacts> | undefined;

// Statistics tracking
let stats: PageStats = {
  scannedCount: 0,
  toxicCount: 0,
  labelCounts: {}
};

// Current settings
let settings: Settings = {
  enabled: true,
  showPlaceholders: true,
  highlightOnly: false
};

// Load settings from storage
async function loadSettings(): Promise<void> {
  try {
    const result = await chrome.storage.local.get(['settings']);
    if (result.settings) {
      settings = result.settings;
    }
  } catch (error) {
    console.error('Error loading settings:', error);
  }
}

// Save stats to storage
async function saveStats(): Promise<void> {
  try {
    await chrome.storage.local.set({ pageStats: stats });
    // Notify popup of stats update
    chrome.runtime.sendMessage({ type: 'statsUpdated', stats }).catch(() => {
      // Popup might not be open, that's fine
    });
  } catch (error) {
    console.error('Error saving stats:', error);
  }
}

// Reset stats for new page
function resetStats(): void {
  stats = {
    scannedCount: 0,
    toxicCount: 0,
    labelCounts: {}
  };
  saveStats();
}

function splitSentences(text: string): string[] {
  const matches = text.match(/[^.!?\n\r]+[.!?\u2026]*\s*/g);
  if (!matches) {
    return [text];
  }
  return matches;
}

function injectStyles(): void {
  if (document.getElementById('toxiscope-style')) {
    return;
  }
  const style = document.createElement('style');
  style.id = 'toxiscope-style';
  style.textContent = `
    .toxiscope-mask {
      background-color: rgba(248, 113, 113, 0.15) !important;
      color: #b91c1c !important;
      font-style: italic !important;
      border-radius: 0.25rem !important;
      padding: 0.1rem 0.2rem !important;
      display: inline !important;
    }
    .toxiscope-highlight {
      background-color: rgba(255, 193, 7, 0.3) !important;
      border-bottom: 2px solid #f44336 !important;
      border-radius: 2px !important;
      display: inline !important;
    }
    }
    .toxiscope-highlight[data-toxicity-labels]:hover::after {
      content: attr(data-toxicity-labels);
      position: absolute;
      background: #333;
      color: #fff;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      white-space: nowrap;
      z-index: 10000;
      transform: translateY(-100%);
      margin-top: -4px;
    }
    .toxiscope-toxic-toxic { border-left: 3px solid #f44336; padding-left: 4px; }
    .toxiscope-toxic-obscene { border-left: 3px solid #ff9800; padding-left: 4px; }
    .toxiscope-toxic-insult { border-left: 3px solid #9c27b0; padding-left: 4px; }
    .toxiscope-toxic-threat { border-left: 3px solid #e91e63; padding-left: 4px; }
    .toxiscope-toxic-identity_hate { border-left: 3px solid #673ab7; padding-left: 4px; }
    .toxiscope-toxic-racism { border-left: 3px solid #3f51b5; padding-left: 4px; }
  `;
  document.head.appendChild(style);
}

async function getArtifacts(): Promise<Artifacts> {
  if (!artifactsPromise) {
    artifactsPromise = loadArtifacts();
  }
  return artifactsPromise;
}

function createPlaceholder(labels: string[], trailingWhitespace: string): Node {
  const span = document.createElement('span');
  span.className = 'toxiscope-mask';
  span.setAttribute('data-toxiscope', 'placeholder');
  span.setAttribute('data-toxicity-labels', labels.join(', '));
  const reason = labels.join(', ');
  span.textContent = `<this sentence was removed for ${reason}>${trailingWhitespace}`;
  processedNodes.add(span);
  return span;
}

function createHighlight(originalText: string, labels: string[]): Node {
  const span = document.createElement('span');
  span.className = 'toxiscope-highlight';
  span.setAttribute('data-toxiscope', 'highlight');
  span.setAttribute('data-toxicity-labels', labels.join(', '));
  span.setAttribute('title', `Detected: ${labels.join(', ')}`);
  
  // Add label-specific styling for the first/main label
  if (labels.length > 0) {
    span.classList.add(`toxiscope-toxic-${labels[0]}`);
  }
  
  span.textContent = originalText;
  processedNodes.add(span);
  return span;
}

function analyseTextNode(node: Text, artifacts: Artifacts): void {
  if (processedNodes.has(node)) {
    return;
  }

  // Check if detection is enabled
  if (!settings.enabled) {
    processedNodes.add(node);
    return;
  }

  const original = node.textContent ?? '';
  if (!original.trim()) {
    processedNodes.add(node);
    return;
  }

  const sentences = splitSentences(original);
  let modified = false;
  const fragments: Node[] = [];

  sentences.forEach((segment) => {
    const trimmed = segment.trim();
    if (!trimmed) {
      const textNode = document.createTextNode(segment);
      processedNodes.add(textNode);
      fragments.push(textNode);
      return;
    }

    // Track scanned count
    stats.scannedCount++;

    const result = scoreText(trimmed, artifacts);
    if (result.activeLabels.length === 0) {
      const cleanNode = document.createTextNode(segment);
      processedNodes.add(cleanNode);
      fragments.push(cleanNode);
      return;
    }

    // Track toxic content
    stats.toxicCount++;
    console.log(`[ToxiScope] Detected toxic content: "${trimmed.substring(0, 30)}..." labels: ${result.activeLabels.join(', ')}`);
    result.activeLabels.forEach(label => {
      stats.labelCounts[label] = (stats.labelCounts[label] || 0) + 1;
    });

    modified = true;
    const trailing = segment.match(/\s+$/)?.[0] ?? '';
    
    // Choose between placeholder and highlight based on settings
    if (settings.highlightOnly) {
      fragments.push(createHighlight(segment, result.activeLabels));
    } else if (settings.showPlaceholders) {
      fragments.push(createPlaceholder(result.activeLabels, trailing));
    } else {
      // Neither placeholder nor highlight - just mark as processed
      const cleanNode = document.createTextNode(segment);
      processedNodes.add(cleanNode);
      fragments.push(cleanNode);
    }
  });

  if (!modified) {
    processedNodes.add(node);
    return;
  }

  const fragment = document.createDocumentFragment();
  fragments.forEach((child) => {
    fragment.appendChild(child);
  });

  const parent = node.parentNode;
  if (!parent) {
    return;
  }
  parent.replaceChild(fragment, node);
}

function walkAndAnalyse(root: Node, artifacts: Artifacts): void {
  const nodesToProcess: Text[] = [];
  const treeWalker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (processedNodes.has(node)) {
        return NodeFilter.FILTER_REJECT;
      }
      const parent = node.parentElement;
      if (!parent) {
        return NodeFilter.FILTER_REJECT;
      }
      if (parent.closest('[data-toxiscope="placeholder"]') || 
          parent.closest('[data-toxiscope="highlight"]')) {
        return NodeFilter.FILTER_REJECT;
      }
      const tagName = parent.tagName;
      if (
        tagName === 'SCRIPT' ||
        tagName === 'STYLE' ||
        tagName === 'NOSCRIPT' ||
        tagName === 'TEXTAREA' ||
        tagName === 'INPUT'
      ) {
        return NodeFilter.FILTER_REJECT;
      }
      if (parent.isContentEditable) {
        return NodeFilter.FILTER_REJECT;
      }
      const text = node.textContent ?? '';
      if (!/\w/.test(text)) {
        return NodeFilter.FILTER_SKIP;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let current: Node | null = treeWalker.nextNode();
  while (current) {
    nodesToProcess.push(current as Text);
    current = treeWalker.nextNode();
  }

  if (nodesToProcess.length > 0) {
    console.log(`[ToxiScope] Scanning ${nodesToProcess.length} new text nodes...`);
    nodesToProcess.forEach(node => analyseTextNode(node, artifacts));
  }
}

async function performScan(): Promise<void> {
  scheduled = false;
  
  // Check if enabled
  if (!settings.enabled) {
    return;
  }
  
  try {
    const artifacts = await getArtifacts();
    const body = document.body;
    if (body) {
      walkAndAnalyse(body, artifacts);
      // Save stats after scan
      saveStats();
    }
  } catch (error) {
    console.error('[ToxiScope] Failed to load artifacts:', error);
    observer?.disconnect();
  }
}

function scheduleScan(): void {
  if (scheduled) {
    return;
  }
  scheduled = true;
  setTimeout(() => {
    void performScan();
  }, 250);
}

function startObserver(): void {
  if (observer) {
    observer.disconnect();
  }
  observer = new MutationObserver((mutations) => {
    const relevant = mutations.some((mutation) => {
      if (mutation.type === 'characterData') {
        return true;
      }
      return mutation.addedNodes.length > 0;
    });
    if (relevant) {
      scheduleScan();
    }
  });
  observer.observe(document.body, {
    childList: true,
    characterData: true,
    subtree: true,
  });
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'settingsChanged') {
    settings = message.settings;
    // Could trigger rescan if needed
  } else if (message.type === 'rescan') {
    // Force rescan by clearing processed nodes (WeakSet doesn't have clear, so we reset)
    resetStats();
    // Reload the page content - simplified approach
    window.location.reload();
  } else if (message.type === 'getStats') {
    sendResponse({ stats });
    saveStats();
  }
  return true;
});

async function init(): Promise<void> {
  if (!document.body) {
    return;
  }
  
  console.log('[ToxiScope] Initializing content script...');
  
  // Load settings first
  await loadSettings();
  
  // Reset stats for new page
  resetStats();
  
  injectStyles();
  scheduleScan();
  startObserver();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => void init(), { once: true });
} else {
  void init();
}
