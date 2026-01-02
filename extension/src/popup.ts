// Popup script for ToxiScope extension

interface PageStats {
  scannedCount: number;
  toxicCount: number;
  labelCounts: { [key: string]: number };
}

interface Settings {
  enabled: boolean;
  showPlaceholders: boolean;
  highlightOnly: boolean;
}

// Default settings
const defaultSettings: Settings = {
  enabled: true,
  showPlaceholders: true,
  highlightOnly: false
};

// Load stats from storage and update UI
async function loadStats(): Promise<void> {
  try {
    const result = await chrome.storage.local.get(['pageStats']);
    const stats: PageStats = result.pageStats || {
      scannedCount: 0,
      toxicCount: 0,
      labelCounts: {}
    };
    updateStatsUI(stats);
  } catch (error) {
    console.error('Error loading stats:', error);
  }
}

// Update the stats display in the popup
function updateStatsUI(stats: PageStats): void {
  const scannedEl = document.getElementById('scanned-count');
  const toxicEl = document.getElementById('toxic-count');
  const rateEl = document.getElementById('toxicity-rate');

  if (scannedEl) scannedEl.textContent = stats.scannedCount.toString();
  if (toxicEl) toxicEl.textContent = stats.toxicCount.toString();
  
  if (rateEl) {
    const rate = stats.scannedCount > 0 
      ? ((stats.toxicCount / stats.scannedCount) * 100).toFixed(1)
      : '0';
    rateEl.textContent = `${rate}%`;
    
    // Color coding based on toxicity rate
    const rateNum = parseFloat(rate);
    if (rateNum > 30) {
      rateEl.className = 'stat-value danger';
    } else if (rateNum > 15) {
      rateEl.className = 'stat-value warning';
    } else {
      rateEl.className = 'stat-value';
    }
  }

  // Update label counts
  const labels = ['toxic', 'obscene', 'insult', 'threat', 'identity_hate', 'racism', 'severe_toxic'];
  labels.forEach(label => {
    const el = document.getElementById(`label-${label}`);
    if (el) {
      el.textContent = (stats.labelCounts[label] || 0).toString();
    }
  });
}

// Load settings from storage
async function loadSettings(): Promise<void> {
  try {
    const result = await chrome.storage.local.get(['settings']);
    const settings: Settings = result.settings || defaultSettings;
    
    const enabledToggle = document.getElementById('toggle-enabled') as HTMLInputElement;
    const placeholdersToggle = document.getElementById('toggle-placeholders') as HTMLInputElement;
    const highlightToggle = document.getElementById('toggle-highlight') as HTMLInputElement;
    
    if (enabledToggle) enabledToggle.checked = settings.enabled;
    if (placeholdersToggle) placeholdersToggle.checked = settings.showPlaceholders;
    if (highlightToggle) highlightToggle.checked = settings.highlightOnly;
  } catch (error) {
    console.error('Error loading settings:', error);
  }
}

// Save settings to storage
async function saveSettings(): Promise<void> {
  const enabledToggle = document.getElementById('toggle-enabled') as HTMLInputElement;
  const placeholdersToggle = document.getElementById('toggle-placeholders') as HTMLInputElement;
  const highlightToggle = document.getElementById('toggle-highlight') as HTMLInputElement;
  
  const settings: Settings = {
    enabled: enabledToggle?.checked ?? true,
    showPlaceholders: placeholdersToggle?.checked ?? true,
    highlightOnly: highlightToggle?.checked ?? false
  };
  
  try {
    await chrome.storage.local.set({ settings });
    // Notify content script of settings change
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab.id) {
      chrome.tabs.sendMessage(tab.id, { type: 'settingsChanged', settings });
    }
  } catch (error) {
    console.error('Error saving settings:', error);
  }
}

// Request rescan of the current page
async function requestRescan(): Promise<void> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab.id) {
      chrome.tabs.sendMessage(tab.id, { type: 'rescan' });
      // Update button text temporarily
      const btn = document.getElementById('refresh-btn');
      if (btn) {
        btn.textContent = '✓ Rescanning...';
        setTimeout(() => {
          btn.textContent = '🔄 Rescan Page';
          loadStats();
        }, 1500);
      }
    }
  } catch (error) {
    console.error('Error requesting rescan:', error);
  }
}

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadSettings();
  
  // Add event listeners for toggles
  const toggles = ['toggle-enabled', 'toggle-placeholders', 'toggle-highlight'];
  toggles.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('change', saveSettings);
    }
  });
  
  // Add event listener for refresh button
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', requestRescan);
  }
  
  // Listen for stats updates from content script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'statsUpdated') {
      updateStatsUI(message.stats);
    }
  });
});

// Request fresh stats when popup opens
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (tabs[0]?.id) {
    chrome.tabs.sendMessage(tabs[0].id, { type: 'getStats' });
  }
});
