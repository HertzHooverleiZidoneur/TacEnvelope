'use strict';

const results = {
  rlt: {name:'RLT', usbBefore:'72.5', usbAfter:'85', usbNote:'+12.5 percentage points', timeBefore:'19', timeAfter:'12', timeNote:'36.8% less time over successful trials', inkBefore:'3', inkAfter:'2', inkNote:'33.3% lower accumulated error', pastryBefore:'2.0', pastryAfter:'4.0', pastryNote:'100% higher combined score'},
  dsrl: {name:'DSRL', usbBefore:'87.5', usbAfter:'95', usbNote:'+7.5 percentage points', timeBefore:'20', timeAfter:'13', timeNote:'35.0% less time over successful trials', inkBefore:'5', inkAfter:'2', inkNote:'60.0% lower accumulated error', pastryBefore:'2.4', pastryAfter:'3.8', pastryNote:'58.3% higher combined score'}
};

document.querySelectorAll('[data-backend]').forEach(button => {
  button.addEventListener('click', () => {
    const selected = results[button.dataset.backend];
    if (!selected) return;
    document.querySelectorAll('[data-backend]').forEach(item => {
      item.setAttribute('aria-pressed', String(item === button));
    });
    for (const [key, value] of Object.entries(selected)) {
      const id = key.replace(/[A-Z]/g, letter => '-' + letter.toLowerCase());
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    }
    document.getElementById('usb-bar-before').style.width = selected.usbBefore + '%';
    document.getElementById('usb-bar-after').style.width = selected.usbAfter + '%';
    document.getElementById('backend-description').textContent = `Comparing ${selected.name} with ${selected.name} + TacEnvelope on the same task.`;
  });
});

const copyButton = document.getElementById('copy-citation');
copyButton.addEventListener('click', async () => {
  const text = document.getElementById('bibtex').textContent;
  const status = document.getElementById('copy-status');
  let copied = false;
  if (navigator.clipboard && window.isSecureContext) {
    try { await navigator.clipboard.writeText(text); copied = true; } catch (_) { /* Try a local-file compatible fallback. */ }
  }
  if (!copied) {
    const field = document.createElement('textarea');
    field.value = text;
    field.style.cssText = 'position:fixed;opacity:0;left:0;top:0';
    field.setAttribute('aria-hidden', 'true');
    document.body.appendChild(field);
    field.focus();
    field.select();
    try { copied = document.execCommand('copy'); } catch (_) { copied = false; }
    field.remove();
    copyButton.focus();
  }
  copyButton.textContent = copied ? 'Copied!' : 'Select BibTeX';
  status.textContent = copied ? 'BibTeX copied to clipboard.' : 'Select and copy the BibTeX text below.';
  if (!copied) {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById('bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
  }
  setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2500);
});
