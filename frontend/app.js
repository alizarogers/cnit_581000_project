/**
 * app.js: PunBot chat frontend (vanilla JS).
 * Keeps conversation history in the page and sends it with each /chat call,
 * so Claude can answer follow-up questions about the pun under discussion.
 */

let sessionId = crypto.randomUUID();
let history = [];

const chatEl     = document.getElementById('chat');
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('userInput');
const sendBtnEl  = document.getElementById('sendBtn');
const typingEl   = document.getElementById('typingIndicator');
const welcomeEl  = messagesEl.firstElementChild.cloneNode(true);

async function sendMessage(textOverride) {
  const text = (textOverride ?? inputEl.value).trim();
  if (!text) return;

  appendMessage('user', text);
  inputEl.value = '';
  autoResize();
  setBusy(true);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text, history }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);

    // Store what Claude actually saw (sentence + pipeline evidence) so follow-ups keep the context
    history.push({ role: 'user', content: data.sent_message || text });
    history.push({ role: 'assistant', content: data.answer });
    appendMessage('assistant', data.answer, false, data.pipeline);
  } catch (err) {
    appendMessage('assistant', `Sorry, something went wrong: ${err.message}`, true);
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

function appendMessage(role, text, isError = false, pipeline = null) {
  const wrapper = document.createElement('div');
  wrapper.className = `message message--${role}${isError ? ' message--error' : ''}`;
  const bubble = document.createElement('div');
  bubble.className = 'message__bubble';
  bubble.innerHTML = role === 'assistant' ? renderMarkdown(text) : `<p>${escapeHtml(text)}</p>`;
  if (pipeline) bubble.appendChild(renderPipeline(pipeline));
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

/** Collapsible summary of Stages 1-2 (spaCy + BERT + WordNet). */
function renderPipeline(p) {
  const details = document.createElement('details');
  details.className = 'pipeline';
  const best = p.best;
  const fmt = (x) => (typeof x === 'number' ? x.toFixed(2) : '');
  const sense = (label, s) => s ? `
    <li><strong>${label}</strong> <code>${escapeHtml(s.synset)}</code> ${escapeHtml(s.definition)}
      <span class="pipeline__meta">${s.trigger ? `triggered by <em>${escapeHtml(s.trigger)}</em> (${escapeHtml(s.trigger_relation || '')})` : 'no separate trigger word'}
      · activation ${fmt(s.activation)} (raw ${fmt(s.activation_raw)}) · support ${fmt(s.support)}</span></li>` : '';
  const others = p.candidates.filter((c) => c.sense_b && (!best || c.text !== best.text)).slice(0, 3)
    .map((c) => `${escapeHtml(c.text)} ${fmt(c.double_score)}`).join(', ');
  details.innerHTML = `
    <summary>Stages 1-2 (spaCy + BERT + WordNet): <strong>${escapeHtml(p.call)}</strong>${best ? `, pivot <em>${escapeHtml(best.text)}</em>` : ''}</summary>
    <div class="pipeline__body">
      ${best ? `
        <p>Pivot <strong>${escapeHtml(best.text)}</strong> (${escapeHtml(best.pos.toLowerCase())}, ${escapeHtml(best.dep)} of “${escapeHtml(best.head)}”).
           Double-sense score <strong>${fmt(best.double_score)}</strong> vs threshold ${fmt(p.threshold)}.
           ${best.senses_kept} of ${best.senses_total} WordNet senses fit its part of speech.</p>
        <ul>${sense('A', best.sense_a)}${sense('B', best.sense_b)}</ul>` : '<p>No word had two senses each supported by a different context word.</p>'}
      ${others ? `<p class="pipeline__meta">Other candidates: ${others}</p>` : ''}
      <p class="pipeline__meta">${escapeHtml(p.models.parser)} · ${escapeHtml(p.models.encoder)} · ${escapeHtml(p.models.lexicon)} · ${p.seconds}s</p>
    </div>`;
  return details;
}

/** Minimal, safe markdown: headings, bold, italics, inline code, lists, paragraphs. */
function renderMarkdown(src) {
  const inline = (s) => escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

  for (const raw of src.split('\n')) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^#{1,6}\s+(.*)/))) { closeList(); out.push(`<h3>${inline(m[1])}</h3>`); }
    else if ((m = line.match(/^\s*[-*]\s+(.*)/))) {
      if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if (line.trim() === '') { closeList(); }
    else { closeList(); out.push(`<p>${inline(line)}</p>`); }
  }
  closeList();
  return out.join('');
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function scrollToBottom() { chatEl.scrollTop = chatEl.scrollHeight; }

function setBusy(busy) {
  typingEl.classList.toggle('visible', busy);
  inputEl.disabled = busy;
  sendBtnEl.disabled = busy;
  document.querySelectorAll('.chip').forEach((c) => (c.disabled = busy));
  if (busy) scrollToBottom();
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
}

function newChat() {
  sessionId = crypto.randomUUID();
  history = [];
  messagesEl.replaceChildren(welcomeEl.cloneNode(true));
  inputEl.focus();
}

// Event wiring
sendBtnEl.addEventListener('click', () => sendMessage());
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
inputEl.addEventListener('input', autoResize);
messagesEl.addEventListener('click', (e) => {
  if (e.target.classList.contains('chip')) sendMessage(e.target.textContent);
});
document.getElementById('newChatBtn').addEventListener('click', newChat);

fetch('/health').then((r) => r.json()).then((d) => {
  document.getElementById('modelLabel').textContent =
    d.status !== 'ok' ? `Server not ready: ${d.status}`
      : d.mode === 'full' ? `Stages 1-2: spaCy + BERT + WordNet · Stage 3: ${d.model}`
      : `Answers generated by ${d.model} (pipeline ${d.pipeline})`;
}).catch(() => {});
