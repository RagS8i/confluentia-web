"use strict";

const state = {
  identity: { userId: "dr_demo", role: "clinician", patientId: "patient_demo" },
  caseId: null,
  caseData: null,
  intelligence: null,
  pollTimer: null,
  toastTimer: null,
  mediaRecorder: null,
  recordedChunks: [],
  recordedAudioBlob: null,
  recordTimerInterval: null,
  recordSeconds: 0,
  intakeJobTimer: null,
  patientRecorder: null,
  patientAudioChunks: [],
  patientRecordingFollowup: null,
  patientFollowupCache: {},
  patientCardCache: {},
  patientLanguageCaseId: null,
  patientAudioUrls: [],
};

const PATIENT_LANGUAGES = [
  { code: "en-IN", short: "en", name: "English" },
  { code: "bn-IN", short: "bn", name: "Bengali" },
  { code: "gu-IN", short: "gu", name: "Gujarati" },
  { code: "hi-IN", short: "hi", name: "Hindi" },
  { code: "kn-IN", short: "kn", name: "Kannada" },
  { code: "ml-IN", short: "ml", name: "Malayalam" },
  { code: "mr-IN", short: "mr", name: "Marathi" },
  { code: "od-IN", short: "or", name: "Odia" },
  { code: "pa-IN", short: "pa", name: "Punjabi" },
  { code: "ta-IN", short: "ta", name: "Tamil" },
  { code: "te-IN", short: "te", name: "Telugu" },
];

function isPatient() {
  return state.identity.role === "patient";
}

function shortLanguageFor(value) {
  if (!value) return "en";
  const raw = String(value).trim().toLowerCase().replace(/_/g, "-");
  const exact = PATIENT_LANGUAGES.find((language) => language.code.toLowerCase() === raw);
  if (exact) return exact.short;
  const primary = raw.split("-")[0];
  const matched = PATIENT_LANGUAGES.find(
    (language) => language.short === primary || language.code.toLowerCase().startsWith(primary + "-"),
  );
  return matched ? matched.short : primary;
}

function bcp47For(value) {
  if (!value) return null;
  const raw = String(value).trim().toLowerCase().replace(/_/g, "-");
  const exact = PATIENT_LANGUAGES.find((language) => language.code.toLowerCase() === raw);
  if (exact) return exact.code;
  const short = shortLanguageFor(raw);
  const matched = PATIENT_LANGUAGES.find((language) => language.short === short);
  return matched ? matched.code : null;
}

function applyRoleVisibility(patient = isPatient()) {
  document.body.classList.toggle("patient-mode", patient);
  const patientView = $("patient-view");
  if (patientView) patientView.hidden = !patient;
  const noteText = $("note-text");
  if (noteText) noteText.disabled = patient;
}

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function badge(value) {
  return `<span class="badge ${escapeHtml(value)}">${escapeHtml(value)}</span>`;
}

function loadIdentity() {
  try {
    const saved = JSON.parse(localStorage.getItem("careflow.identity") || "null");
    if (saved && saved.userId && saved.role) state.identity = saved;
  } catch (_) {}
  $("user-id").value = state.identity.userId;
  $("role").value = state.identity.role;
  $("patient-id").value = state.identity.patientId || "";
}

function identityHeaders() {
  const headers = {
    "X-User-Id": state.identity.userId,
    "X-Role": state.identity.role,
  };
  if (state.identity.role === "patient") {
    headers["X-Patient-Id"] = state.identity.patientId || state.identity.userId;
  }
  return headers;
}

function normalizeApiUrl(raw) {
  if (!raw) return "";
  let clean = raw.trim().replace(/\/+$/, "");
  if (clean.endsWith("/api")) {
    clean = clean.slice(0, -4).replace(/\/+$/, "");
  }
  return clean;
}

function getApiBaseUrl() {
  try {
    const urlParam = new URLSearchParams(window.location.search).get("api");
    if (urlParam) {
      const clean = normalizeApiUrl(urlParam);
      localStorage.setItem("careflow.apiUrl", clean);
      return clean;
    }
    const saved = localStorage.getItem("careflow.apiUrl");
    if (saved) return normalizeApiUrl(saved);
  } catch (_) {}
  return "";
}

function setApiBaseUrl(url) {
  const clean = normalizeApiUrl(url);
  try {
    if (clean) {
      localStorage.setItem("careflow.apiUrl", clean);
    } else {
      localStorage.removeItem("careflow.apiUrl");
    }
  } catch (_) {}
  const input = $("api-url");
  if (input) input.value = clean;
  checkApiHealth();
}

async function checkApiHealth() {
  const el = $("api-status-pill");
  if (!el) return;
  el.textContent = "connecting...";
  el.className = "badge PRIORITY_REVIEW";
  try {
    const base = getApiBaseUrl();
    const endpoint = base ? `${base}/api/health` : "/api/health";
    const res = await fetch(endpoint, {
      headers: {
        "ngrok-skip-browser-warning": "true",
      },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    if (text.trim().startsWith("<")) {
      throw new Error("Received HTML instead of JSON. Ensure your tunnel is pointing to port 8000.");
    }
    const data = JSON.parse(text);
    const model = data.provider && data.provider.model ? data.provider.model : "";
    el.textContent = model ? `online (${model})` : "online";
    el.className = "badge ROUTINE_REVIEW";
    el.title = `Connected to CareFlow backend (${base || "local origin"})`;
  } catch (err) {
    el.textContent = "offline";
    el.className = "badge EMERGENCY";
    el.title = `Cannot reach backend at ${getApiBaseUrl() || window.location.origin}: ${err.message}`;
  }
}

async function api(path, options = {}) {
  const opts = { ...options };
  const headers = {
    ...identityHeaders(),
    "ngrok-skip-browser-warning": "true",
    ...(options.headers || {}),
  };
  if (opts.body && !(opts.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  opts.headers = headers;
  const base = getApiBaseUrl();
  const endpoint = base ? `${base}/api${path}` : `/api${path}`;
  const response = await fetch(endpoint, opts);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { data = { detail: text }; }
  if (!response.ok) {
    let message = response.statusText;
    if (data && data.detail) {
      message = typeof data.detail === "string" ? data.detail : (data.detail.message || JSON.stringify(data.detail));
    }
    const error = new Error(message);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function setStatus(message, isError) {
  $("ingest-status").textContent = message || "";
  $("ingest-status").className = isError ? "error" : "hint";
  notify(message, isError);
}

function notify(message, isError) {
  if (!message) return;
  const toast = $("toast");
  toast.textContent = message;
  toast.className = isError ? "toast error" : "toast";
  toast.hidden = false;
  if (state.toastTimer) clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { toast.hidden = true; }, 4200);
}

function setAudioFromBase64(element, base64, mime) {
  if (!element) return;
  if (element.dataset.objectUrl) {
    try { URL.revokeObjectURL(element.dataset.objectUrl); } catch (_) {}
    delete element.dataset.objectUrl;
  }
  if (!base64) {
    element.removeAttribute("src");
    element.hidden = true;
    return;
  }
  try {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    const blob = new Blob([bytes], { type: mime || "audio/mpeg" });
    const url = URL.createObjectURL(blob);
    element.src = url;
    element.dataset.objectUrl = url;
    element.hidden = false;
  } catch (_) {
    element.removeAttribute("src");
    element.hidden = true;
  }
}

function setPatientVoiceStatus(message, isError = false) {
  const element = $("patient-card-status");
  if (!element) return;
  element.textContent = message || "";
  element.className = isError ? "error" : "hint";
}

function roleCanReview() {
  return [
    "clinician",
    "doctor",
    "nurse",
    "physician",
    "physician_assistant",
    "nurse_practitioner",
    "admin",
  ].includes(state.identity.role);
}

function saveIdentity() {
  state.identity = {
    userId: $("user-id").value.trim() || "user_demo",
    role: $("role").value,
    patientId: $("patient-id").value.trim(),
  };
  if (state.identity.role === "patient" && !state.identity.patientId) {
    state.identity.patientId = state.identity.userId;
  }
  localStorage.setItem("careflow.identity", JSON.stringify(state.identity));
  applyRoleVisibility();
  loadQueue();
  loadIntelligence();
  if (state.caseId) loadCase(state.caseId, false);
}

// ------------------------------------------------------------------- queue
async function loadQueue() {
  const container = $("queue");
  container.innerHTML = '<p class="hint">Loading…</p>';
  try {
    const data = await api("/cases");
    if (!data.items.length) {
      container.innerHTML = '<p class="hint">No cases yet. Create one to begin.</p>';
      return;
    }
    container.innerHTML = data.items.map((item) => {
      const patient = isPatient();
      const flags = [];
      if (!patient) {
        if (item.warning_count) flags.push(`${item.warning_count} warning(s)`);
        if (item.contradiction_count) flags.push(`${item.contradiction_count} discrepancy(ies)`);
        if (item.missing_info_count) flags.push(`${item.missing_info_count} gap(s)`);
        if (item.processing_failures) flags.push(`${item.processing_failures} failure(s)`);
        if (item.urgency_score != null) flags.unshift(`urgency ${item.urgency_score}/100`);
      }
      return `
        <div class="queue-item ${escapeHtml(item.workflow_status)} ${item.case_id === state.caseId ? "active" : ""}"
             data-case="${escapeHtml(item.case_id)}">
          <div class="qid">${escapeHtml(item.case_id)}</div>
          <div class="qmeta">
            ${patient ? badge(item.status) : `${badge(item.workflow_status)} ${badge(item.status)}`}
            ${!patient && item.overdue ? '<span class="badge EMERGENCY">overdue</span>' : ""}
          </div>
          <div class="qmeta">${patient
            ? `consent: ${escapeHtml(item.consent)}`
            : `${escapeHtml(item.source_modalities.join(", ") || "no sources")} · consent: ${escapeHtml(item.consent)}${item.awaiting_minutes != null ? ` · waiting ${item.awaiting_minutes} min` : ""}`}
          </div>
          ${patient ? "" : `<div class="qmeta">${escapeHtml(flags.join(" · ") || "no flags")}</div>`}
        </div>`;
    }).join("");
    container.querySelectorAll("[data-case]").forEach((element) => {
      element.addEventListener("click", () => loadCase(element.dataset.case));
    });
  } catch (error) {
    container.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  }
}

// ---------------------------------------------------------- intelligence
async function loadIntelligence() {
  const metrics = $("bottlenecks");
  const recommendations = $("recommendations");
  if (!roleCanReview() && state.identity.role !== "intake") {
    metrics.innerHTML = "";
    recommendations.innerHTML = '<p class="hint">Queue intelligence requires a staff role.</p>';
    return;
  }
  try {
    const data = await api("/queue/intelligence");
    state.intelligence = data;
    renderBottlenecks(data.bottlenecks);
    renderRecommendations(data.items);
    fillStaffing(data.staffing);
  } catch (error) {
    metrics.innerHTML = "";
    recommendations.innerHTML = `<p class="hint">${escapeHtml(error.message)}</p>`;
  }
}

function renderBottlenecks(summary) {
  const cells = [
    ["waiting", "Waiting", summary.total_waiting, false],
    ["overdue", "Overdue", summary.overdue, summary.overdue > 0],
    ["unassigned", "Unassigned", summary.unassigned, summary.unassigned > 0],
    ["emergency", "Emergency", summary.emergency, summary.emergency > 0],
    ["open", "Open escalations", summary.open_escalations, summary.open_escalations > 0],
    ["followups", "Open follow-ups", summary.pending_followups, false],
  ];
  $("bottlenecks").innerHTML = cells.map(([key, label, value, alert]) =>
    `<div class="metric ${alert ? "alert" : ""}" data-metric="${key}"><strong>${Number(value || 0)}</strong><span>${escapeHtml(label)}</span></div>`
  ).join("");
  const queueCount = $("queue-count");
  if (queueCount && summary.oldest_waiting_minutes != null) {
    queueCount.textContent = `Oldest active wait ${summary.oldest_waiting_minutes} min · no automatic reordering`;
  }
}

function fillStaffing(staffing) {
  if (!staffing) return;
  if (document.activeElement !== $("staff-clinicians")) $("staff-clinicians").value = staffing.clinicians_available || 0;
  if (document.activeElement !== $("staff-average")) {
    $("staff-average").value = staffing.average_review_minutes == null ? "" : staffing.average_review_minutes;
  }
  if (document.activeElement !== $("staff-sample")) $("staff-sample").value = staffing.sample_size || 0;
}

function renderRecommendations(items) {
  const container = $("recommendations");
  if (!items || !items.length) {
    container.innerHTML = '<p class="hint">No cases in the queue.</p>';
    return;
  }
  container.innerHTML = items.map((item) => {
    const estimate = item.wait_estimate && item.wait_estimate.available
      ? `${item.wait_estimate.minimum_minutes}–${item.wait_estimate.maximum_minutes} min estimate`
      : "wait estimate unavailable";
    return `
      <div class="recommendation" data-case="${escapeHtml(item.case_id)}">
        <div class="recommendation-head">
          <div>${badge(item.workflow_status)} <span class="badge">${escapeHtml(item.case_id)}</span></div>
          <span class="score">${item.urgency_score != null ? `urgency ${item.urgency_score}/100 · ` : ""}score ${Number(item.score || 0).toFixed(1)}</span>
        </div>
        <div class="action">${escapeHtml(item.recommended_action)}</div>
        <div class="qmeta small-text">${escapeHtml(estimate)}${item.overdue ? " · deadline passed" : ""}</div>
      </div>`;
  }).join("");
  container.querySelectorAll("[data-case]").forEach((element) => {
    element.addEventListener("click", () => loadCase(element.dataset.case));
  });
}

async function saveStaffing() {
  const averageRaw = $("staff-average").value.trim();
  try {
    const body = {
      clinicians_available: Number($("staff-clinicians").value || 0),
      intake_available: Number($("staff-clinicians").value || 0),
      average_review_minutes: averageRaw === "" ? null : Number(averageRaw),
      sample_size: Number($("staff-sample").value || 0),
      note: "updated from workstation",
    };
    await api("/operations/staffing", { method: "PUT", body });
    notify("Operational staffing inputs updated. Wait estimates are not guarantees.");
    await loadIntelligence();
    if (state.caseId) await loadCase(state.caseId);
  } catch (error) {
    notify(error.message, true);
  }
}

// -------------------------------------------------------------------- case
async function loadCase(caseId, updateUrl = true) {
  state.caseId = caseId;
  if (updateUrl && window.history && window.history.replaceState) {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("case_id", caseId);
      window.history.replaceState({}, "", url.toString());
    } catch (_) {}
  }
  try {
    const data = await api(`/cases/${encodeURIComponent(caseId)}`);
    state.caseData = data;
    $("welcome").hidden = true;
    $("case-view").hidden = false;
    renderCase(data);
    loadQueue();
  } catch (error) {
    alert(error.message);
  }
}

function copyCaseLink() {
  if (!state.caseId) return;
  const url = `${window.location.origin}${window.location.pathname}?case_id=${encodeURIComponent(state.caseId)}`;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(url).then(() => {
      notify(`Direct case link copied to clipboard: ${url}`);
    }).catch(() => {
      prompt("Direct case link:", url);
    });
  } else {
    prompt("Direct case link:", url);
  }
}

function renderCase(data) {
  const c = data.case;
  $("case-title").textContent = `Case ${c.case_id}`;
  $("case-meta").textContent = `Patient ${c.patient_id} · received ${new Date(c.received_at).toLocaleString()} · assigned ${c.assigned_reviewer || "unassigned"}`;

  applyRoleVisibility();
  if (isPatient()) {
    $("patient-status-strip").innerHTML = `
      ${badge(c.status)}
      <span class="badge">consent: ${c.consent_withdrawn_at ? "withdrawn" : c.consent_granted_at ? "granted" : "required"}</span>
    `;
    const languageSelect = $("patient-language");
    if (state.patientLanguageCaseId !== c.case_id) {
      state.patientLanguageCaseId = c.case_id;
      state.patientFollowupCache = {};
      state.patientCardCache = {};
      const hint = bcp47For(c.language_hint);
      if (hint) languageSelect.value = hint;
    }
    renderPatientCommunications(data.communications || []);
    renderPatientFollowups(data.followups || [], data.followup_answers || []);
    loadPatientCard(false).catch(() => {});
    localizePatientFollowups().catch(() => {});
    return;
  }

  const meta = c.meta || {};
  const deadline = (meta.review_deadlines_minutes || {})[c.workflow_status];
  const failures = (meta.processing_failures || []).length + (meta.stage_failures || []).length;
  const estimate = data.queue_estimate;
  const estimateText = estimate && estimate.available
    ? `operational wait ${estimate.minimum_minutes}–${estimate.maximum_minutes} min (${estimate.confidence})`
    : "wait estimate unavailable";
  $("status-strip").innerHTML = `
    ${badge(c.workflow_status)}
    ${badge(c.status)}
    <span class="badge">consent: ${c.consent_withdrawn_at ? "withdrawn" : c.consent_granted_at ? "granted" : "required"}</span>
    <span class="badge">rules: ${escapeHtml(meta.rule_config_id || "n/a")} v${escapeHtml(meta.rule_config_version || "n/a")}</span>
    <span class="badge">model: ${escapeHtml(meta.provider || "n/a")} / ${escapeHtml(meta.model || "n/a")}</span>
    <span class="badge">facts: ${meta.fact_count != null ? meta.fact_count : "0"}</span>
    <span class="badge">sources: ${meta.usable_source_count != null ? meta.usable_source_count : "0"}</span>
    <span class="badge">urgency: ${c.urgency ? `${c.urgency.score}/100 ${c.urgency.source}` : "not assessed"}</span>
    ${deadline != null ? `<span class="badge">review deadline: ${deadline} min</span>` : ""}
    <span class="badge">${escapeHtml(estimateText)}</span>
    ${failures ? `<span class="badge EMERGENCY">${failures} processing failure(s)</span>` : ""}
    ${meta.summary_stale ? '<span class="badge PRIORITY_REVIEW">summary stale after correction</span>' : ""}
    <span class="badge">${escapeHtml(meta.rule_config_status || "illustrative")}</span>
  `;
  const urgency = c.urgency;
  const canEditUrgency = roleCanReview() && !c.consent_withdrawn_at && !c.deleted_at;
  $("urgency-score").textContent = urgency ? `${urgency.score}/100` : "not assessed";
  $("urgency-meta").textContent = urgency
    ? `Source: ${urgency.source}${urgency.model ? " · model " + urgency.model : ""}` +
      `${urgency.assessed_at ? " · assessed " + new Date(urgency.assessed_at).toLocaleString() : ""}` +
      `${urgency.edited_by ? " · edited by " + urgency.edited_by : ""}` +
      `${urgency.edit_reason ? " · reason: " + urgency.edit_reason : ""}`
    : "Run processing to generate an advisory SLM assessment.";
  $("urgency-rationale").textContent = urgency ? urgency.rationale : "";
  $("urgency-input").value = urgency ? urgency.score : "";
  $("urgency-input").disabled = !canEditUrgency;
  $("urgency-reason").disabled = !canEditUrgency;
  $("save-urgency").disabled = !canEditUrgency;

  $("consent-btn").disabled = Boolean(c.consent_withdrawn_at || c.deleted_at);
  $("process-btn").disabled = Boolean(c.consent_withdrawn_at || c.deleted_at);
  $("assign-btn").disabled = !roleCanReview();
  $("draft-message").disabled = !roleCanReview();
  $("sync-followups").disabled = !roleCanReview();

  renderEscalations(data.escalations || []);
  renderCommunications(data.communications || []);
  renderFollowups(data.followups || [], data.followup_answers || []);
  renderSummary(data.summary);
  renderTimeline(data.summary ? data.summary.timeline : []);
  renderDiscrepancies(data.discrepancies);
  renderSources(data.sources || []);
  renderFacts(data.facts);
  renderWarnings(data.warnings);
  renderGaps(data.summary);
  const note = data.note || (data.summary && data.summary.draft_note);
  $("note-text").value = note ? note.text : "";
  $("note-status").textContent = note
    ? `Status: ${note.status} · version ${note.version} · ${note.approved_by ? "approved by " + note.approved_by : "not approved"}${note.edited_by ? " · edited by " + note.edited_by : ""}`
    : "No note yet. Run the pipeline to generate an AI draft.";
  $("approve-note").disabled = !note || note.status === "approved";
  loadAudit();
  updateIntakeGate();
}

function evidenceButton(ref) {
  const source = escapeHtml(ref.source_id || "");
  const start = ref.start_char == null ? "" : ref.start_char;
  const end = ref.end_char == null ? "" : ref.end_char;
  return `<button class="evidence-link" data-source="${source}" data-start="${start}" data-end="${end}">view source</button>`;
}

function refsButtons(refs) {
  if (!refs || !refs.length) return "";
  const seen = new Set();
  const buttons = [];
  refs.forEach((ref) => {
    const key = `${ref.source_id}:${ref.start_char}:${ref.end_char}`;
    if (seen.has(key)) return;
    seen.add(key);
    buttons.push(evidenceButton(ref));
  });
  return buttons.join(" ");
}

function renderSummary(summary) {
  const container = $("summary");
  if (!summary) {
    container.innerHTML = '<p class="hint">No summary yet. Grant consent, add a source, then run the pipeline.</p>';
    return;
  }
  const disclaimers = (summary.disclaimers || []).map((d) => `<li>${escapeHtml(d)}</li>`).join("");
  let html = `<p class="hint">Generated ${new Date(summary.generated_at).toLocaleString()} · model ${escapeHtml(summary.model || "n/a")} · evidence coverage ${Math.round((summary.evidence_coverage?.evidence_coverage_ratio || 0) * 100)}%</p>`;
  if (summary.presenting_complaint) {
    html += `<div class="section"><h3>${escapeHtml(summary.presenting_complaint.title)}</h3>
      <p>${escapeHtml(summary.presenting_complaint.items.join(" "))}</p>
      ${refsButtons(summary.presenting_complaint.source_refs)}</div>`;
  }
  (summary.sections || []).forEach((section) => {
    html += `<div class="section">
      <h3>${escapeHtml(section.title)} <span class="badge">${escapeHtml(section.epistemic_status)}</span></h3>
      <ul>${(section.items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      ${refsButtons(section.source_refs)}
    </div>`;
  });
  html += `<div class="section"><h3>Disclaimers</h3><ul>${disclaimers}</ul></div>`;
  container.innerHTML = html;
  container.querySelectorAll("[data-source]").forEach((button) => {
    button.addEventListener("click", () => openSource(button.dataset.source, button.dataset.start, button.dataset.end));
  });
}

function renderTimeline(timeline) {
  const container = $("timeline");
  if (!timeline || !timeline.length) {
    container.innerHTML = '<p class="hint">No normalized timeline entries. Missing time expressions are not invented.</p>';
    return;
  }
  container.innerHTML = timeline.map((entry) => `
    <div class="timeline-item">
      <strong>${escapeHtml(entry.label)}</strong>
      <div class="small-text">Source expression: ${escapeHtml(entry.raw_time || "not stated")}</div>
      <div class="small-text muted">Normalized: ${escapeHtml(entry.normalized_time || "not normalized")} · certainty: ${escapeHtml(entry.certainty)}
        ${entry.estimated_start_at ? " · estimated " + escapeHtml(new Date(entry.estimated_start_at).toLocaleString()) : ""}</div>
      ${entry.progression ? `<div class="small-text">Progression: ${escapeHtml(entry.progression)}</div>` : ""}
      ${refsButtons(entry.source_refs)}
    </div>`).join("");
  container.querySelectorAll("[data-source]").forEach((button) => {
    button.addEventListener("click", () => openSource(button.dataset.source, button.dataset.start, button.dataset.end));
  });
}

function renderDiscrepancies(items) {
  const container = $("discrepancies");
  if (!items || !items.length) {
    container.innerHTML = '<p class="hint">No candidate contradictions detected by the configured checks.</p>';
    return;
  }
  container.innerHTML = items.map((item) => `
    <div class="discrepancy">
      <div><span class="badge ${item.severity === "high" ? "EMERGENCY" : "PRIORITY_REVIEW"}">${escapeHtml(item.severity)}</span>
        <strong>${escapeHtml(item.category)}</strong></div>
      <p>${escapeHtml(item.summary)}</p>
      <p class="small-text muted">${escapeHtml(item.explanation)}</p>
      <div class="small-text">Claims: ${escapeHtml((item.fact_ids || []).join(", ") || "document-level")}</div>
      ${refsButtons(item.source_refs)}
    </div>`).join("");
  container.querySelectorAll("[data-source]").forEach((button) => {
    button.addEventListener("click", () => openSource(button.dataset.source, button.dataset.start, button.dataset.end));
  });
}

function renderSources(sources) {
  const container = $("sources-list");
  if (!container) return;
  const countSpan = $("sources-count");
  if (countSpan) countSpan.textContent = sources && sources.length ? `(${sources.length})` : "";
  if (!sources || !sources.length) {
    container.innerHTML = '<p class="hint">No sources ingested yet. Record audio, upload a medical report, or send a message on the intake panel to begin.</p>';
    return;
  }
  container.innerHTML = sources.map((source) => {
    const modalityBadge = badge(source.modality.toUpperCase());
    const langBadge = `<span class="badge">${escapeHtml(source.language || "unknown")}</span>`;
    const docLink = source.document_ref
      ? `<a href="/api/documents/${encodeURIComponent(source.document_ref)}?inline=true" target="_blank" rel="noopener" class="small-text" style="margin-left: 0.5rem; text-decoration: underline;">View Raw File ↗</a>`
      : "";
    const timing = source.audio_start != null
      ? `<span class="small-text muted"> · ${source.audio_start}s–${source.audio_end != null ? source.audio_end : "?"}s</span>`
      : (source.page != null ? `<span class="small-text muted"> · Page ${source.page}</span>` : "");
    const excerpt = source.original_text || "(empty transcript or text)";

    return `
      <div class="fact" style="margin-bottom: 0.65rem;">
        <div class="fact-head">
          <div>
            ${modalityBadge} ${langBadge} ${timing}
          </div>
          <div>
            ${docLink}
            <button class="btn secondary small" data-source-preview="${escapeHtml(source.source_id)}" style="margin-left: 0.4rem;">Inspect Source</button>
          </div>
        </div>
        <div class="quote" style="margin-top: 0.4rem;">${escapeHtml(excerpt)}</div>
      </div>
    `;
  }).join("");

  container.querySelectorAll("[data-source-preview]").forEach((button) => {
    button.addEventListener("click", () => openSource(button.dataset.sourcePreview));
  });
}

function renderFacts(facts) {
  const container = $("facts");
  if (!facts || !facts.length) {
    container.innerHTML = '<p class="hint">No structured facts. The SLM was not asked to treat empty input as a clinical record.</p>';
    return;
  }
  container.innerHTML = facts.map((fact) => {
    const sourceRefs = fact.source_refs || [];
    const refHtml = sourceRefs.length
      ? sourceRefs.map((ref) => evidenceButton(ref)).join(" ")
      : '<span class="error small-text">No source reference — unverified AI inference</span>';
    const temporal = fact.temporal && fact.temporal.raw_expression
      ? `<div class="small-text">Time: ${escapeHtml(fact.temporal.raw_expression)} → ${escapeHtml(fact.temporal.normalized_expression || "not normalized")}</div>`
      : "";
    return `
      <div class="fact">
        <div class="fact-head">
          <div><strong>${escapeHtml(fact.label)}</strong>
            ${fact.value != null ? `<span class="muted">: ${escapeHtml(fact.value)} ${escapeHtml(fact.unit || "")}</span>` : ""}</div>
          <div>${badge(fact.status)} <span class="badge">${escapeHtml(fact.assertion)}</span>
            <span class="badge">${escapeHtml(fact.verification_status)}</span></div>
        </div>
        <div class="small-text muted">${escapeHtml(fact.fact_type)} · ${escapeHtml(fact.language)} · confidence ${fact.confidence != null ? fact.confidence : "n/a"}
          ${fact.ai_suggestion ? " · AI inference" : ""}</div>
        <div class="quote">${escapeHtml(fact.original_text || "(empty quote)")}</div>
        ${temporal}
        ${fact.warnings && fact.warnings.length ? `<div class="small-text error">${escapeHtml(fact.warnings.join(", "))}</div>` : ""}
        <div class="small-text">${refHtml}</div>
        <div class="actions">
          <button class="btn secondary small" data-fact="${escapeHtml(fact.fact_id)}" data-action="verify">Verify</button>
          <button class="btn secondary small" data-fact="${escapeHtml(fact.fact_id)}" data-action="correct">Correct</button>
          <button class="btn secondary small" data-fact="${escapeHtml(fact.fact_id)}" data-action="dispute">Dispute</button>
          <button class="btn danger small" data-fact="${escapeHtml(fact.fact_id)}" data-action="reject">Reject</button>
        </div>
      </div>`;
  }).join("");
  container.querySelectorAll("[data-source]").forEach((button) => {
    button.addEventListener("click", () => openSource(button.dataset.source, button.dataset.start, button.dataset.end));
  });
  container.querySelectorAll("[data-fact]").forEach((button) => {
    button.addEventListener("click", () => reviewFact(button.dataset.fact, button.dataset.action));
  });
}

function renderWarnings(warnings) {
  const container = $("warnings");
  if (!warnings || !warnings.length) {
    container.innerHTML = '<p class="hint">No warnings recorded. Absence of a warning is not clinical clearance.</p>';
    return;
  }
  container.innerHTML = warnings.map((warning) => `
    <div class="warning ${warning.kind === "rule" ? "" : "component"}">
      <div><strong>${escapeHtml(warning.label)}</strong>
        ${warning.outcome ? badge(warning.outcome) : ""}
        <span class="badge">${escapeHtml(warning.code)}</span></div>
      <p class="small-text">${escapeHtml(warning.explanation)}</p>
      ${warning.rule_id ? `<div class="small-text">Rule ${escapeHtml(warning.rule_id)} v${escapeHtml(warning.rule_version || "")} · ${escapeHtml(warning.status)}</div>` : ""}
      <div class="small-text">Evidence claims: ${escapeHtml((warning.evidence_fact_ids || []).join(", ") || "document-level")}</div>
      ${refsButtons(warning.source_refs)}
      ${warning.status === "active" ? `<button class="btn secondary small" data-warning="${escapeHtml(warning.warning_id)}">Acknowledge</button>` : ""}
    </div>`).join("");
  container.querySelectorAll("[data-source]").forEach((button) => {
    button.addEventListener("click", () => openSource(button.dataset.source, button.dataset.start, button.dataset.end));
  });
  container.querySelectorAll("[data-warning]").forEach((button) => {
    button.addEventListener("click", () => acknowledgeWarning(button.dataset.warning));
  });
}

function renderGaps(summary) {
  const gaps = summary ? summary.gaps || [] : [];
  const questions = summary ? summary.follow_up_questions || [] : [];
  $("gaps").innerHTML = gaps.length
    ? `<ul>${gaps.map((gap) => `<li><strong>${escapeHtml(gap.code)}</strong>: ${escapeHtml(gap.description)}</li>`).join("")}</ul>`
    : '<p class="hint">No missing-information gaps identified by the configured checks.</p>';
  $("questions").innerHTML = questions.length
    ? `<h3>Suggested questions</h3><ul>${questions.map((q) => `<li>${escapeHtml(q.text)} <span class="badge">${escapeHtml(q.language)}</span><span class="small-text muted">${escapeHtml(q.rationale)}</span></li>`).join("")}</ul>`
    : '<p class="hint">No follow-up questions (emergency escalation suppresses non-essential questions).</p>';
}

function renderEscalations(items) {
  const panel = $("escalation-panel");
  const container = $("escalations");
  if (!items.length) {
    panel.hidden = true;
    container.innerHTML = "";
    return;
  }
  panel.hidden = false;
  container.innerHTML = items.map((item) => `
    <div class="escalation">
      <div>
        <div>${badge(item.status)} <span class="badge">${escapeHtml(item.severity)}</span></div>
        <strong>${escapeHtml(item.trigger_label)}</strong>
        <div class="small-text muted">${escapeHtml(item.trigger_code)} · created ${escapeHtml(new Date(item.created_at).toLocaleString())}
          · attempts ${Number(item.attempts || 0)}${item.last_error ? " · " + escapeHtml(item.last_error) : ""}</div>
      </div>
      <div class="actions">
        ${item.status !== "acknowledged" ? `<button class="btn small" data-escalation="${escapeHtml(item.escalation_id)}" data-escalation-action="acknowledge">Acknowledge</button>` : ""}
        ${["pending", "failed"].includes(item.status) ? `<button class="btn secondary small" data-escalation="${escapeHtml(item.escalation_id)}" data-escalation-action="retry">Retry delivery</button>` : ""}
      </div>
    </div>`).join("");
  container.querySelectorAll("[data-escalation]").forEach((button) => {
    button.addEventListener("click", () => escalationAction(button.dataset.escalation, button.dataset.escalationAction));
  });
}

function renderCommunications(items) {
  const container = $("communications");
  if (!items.length) {
    container.innerHTML = '<p class="hint">No draft or sent patient messages. Use “Draft” to create a curated message for staff approval.</p>';
    return;
  }
  container.innerHTML = items.map((item) => `
    <article class="message-card ${escapeHtml(item.status)}">
      <div class="message-head">
        <div>
          ${badge(item.status)}
          <span class="badge">${escapeHtml(item.intent)}</span>
          <span class="badge">${escapeHtml(item.language)}</span>
          ${item.approved_by ? `<span class="badge">approved by ${escapeHtml(item.approved_by)}</span>` : ""}
          ${item.delivery ? `<span class="badge">${escapeHtml(item.delivery)}</span>` : ""}
        </div>
        <span class="small-text muted">${escapeHtml(new Date(item.created_at).toLocaleString())}</span>
      </div>
      <div class="message-body">${escapeHtml(item.body)}</div>
      ${item.last_error ? `<div class="error small-text">${escapeHtml(item.last_error)}</div>` : ""}
      <div class="message-actions">
        ${["draft", "failed"].includes(item.status) ? `<button class="btn small" data-message="${escapeHtml(item.message_id)}" data-message-action="approve">Review &amp; approve</button>` : ""}
        ${item.status === "approved" ? `<button class="btn small" data-message="${escapeHtml(item.message_id)}" data-message-action="send">Send</button>` : ""}
        ${!["sent", "dismissed"].includes(item.status) ? `<button class="btn danger small" data-message="${escapeHtml(item.message_id)}" data-message-action="dismiss">Dismiss</button>` : ""}
      </div>
    </article>`).join("");
  container.querySelectorAll("[data-message]").forEach((button) => {
    button.addEventListener("click", () => communicationAction(button.dataset.message, button.dataset.messageAction));
  });
}

function renderFollowups(items, answers) {
  const container = $("followups");
  if (!items.length) {
    const isEmergency = state.caseData?.case?.workflow_status === "EMERGENCY";
    container.innerHTML = isEmergency
      ? '<p class="hint" style="color: var(--danger);">Routine follow-up questions are suppressed because an EMERGENCY workflow status is active. Immediate escalation takes priority.</p>'
      : '<p class="hint">No follow-up questions yet. Run the pipeline, then sync from the summary.</p>';
    $("followup-answers").innerHTML = "";
    return;
  }
  container.innerHTML = items.map((item) => `
    <article class="followup-card">
      <div class="followup-head">
        <div>${badge(item.status)} <span class="badge">${escapeHtml(item.language)}</span>
          ${item.addresses && item.addresses.length ? `<span class="badge">${escapeHtml(item.addresses.join(", "))}</span>` : ""}</div>
        <span class="small-text muted">${escapeHtml(item.origin || "")}</span>
      </div>
      <div class="followup-text">${escapeHtml(item.text)}</div>
      <div class="small-text muted">${escapeHtml(item.rationale || "")}</div>
      ${item.last_error ? `<div class="error small-text">${escapeHtml(item.last_error)}</div>` : ""}
      <div class="followup-actions">
        ${item.status === "suggested" ? `<button class="btn small" data-followup="${escapeHtml(item.followup_id)}" data-followup-action="approve">Review &amp; approve</button>` : ""}
        ${item.status === "approved" ? `<button class="btn small" data-followup="${escapeHtml(item.followup_id)}" data-followup-action="send">Send</button>` : ""}
        ${item.status === "sent" ? `<button class="btn secondary small" data-followup="${escapeHtml(item.followup_id)}" data-followup-action="answer">Record answer</button>` : ""}
        ${!["answered", "dismissed"].includes(item.status) ? `<button class="btn danger small" data-followup="${escapeHtml(item.followup_id)}" data-followup-action="dismiss">Dismiss</button>` : ""}
      </div>
    </article>`).join("");
  container.querySelectorAll("[data-followup]").forEach((button) => {
    button.addEventListener("click", () => followupAction(button.dataset.followup, button.dataset.followupAction));
  });
  $("followup-answers").innerHTML = answers.length
    ? `<h3>Recorded answers</h3>${answers.map((answer) => `
        <div class="answer"><strong>${escapeHtml(answer.answered_by)} (${escapeHtml(answer.role)})</strong>
          <span class="small-text muted">${escapeHtml(new Date(answer.created_at).toLocaleString())}</span>
          <div>${escapeHtml(answer.text)}</div>
        </div>`).join("")}`
    : "";
}

function renderPatientCommunications(items) {
  const container = $("patient-communications");
  if (!container) return;
  if (!items.length) {
    container.innerHTML = '<p class="hint">No messages from your care team yet.</p>';
    return;
  }
  container.innerHTML = items.map((item) => `
    <article class="message-card">
      <div class="message-head">
        <div>${badge(item.status)} <span class="badge">${escapeHtml(item.language)}</span></div>
        <span class="small-text muted">${escapeHtml(new Date(item.created_at || Date.now()).toLocaleString())}</span>
      </div>
      <div class="message-body">${escapeHtml(item.body)}</div>
    </article>`).join("");
}

function renderPatientFollowups(items, answers) {
  const container = $("patient-followups");
  if (!container) return;
  const visible = items.filter((item) => ["sent", "answered"].includes(item.status));
  if (!visible.length) {
    container.innerHTML = '<p class="hint">No follow-up questions right now. If your care team sends one, it will appear here.</p>';
    return;
  }
  container.innerHTML = visible.map((item) => {
    const answer = [...answers].reverse().find((entry) => entry.followup_id === item.followup_id);
    const answered = item.status === "answered";
    return `
      <article class="followup-card patient-followup-card" data-patient-card="${escapeHtml(item.followup_id)}">
        <div class="followup-head">
          <div>${badge(item.status)} <span class="badge">${escapeHtml(item.language)}</span></div>
          <span class="small-text muted">Your care team</span>
        </div>
        <div class="followup-text" data-patient-followup-text="${escapeHtml(item.followup_id)}"
             data-patient-followup-original="${escapeHtml(item.text)}">${escapeHtml(item.text)}</div>
        <div class="followup-actions">
          <button class="btn secondary small" data-patient-play="${escapeHtml(item.followup_id)}">🔊 Listen</button>
        </div>
        ${answer ? `<div class="answer"><strong>Your answer</strong>
          <div>${escapeHtml(answer.text)}</div>
          <span class="small-text muted">${answer.metadata && answer.metadata.answer_modality === "audio" ? "Voice answer · " : ""}${escapeHtml(new Date(answer.created_at).toLocaleString())}</span>
        </div>` : ""}
        ${!answered ? `<div class="patient-answer">
          <textarea data-patient-answer-text="${escapeHtml(item.followup_id)}" placeholder="Type your answer in your own words..."></textarea>
          <div class="row compact">
            <button class="btn small" data-patient-send-text="${escapeHtml(item.followup_id)}">Send text answer</button>
            <button class="btn secondary small" data-patient-record="${escapeHtml(item.followup_id)}">● Record voice answer</button>
            <button class="btn danger small" data-patient-stop="${escapeHtml(item.followup_id)}" disabled>Stop &amp; send</button>
          </div>
          <div class="row compact">
            <input type="file" data-patient-voice-file="${escapeHtml(item.followup_id)}" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm" />
            <button class="btn secondary small" data-patient-send-file="${escapeHtml(item.followup_id)}">Upload voice answer</button>
          </div>
          <p class="hint patient-answer-status" data-patient-answer-status="${escapeHtml(item.followup_id)}"></p>
        </div>` : ""}
      </article>`;
  }).join("");
}

async function loadPatientCard(includeAudio = false, autoplay = false) {
  if (!isPatient() || !state.caseId) return;
  const language = $("patient-language").value || "en-IN";
  const loadButton = $("patient-card-load");
  const listenButton = $("patient-card-listen");
  loadButton.disabled = true;
  listenButton.disabled = true;
  setPatientVoiceStatus(includeAudio ? "Preparing your summary and voice playback..." : "Preparing your summary...");
  try {
    const card = await api(`/cases/${encodeURIComponent(state.caseId)}/patient-voice/context-card`, {
      method: "POST",
      body: { language, include_audio: Boolean(includeAudio) },
    });
    $("patient-card-text").textContent = card.text || "";
    const audio = $("patient-card-audio");
    if (includeAudio && card.audio_base64) {
      setAudioFromBase64(audio, card.audio_base64, card.audio_mime);
      if (autoplay) {
        try { await audio.play(); } catch (_) {}
      }
      setPatientVoiceStatus(`Summary in ${card.language_name}. You can play it with the audio controls.`);
    } else {
      setAudioFromBase64(audio, null);
      const note = includeAudio && card.audio_error
        ? ` ${card.audio_error}`
        : "";
      setPatientVoiceStatus(`Summary in ${card.language_name}.${note}`);
    }
    state.patientCardCache[language] = card;
  } catch (error) {
    setPatientVoiceStatus(error.message, true);
  } finally {
    loadButton.disabled = false;
    listenButton.disabled = false;
  }
}

async function localizePatientFollowups() {
  if (!isPatient() || !state.caseId) return;
  const language = $("patient-language").value || "en-IN";
  const nodes = Array.from(document.querySelectorAll("[data-patient-followup-text]"));
  await Promise.all(nodes.map(async (node) => {
    const followupId = node.dataset.patientFollowupText;
    const key = `${language}:${followupId}`;
    if (state.patientFollowupCache[key]) {
      node.textContent = state.patientFollowupCache[key];
      return;
    }
    try {
      const result = await api(`/cases/${encodeURIComponent(state.caseId)}/patient-voice/followups/${encodeURIComponent(followupId)}`, {
        method: "POST",
        body: { language, include_audio: false },
      });
      state.patientFollowupCache[key] = result.text;
      node.textContent = result.text;
    } catch (_) {
      node.textContent = node.dataset.patientFollowupOriginal || node.textContent;
    }
  }));
}

async function playPatientFollowup(followupId) {
  if (!state.caseId) return;
  const language = $("patient-language").value || "en-IN";
  const button = document.querySelector(`[data-patient-play="${CSS.escape(followupId)}"]`);
  if (button) button.disabled = true;
  try {
    const result = await api(`/cases/${encodeURIComponent(state.caseId)}/patient-voice/followups/${encodeURIComponent(followupId)}`, {
      method: "POST",
      body: { language, include_audio: true },
    });
    const textNode = document.querySelector(`[data-patient-followup-text="${CSS.escape(followupId)}"]`);
    if (textNode && result.text) {
      textNode.textContent = result.text;
      state.patientFollowupCache[`${language}:${followupId}`] = result.text;
    }
    const audio = $("patient-followup-audio");
    if (result.audio_base64) {
      setAudioFromBase64(audio, result.audio_base64, result.audio_mime);
      try { await audio.play(); } catch (_) {}
    } else {
      notify(result.audio_error || "Voice playback is not available for this question.", true);
    }
  } catch (error) {
    notify(error.message, true);
  } finally {
    if (button) button.disabled = false;
  }
}

function setPatientAnswerStatus(followupId, message, isError = false) {
  const element = document.querySelector(`[data-patient-answer-status="${CSS.escape(followupId)}"]`);
  if (!element) return;
  element.textContent = message || "";
  element.className = isError ? "error small-text" : "hint";
}

async function submitPatientTextAnswer(followupId) {
  if (!state.caseId) return;
  const input = document.querySelector(`[data-patient-answer-text="${CSS.escape(followupId)}"]`);
  const text = input ? input.value.trim() : "";
  if (!text) {
    setPatientAnswerStatus(followupId, "Please type an answer first.", true);
    return;
  }
  const language = shortLanguageFor($("patient-language").value || "en-IN");
  const button = document.querySelector(`[data-patient-send-text="${CSS.escape(followupId)}"]`);
  if (button) button.disabled = true;
  setPatientAnswerStatus(followupId, "Saving your answer...");
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/followups/${encodeURIComponent(followupId)}/answers`, {
      method: "POST",
      body: { text, language, reprocess: true },
    });
    notify("Your answer was saved.");
    await loadCase(state.caseId);
  } catch (error) {
    setPatientAnswerStatus(followupId, error.message, true);
    if (button) button.disabled = false;
  }
}

async function submitPatientAudioAnswer(followupId, file) {
  if (!state.caseId || !file) return;
  const form = new FormData();
  form.append("file", file, file.name || "voice_answer.webm");
  setPatientAnswerStatus(followupId, "Uploading and transcribing your voice answer...");
  try {
    const result = await api(`/cases/${encodeURIComponent(state.caseId)}/followups/${encodeURIComponent(followupId)}/answers/audio`, {
      method: "POST",
      body: form,
    });
    const detected = bcp47For(result.detected_language);
    if (detected) {
      $("patient-language").value = detected;
      state.patientFollowupCache = {};
      state.patientCardCache = {};
    }
    notify(result.detected_language
      ? `Voice answer saved. Detected ${result.detected_language}.`
      : "Voice answer saved.");
    await loadCase(state.caseId);
  } catch (error) {
    setPatientAnswerStatus(followupId, error.message, true);
  }
}

async function startPatientRecording(followupId, button) {
  if (state.patientRecorder && state.patientRecorder.state === "recording") {
    notify("Please stop the current recording first.", true);
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    notify("Audio recording is not supported in this browser. Please upload an audio file instead.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.patientAudioChunks = [];
    state.patientRecordingFollowup = followupId;
    const recorder = new MediaRecorder(stream);
    state.patientRecorder = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) state.patientAudioChunks.push(event.data);
    };
    recorder.onstop = () => {
      const mimeType = recorder.mimeType || "audio/webm";
      const blob = new Blob(state.patientAudioChunks, { type: mimeType });
      stream.getTracks().forEach((track) => track.stop());
      state.patientRecorder = null;
      state.patientRecordingFollowup = null;
      const recordButton = document.querySelector(`[data-patient-record="${CSS.escape(followupId)}"]`);
      const stopButton = document.querySelector(`[data-patient-stop="${CSS.escape(followupId)}"]`);
      if (recordButton) {
        recordButton.classList.remove("patient-record-active");
        recordButton.textContent = "● Record voice answer";
      }
      if (stopButton) stopButton.disabled = true;
      if (blob.size) {
        const extension = mimeType.includes("ogg") ? "ogg" : (mimeType.includes("mp4") ? "m4a" : "webm");
        const file = new File([blob], `voice_answer.${extension}`, { type: mimeType });
        submitPatientAudioAnswer(followupId, file);
      }
    };
    recorder.start();
    if (button) {
      button.classList.add("patient-record-active");
      button.textContent = "Recording...";
    }
    const stopButton = document.querySelector(`[data-patient-stop="${CSS.escape(followupId)}"]`);
    if (stopButton) stopButton.disabled = false;
    setPatientAnswerStatus(followupId, "Recording. Speak in any supported language, then stop and send.");
  } catch (error) {
    notify(`Could not access microphone: ${error.message}`, true);
  }
}

function stopPatientRecording(followupId) {
  if (
    state.patientRecorder
    && state.patientRecorder.state === "recording"
    && state.patientRecordingFollowup === followupId
  ) {
    state.patientRecorder.stop();
  }
}

async function detectPatientLanguage() {
  if (!state.caseId) return;
  const input = $("patient-detect-file");
  const file = input && input.files ? input.files[0] : null;
  if (!file) {
    notify("Choose a short voice recording first.", true);
    return;
  }
  const button = $("patient-detect-language");
  button.disabled = true;
  setPatientVoiceStatus("Listening to the voice sample...");
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const result = await api(`/cases/${encodeURIComponent(state.caseId)}/patient-voice/detect-language`, {
      method: "POST",
      body: form,
    });
    const detected = bcp47For(result.language_code);
    if (!detected) {
      setPatientVoiceStatus(
        `Detected ${result.language_name || result.language_code}, but voice playback is not available for that language. Please choose one of the listed languages.`,
        true,
      );
      return;
    }
    $("patient-language").value = detected;
    state.patientFollowupCache = {};
    state.patientCardCache = {};
    notify(`Detected ${result.language_name || detected}.`);
    await loadPatientCard(false);
    await localizePatientFollowups();
  } catch (error) {
    setPatientVoiceStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
}


async function loadAudit() {
  const container = $("audit");
  if (!roleCanReview()) {
    container.innerHTML = '<p class="hint">Audit trail is available to clinician and admin roles.</p>';
    return;
  }
  container.innerHTML = '<p class="hint">Loading.</p>';
  try {
    const data = await api(`/cases/${encodeURIComponent(state.caseId)}/audit`);
    container.innerHTML = data.items.length
      ? data.items.map((entry) => `<div>${escapeHtml(new Date(entry.created_at).toLocaleString())} · ${escapeHtml(entry.actor)} (${escapeHtml(entry.role)}) · ${escapeHtml(entry.action)}</div>`).join("")
      : '<p class="hint">No audit records.</p>';
  } catch (error) {
    container.innerHTML = `<p class="hint">Audit requires clinician/admin role. ${escapeHtml(error.message)}</p>`;
  }
}

// --------------------------------------------------------------- operations
async function createCase() {
  let patientId = state.identity.role === "patient" ? state.identity.patientId : "";
  if (state.identity.role !== "patient") {
    patientId = window.prompt("Synthetic patient ID:", "patient_demo");
    if (!patientId) return;
  }
  try {
    const created = await api("/cases", { method: "POST", body: { patient_id: patientId } });
    await loadCase(created.case_id);
  } catch (error) { alert(error.message); }
}

async function grantConsent() {
  if (!state.caseId) return;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/consent`, { method: "POST", body: { scope: "careflow_processing_and_review" } });
    setStatus("Consent granted.");
    await loadCase(state.caseId);
  } catch (error) { setStatus(error.message, true); }
}

// ------------------------------------------------------------------- intake
function switchIntakeTab(tabName) {
  document.querySelectorAll(".intake-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  $("tab-audio").hidden = tabName !== "audio";
  $("tab-document").hidden = tabName !== "document";
  $("tab-message").hidden = tabName !== "message";
}

function updateIntakeGate() {
  const c = state.caseData?.case;
  const hasConsent = Boolean(state.caseId && c && c.consent_granted_at && !c.consent_withdrawn_at && !c.deleted_at);
  const gate = $("intake-gate");
  if (gate) gate.hidden = hasConsent;
  const buttons = [
    $("record-btn"),
    $("submit-audio-btn"),
    $("submit-doc-btn"),
    $("submit-message-btn"),
    $("upload-btn"),
  ];
  buttons.forEach((btn) => {
    if (btn) btn.disabled = !hasConsent;
  });
}

async function toggleRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    notify("Audio recording is not supported in this browser.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.recordedChunks = [];
    state.mediaRecorder = new MediaRecorder(stream);
    state.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) state.recordedChunks.push(e.data);
    };
    state.mediaRecorder.onstop = () => {
      const mimeType = state.mediaRecorder.mimeType || "audio/webm";
      state.recordedAudioBlob = new Blob(state.recordedChunks, { type: mimeType });
      const preview = $("record-preview");
      if (preview) {
        preview.src = URL.createObjectURL(state.recordedAudioBlob);
        preview.hidden = false;
      }
      stream.getTracks().forEach((track) => track.stop());
      $("record-btn").textContent = "● Record";
      $("stop-record-btn").disabled = true;
      clearInterval(state.recordTimerInterval);
      state.recordTimerInterval = null;
      $("record-timer").hidden = true;
    };
    state.mediaRecorder.start();
    $("record-btn").textContent = "Recording...";
    $("stop-record-btn").disabled = false;
    state.recordSeconds = 0;
    $("record-timer").textContent = "00:00";
    $("record-timer").hidden = false;
    state.recordTimerInterval = setInterval(() => {
      state.recordSeconds++;
      const mins = String(Math.floor(state.recordSeconds / 60)).padStart(2, "0");
      const secs = String(state.recordSeconds % 60).padStart(2, "0");
      $("record-timer").textContent = `${mins}:${secs}`;
    }, 1000);
  } catch (err) {
    notify(`Could not access microphone: ${err.message}`, true);
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    state.mediaRecorder.stop();
  }
}

async function submitAudioIntake() {
  if (!state.caseId) { notify("Select a case first.", true); return; }
  let audioFile = null;
  const fileInput = $("audio-file");
  if (fileInput && fileInput.files && fileInput.files[0]) {
    audioFile = fileInput.files[0];
  } else if (state.recordedAudioBlob) {
    const ext = state.recordedAudioBlob.type.includes("wav") ? "wav" : (state.recordedAudioBlob.type.includes("ogg") ? "ogg" : "webm");
    audioFile = new File([state.recordedAudioBlob], `recorded_intake.${ext}`, { type: state.recordedAudioBlob.type });
  }
  if (!audioFile) {
    notify("Please record audio or select an audio file first.", true);
    return;
  }

  const autoProcess = $("audio-auto-process") ? $("audio-auto-process").checked : true;
  const targetLang = $("audio-target-lang") ? $("audio-target-lang").value : "en-IN";

  const form = new FormData();
  form.append("file", audioFile);

  $("submit-audio-btn").disabled = true;
  try {
    const res = await api(`/cases/${encodeURIComponent(state.caseId)}/intake/audio?auto_process=${autoProcess}&target_language=${encodeURIComponent(targetLang)}`, {
      method: "POST",
      body: form,
    });
    notify(`Audio uploaded. Starting transcription job ${res.job_id}...`);
    if (fileInput) fileInput.value = "";
    state.recordedAudioBlob = null;
    const preview = $("record-preview");
    if (preview) { preview.src = ""; preview.hidden = true; }
    startIntakeJobPolling(res.job_id, "audio", autoProcess);
  } catch (err) {
    notify(err.message, true);
    $("submit-audio-btn").disabled = false;
  }
}

async function submitDocumentIntake() {
  if (!state.caseId) { notify("Select a case first.", true); return; }
  const fileInput = $("doc-file");
  const docFile = fileInput && fileInput.files ? fileInput.files[0] : null;
  if (!docFile) {
    notify("Please select a document or image file first.", true);
    return;
  }

  const autoProcess = $("doc-auto-process") ? $("doc-auto-process").checked : true;
  const form = new FormData();
  form.append("file", docFile);

  $("submit-doc-btn").disabled = true;
  try {
    const res = await api(`/cases/${encodeURIComponent(state.caseId)}/intake/document?auto_process=${autoProcess}`, {
      method: "POST",
      body: form,
    });
    notify(`Document uploaded. Starting OCR job ${res.job_id}...`);
    if (fileInput) fileInput.value = "";
    startIntakeJobPolling(res.job_id, "document", autoProcess);
  } catch (err) {
    notify(err.message, true);
    $("submit-doc-btn").disabled = false;
  }
}

async function submitMessageIntake() {
  if (!state.caseId) { notify("Select a case first.", true); return; }
  const text = $("typed-message-text").value.trim();
  if (!text) { notify("Please enter a patient message first.", true); return; }
  const language = $("typed-message-lang").value;

  $("submit-message-btn").disabled = true;
  try {
    const res = await api(`/cases/${encodeURIComponent(state.caseId)}/sources/message`, {
      method: "POST",
      body: { text, language },
    });
    notify(`Message ingested (${res.inserted.length} source created).`);
    $("typed-message-text").value = "";
    await loadCase(state.caseId);
    await loadQueue();
    await loadIntelligence();
  } catch (err) {
    notify(err.message, true);
  } finally {
    $("submit-message-btn").disabled = false;
  }
}

function startIntakeJobPolling(jobId, kind, autoProcess) {
  const panel = $("intake-job-panel");
  const stageEl = $("intake-job-stage");
  const timerEl = $("intake-job-timer");
  const msgEl = $("intake-job-msg");
  const cpuWarning = $("intake-cpu-warning");

  if (panel) panel.hidden = false;
  if (cpuWarning) cpuWarning.hidden = (kind !== "document");

  let elapsed = 0;
  if (state.intakeJobTimer) clearInterval(state.intakeJobTimer);

  state.intakeJobTimer = setInterval(async () => {
    elapsed += 1.5;
    if (timerEl) timerEl.textContent = `${Math.round(elapsed)}s`;

    try {
      const job = await api(`/cases/${encodeURIComponent(state.caseId)}/intake/jobs/${encodeURIComponent(jobId)}`);
      if (stageEl) {
        stageEl.textContent = job.stage;
        stageEl.className = `badge ${job.status === "failed" ? "EMERGENCY" : (job.status === "complete" ? "ROUTINE_REVIEW" : "PRIORITY_REVIEW")}`;
      }
      if (msgEl) {
        if (job.stage === "transcribing") msgEl.textContent = "Transcribing audio via Sarvam STT...";
        else if (job.stage === "reading") msgEl.textContent = "Reading layout and text via GLM-OCR...";
        else if (job.stage === "organizing") msgEl.textContent = "Filtering clinically relevant text...";
        else if (job.stage === "ingesting") msgEl.textContent = "Persisting normalized sources and metadata...";
        else if (job.stage === "processing") msgEl.textContent = "Submitting case to clinical pipeline...";
        else if (job.status === "complete") {
          const timings = job.result?.timings || {};
          const timingText = timings.ocr_seconds != null
            ? ` · OCR ${timings.ocr_seconds}s, organizer ${timings.organizer_seconds ?? "?"}s`
            : "";
          msgEl.textContent = `Completed! ${job.result?.inserted || 0} source(s) inserted.${timingText}`;
        }
        else if (job.status === "failed") {
          const isNonMedical = (job.error || "").toLowerCase().includes("non-medical") || (job.error || "").toLowerCase().includes("clinically relevant");
          if (isNonMedical) {
            msgEl.textContent = "Rejected: OCR read the document, but the AI clinical safety gate detected no medical content (e.g. ticket, receipt, or general document). Please upload a clinical report, lab slip, or prescription.";
          } else {
            msgEl.textContent = `Error: ${job.error || "Intake processing failed."}`;
          }
        }
      }

      if (job.status === "complete") {
        clearInterval(state.intakeJobTimer);
        state.intakeJobTimer = null;
        if (cpuWarning) cpuWarning.hidden = true;
        $("submit-audio-btn").disabled = false;
        $("submit-doc-btn").disabled = false;
        notify(`Intake finished: ${kind} data successfully ingested.`);
        await loadCase(state.caseId);
        await loadQueue();
        await loadIntelligence();
        if (autoProcess) {
          pollPipelineStatus(state.caseId);
        }
      } else if (job.status === "failed") {
        clearInterval(state.intakeJobTimer);
        state.intakeJobTimer = null;
        if (cpuWarning) cpuWarning.hidden = true;
        $("submit-audio-btn").disabled = false;
        $("submit-doc-btn").disabled = false;
        const isNonMedical = (job.error || "").toLowerCase().includes("non-medical") || (job.error || "").toLowerCase().includes("clinically relevant");
        if (isNonMedical) {
          notify("Non-medical document rejected: Please upload a clinical report or prescription.", true);
        } else {
          notify(`Intake failed: ${job.error || "Unknown error"}`, true);
        }
      }
    } catch (err) {
      clearInterval(state.intakeJobTimer);
      state.intakeJobTimer = null;
      if (cpuWarning) cpuWarning.hidden = true;
      $("submit-audio-btn").disabled = false;
      $("submit-doc-btn").disabled = false;
      notify(`Error polling intake job: ${err.message}`, true);
    }
  }, 1500);
}

function pollPipelineStatus(caseId) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  let elapsed = 0;
  state.pollTimer = setInterval(async () => {
    elapsed += 2;
    try {
      const status = await api(`/cases/${encodeURIComponent(caseId)}/status`);
      if (!["queued", "processing", "received", "consent_pending"].includes(status.status)) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        notify(`Pipeline finished: ${status.status} / ${status.workflow_status}.`);
        await loadCase(caseId);
        await loadQueue();
        await loadIntelligence();
      } else if (elapsed > 180) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    } catch (_) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }, 2000);
}

async function ingest(kind) {
  if (!state.caseId) { setStatus("Select a case first.", true); return; }
  const raw = $("ingest-json").value.trim();
  if (!raw) { setStatus("Paste STT/OCR JSON or a patient message first.", true); return; }
  let body;
  if (kind === "message") {
    try {
      body = JSON.parse(raw);
      if (typeof body === "string") body = { text: body };
    } catch (_) { body = { text: raw }; }
  } else {
    try { body = JSON.parse(raw); } catch (_) { setStatus("Invalid JSON.", true); return; }
  }
  const path = kind === "message" ? "message" : kind;
  try {
    const result = await api(`/cases/${encodeURIComponent(state.caseId)}/sources/${path}`, { method: "POST", body });
    setStatus(`Ingested ${result.inserted.length} segment(s); ${result.duplicates.length} duplicate(s) skipped.`);
    $("ingest-json").value = "";
    await loadCase(state.caseId);
  } catch (error) { setStatus(error.message, true); }
}

async function uploadFile() {
  if (!state.caseId) { setStatus("Select a case first.", true); return; }
  const file = $("upload-file").files[0];
  if (!file) { setStatus("Choose a file first.", true); return; }
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await api(`/cases/${encodeURIComponent(state.caseId)}/uploads`, { method: "POST", body: form });
    setStatus(result.duplicate ? "Duplicate upload detected; existing document reused." : `Uploaded document ${result.document_id}.`);
    await loadCase(state.caseId);
  } catch (error) { setStatus(error.message, true); }
}

async function processCase() {
  if (!state.caseId) return;
  $("process-btn").disabled = true;
  setStatus("Processing queued. Polling status…");
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/process`, { method: "POST" });
    let elapsed = 0;
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      elapsed += 2;
      try {
        const status = await api(`/cases/${encodeURIComponent(state.caseId)}/status`);
        if (!["queued", "processing", "received", "consent_pending"].includes(status.status)) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          $("process-btn").disabled = false;
          setStatus(`Processing finished: ${status.status} / ${status.workflow_status}.`);
          await loadCase(state.caseId);
          await loadQueue();
          await loadIntelligence();
        } else if (elapsed > 180) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          $("process-btn").disabled = false;
          setStatus("Processing is taking longer than expected. Check status again shortly.", true);
        }
      } catch (error) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        $("process-btn").disabled = false;
        setStatus(error.message, true);
      }
    }, 2000);
  } catch (error) {
    $("process-btn").disabled = false;
    setStatus(error.message, true);
  }
}

async function withdrawConsent() {
  if (!state.caseId) return;
  if (!confirm("Withdraw consent and delete source files, facts, summaries and queued data? This cannot be undone.")) return;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/consent/withdraw`, {
      method: "POST",
      body: { reason: "withdrawn from UI", delete_data: true },
    });
    setStatus("Consent withdrawn and deletion job executed.");
    state.caseId = null;
    state.caseData = null;
    $("case-view").hidden = true;
    $("welcome").hidden = false;
    await loadQueue();
    updateIntakeGate();
  } catch (error) { setStatus(error.message, true); }
}

async function reviewFact(factId, action) {
  if (!state.caseId) return;
  const body = { action };
  if (action === "correct") {
    const value = window.prompt("Corrected value (leave blank to keep current):", "");
    const note = window.prompt("Correction note (required):", "");
    if (note) body.note = note;
    if (value) body.value = value;
    if (!body.note && !body.value) return;
  } else if (action === "reject" || action === "dispute" || action === "verify") {
    const note = window.prompt("Optional review note:", "");
    if (note) body.note = note;
  }
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/facts/${encodeURIComponent(factId)}`, { method: "PATCH", body });
    await loadCase(state.caseId);
  } catch (error) { alert(error.message); }
}

async function acknowledgeWarning(warningId) {
  if (!state.caseId) return;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/warnings/${encodeURIComponent(warningId)}/acknowledge`, { method: "POST" });
    await loadCase(state.caseId);
  } catch (error) { alert(error.message); }
}

async function saveNote() {
  if (!state.caseId) return;
  const text = $("note-text").value;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/note`, { method: "PATCH", body: { text } });
    await loadCase(state.caseId);
  } catch (error) { alert(error.message); }
}

async function approveNote() {
  if (!state.caseId) return;
  if (!confirm("Approve this note as an authorized clinician? Your identity and timestamp will be recorded.")) return;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/note/approve`, { method: "POST" });
    await loadCase(state.caseId);
  } catch (error) { alert(error.message); }
}

async function overrideStatus() {
  if (!state.caseId) return;
  const reason = $("override-reason").value.trim();
  if (reason.length < 3) { alert("A documented reason is required."); return; }
  try {
    if (!confirm("Confirm this workflow status override? The original status, your identity, timestamp and reason will be retained.")) return;
    await api(`/cases/${encodeURIComponent(state.caseId)}/override`, {
      method: "POST",
      body: { final_status: $("override-status").value, reason, confirmed: true },
    });
    $("override-reason").value = "";
    notify("Workflow status overridden with an audit record.");
    await loadCase(state.caseId);
    await loadIntelligence();
  } catch (error) { notify(error.message, true); }
}

async function saveUrgency() {
  if (!state.caseId) return;
  const score = Number($("urgency-input").value);
  if (!Number.isInteger(score) || score < 0 || score > 100) {
    notify("Urgency must be a whole number from 0 to 100.", true);
    return;
  }
  const reason = $("urgency-reason").value.trim();
  if (reason.length < 3) {
    notify("A documented reason is required to edit urgency.", true);
    return;
  }
  if (!confirm("Update this advisory urgency score? Your identity, role, timestamp and reason will be recorded. The workflow category will not change.")) return;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/urgency`, {
      method: "PATCH",
      body: { score, reason, confirmed: true },
    });
    $("urgency-reason").value = "";
    notify("Urgency score updated with an audit record.");
    await loadCase(state.caseId);
    await loadIntelligence();
  } catch (error) { notify(error.message, true); }
}

async function assignCase() {
  if (!state.caseId) return;
  const reviewer = window.prompt("Assign this case to a clinician (user ID):", state.caseData?.case?.assigned_reviewer || "");
  if (!reviewer) return;
  if (!confirm(`Assign or reassign this case to ${reviewer}? This action is audited.`)) return;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/assign`, {
      method: "POST",
      body: { reviewer, confirmed: true },
    });
    notify(`Case assigned to ${reviewer}.`);
    await loadCase(state.caseId);
    await loadIntelligence();
  } catch (error) { notify(error.message, true); }
}

async function draftCommunication() {
  if (!state.caseId) return;
  const intent = $("communication-intent").value;
  const language = $("communication-language").value;
  try {
    await api(`/cases/${encodeURIComponent(state.caseId)}/communications/draft`, {
      method: "POST",
      body: { intent, language },
    });
    notify("Draft created. Review and approve it before sending.");
    await loadCase(state.caseId);
  } catch (error) { notify(error.message, true); }
}

async function communicationAction(messageId, action) {
  if (!state.caseId) return;
  const message = (state.caseData?.communications || []).find((item) => item.message_id === messageId);
  try {
    if (action === "approve") {
      const edited = window.prompt("Edit before approval (leave unchanged to keep the curated template):", message ? message.body : "");
      if (edited === null) return;
      if (!confirm("Approve this patient message? Your identity and timestamp will be recorded.")) return;
      const body = { text: message && edited === message.body ? null : edited };
      await api(`/cases/${encodeURIComponent(state.caseId)}/communications/${encodeURIComponent(messageId)}/approve`, {
        method: "POST", body,
      });
      notify("Message approved. Send it explicitly when ready.");
    } else if (action === "send") {
      if (!confirm("Send this approved patient message through the configured outbox?")) return;
      const result = await api(`/cases/${encodeURIComponent(state.caseId)}/communications/${encodeURIComponent(messageId)}/send`, { method: "POST" });
      notify(result.status === "sent" ? "Message delivered to the configured outbox." : `Delivery failed: ${result.last_error || "unknown error"}`, result.status !== "sent");
    } else if (action === "dismiss") {
      const reason = window.prompt("Reason for dismissing this draft:", "superseded by clinician");
      if (!reason) return;
      await api(`/cases/${encodeURIComponent(state.caseId)}/communications/${encodeURIComponent(messageId)}/dismiss`, {
        method: "POST", body: { reason },
      });
      notify("Draft dismissed.");
    }
    await loadCase(state.caseId);
  } catch (error) { notify(error.message, true); }
}

async function followupAction(followupId, action) {
  if (!state.caseId) return;
  const item = (state.caseData?.followups || []).find((entry) => entry.followup_id === followupId);
  try {
    if (action === "approve") {
      const edited = window.prompt("Edit before approval (leave unchanged to keep the question):", item ? item.text : "");
      if (edited === null) return;
      const language = window.prompt("Question language (en, hi, or hinglish):", item ? item.language : "en");
      if (!["en", "hi", "hinglish"].includes(language)) {
        notify("Choose en, hi, or hinglish. Other variants are not machine-translated.", true);
        return;
      }
      if (!confirm("Approve this follow-up question? Staff approval is recorded.")) return;
      await api(`/cases/${encodeURIComponent(state.caseId)}/followups/${encodeURIComponent(followupId)}/approve`, {
        method: "POST",
        body: { text: item && edited === item.text ? null : edited, language },
      });
      notify("Follow-up question approved.");
    } else if (action === "send") {
      if (!confirm("Send this approved follow-up question through the configured outbox?")) return;
      const result = await api(`/cases/${encodeURIComponent(state.caseId)}/followups/${encodeURIComponent(followupId)}/send`, { method: "POST" });
      notify(result.status === "sent" ? "Follow-up question sent." : `Send failed: ${result.last_error || "unknown error"}`, result.status !== "sent");
    } else if (action === "answer") {
      const text = window.prompt("Record the patient's answer in their own words:");
      if (!text) return;
      const result = await api(`/cases/${encodeURIComponent(state.caseId)}/followups/${encodeURIComponent(followupId)}/answers`, {
        method: "POST",
        body: { text, language: item ? item.language : "en", reprocess: true },
      });
      notify(`Answer stored as a new source. Processing: ${result.processing}.`);
      await loadCase(state.caseId);
      await loadQueue();
      await loadIntelligence();
      return;
    } else if (action === "dismiss") {
      const reason = window.prompt("Reason for dismissing this follow-up:", "not needed after clinician review");
      if (!reason) return;
      await api(`/cases/${encodeURIComponent(state.caseId)}/followups/${encodeURIComponent(followupId)}/dismiss`, {
        method: "POST", body: { reason },
      });
      notify("Follow-up dismissed.");
    }
    await loadCase(state.caseId);
  } catch (error) { notify(error.message, true); }
}

async function escalationAction(escalationId, action) {
  try {
    if (action === "acknowledge") {
      const note = window.prompt("Optional acknowledgement note:", "");
      if (note === null) return;
      await api(`/escalations/${encodeURIComponent(escalationId)}/acknowledge`, {
        method: "POST", body: { note: note || null },
      });
      notify("Emergency escalation acknowledged.");
    } else if (action === "retry") {
      const result = await api(`/escalations/${encodeURIComponent(escalationId)}/retry`, { method: "POST" });
      notify(result.status === "sent" ? "Escalation delivery retried successfully." : `Retry failed: ${result.last_error || "unknown error"}`, result.status !== "sent");
    }
    if (state.caseId) await loadCase(state.caseId);
    await loadIntelligence();
  } catch (error) { notify(error.message, true); }
}

async function syncFollowups() {
  if (!state.caseId) return;
  try {
    const res = await api(`/cases/${encodeURIComponent(state.caseId)}/followups/sync`, { method: "POST" });
    const count = (res && res.items) ? res.items.length : 0;
    if (count > 0) {
      notify(`Synced ${count} follow-up question(s) from current summary.`);
    } else {
      if (state.caseData?.case?.workflow_status === "EMERGENCY") {
        notify("Routine follow-up questions are suppressed for EMERGENCY cases. Immediate escalation takes priority.", true);
      } else if (!state.caseData?.summary) {
        notify("No clinical summary available yet. Please run the pipeline first.", true);
      } else {
        notify("No new follow-up questions suggested by the gap checks.");
      }
    }
    await loadCase(state.caseId);
  } catch (error) { notify(error.message, true); }
}

// ---------------------------------------------------------------- evidence
async function openSource(sourceId, start, end) {
  if (!sourceId) return;
  try {
    const source = await api(`/cases/${encodeURIComponent(state.caseId)}/sources/${encodeURIComponent(sourceId)}`);
    const text = source.original_text || "";
    const from = start === "" || start == null ? null : Number(start);
    const to = end === "" || end == null ? null : Number(end);
    let body;
    if (from != null && to != null && from >= 0 && to <= text.length && to > from) {
      body = `${escapeHtml(text.slice(0, from))}<mark>${escapeHtml(text.slice(from, to))}</mark>${escapeHtml(text.slice(to))}`;
    } else {
      body = escapeHtml(text);
    }
    $("modal-title").textContent = `Source evidence — ${source.modality}`;

    let mediaPreviewHtml = "";
    if (source.document_ref) {
      const docUrl = `/api/documents/${encodeURIComponent(source.document_ref)}?inline=true`;
      if (source.modality === "stt") {
        mediaPreviewHtml = `
          <div class="evidence-media">
            <p class="small-text muted">Audio evidence recording · seeking to ${source.audio_start != null ? source.audio_start + 's' : '0s'}:</p>
            <audio id="evidence-audio-player" controls preload="metadata" src="${docUrl}"></audio>
          </div>
        `;
      } else if (source.modality === "ocr") {
        mediaPreviewHtml = `
          <div class="evidence-media">
            <div class="evidence-doc-header">
              <span>Original document: <a href="${docUrl}" target="_blank" rel="noopener">${escapeHtml(source.document_ref)}</a></span>
              ${source.page != null ? `<span class="badge">Page ${source.page}</span>` : ""}
              ${source.bbox && source.bbox.length ? `<span class="badge">BBox: [${source.bbox.map((n) => Math.round(n)).join(", ")}]</span>` : ""}
            </div>
            <img class="evidence-image-preview" src="${docUrl}" alt="Evidence preview" onerror="this.style.display='none'" />
          </div>
        `;
      }
    }

    $("modal-body").innerHTML = `
      <p class="small-text muted">
        source_id: ${escapeHtml(source.source_id)} · segment: ${escapeHtml(source.segment_id)} ·
        language: ${escapeHtml(source.language)} (${escapeHtml(source.language_status)}) ·
        confidence: ${source.confidence != null ? source.confidence : "not provided"}<br>
        ${source.document_ref ? "document: " + escapeHtml(source.document_ref) + " · " : ""}
        ${source.page != null ? "page: " + source.page + " · " : ""}
        ${source.bbox ? "bbox: [" + source.bbox.join(", ") + "] · " : ""}
        ${source.audio_start != null ? "audio: " + source.audio_start + "s–" + (source.audio_end != null ? source.audio_end : "?") + "s · " : ""}
        ${source.warnings && source.warnings.length ? "warnings: " + escapeHtml(source.warnings.join(", ")) : ""}
      </p>
      ${mediaPreviewHtml}
      <div class="evidence-text">${body}</div>`;
    $("modal").hidden = false;

    // If audio element exists, seek to audio_start
    const player = $("evidence-audio-player");
    if (player && source.audio_start != null) {
      const seekTime = Number(source.audio_start);
      if (player.readyState >= 1) {
        player.currentTime = seekTime;
      } else {
        player.addEventListener("loadedmetadata", () => {
          player.currentTime = seekTime;
        }, { once: true });
      }
    }
  } catch (error) { alert(error.message); }
}

// ------------------------------------------------------------------- wiring
function wire() {
  $("save-identity").addEventListener("click", saveIdentity);
  $("refresh-queue").addEventListener("click", loadQueue);
  $("refresh-intel").addEventListener("click", loadIntelligence);
  $("new-case").addEventListener("click", createCase);
  $("copy-case-link").addEventListener("click", copyCaseLink);
  $("refresh-case").addEventListener("click", () => { if (state.caseId) loadCase(state.caseId); });
  $("assign-btn").addEventListener("click", assignCase);
  $("save-urgency").addEventListener("click", saveUrgency);
  $("consent-btn").addEventListener("click", grantConsent);
  $("process-btn").addEventListener("click", processCase);
  $("withdraw-btn").addEventListener("click", withdrawConsent);
  $("upload-btn").addEventListener("click", uploadFile);
  $("save-staffing").addEventListener("click", saveStaffing);
  $("draft-message").addEventListener("click", draftCommunication);
  $("sync-followups").addEventListener("click", syncFollowups);
  $("save-note").addEventListener("click", saveNote);
  $("approve-note").addEventListener("click", approveNote);
  $("override-btn").addEventListener("click", overrideStatus);
  $("modal-close").addEventListener("click", () => { $("modal").hidden = true; });
  $("modal").addEventListener("click", (event) => {
    if (event.target === $("modal")) $("modal").hidden = true;
  });
  document.querySelectorAll("[data-ingest]").forEach((button) => {
    button.addEventListener("click", () => ingest(button.dataset.ingest));
  });
  $("role").addEventListener("change", () => {
    $("patient-id").style.display = $("role").value === "patient" ? "" : "none";
  });

  // API Bridge wiring
  const apiInput = $("api-url");
  if (apiInput) {
    apiInput.value = localStorage.getItem("careflow.apiUrl") || "";
    $("save-api-url").addEventListener("click", () => {
      setApiBaseUrl(apiInput.value);
      loadQueue();
      loadIntelligence();
      if (state.caseId) loadCase(state.caseId, false);
    });
    apiInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        setApiBaseUrl(apiInput.value);
        loadQueue();
        loadIntelligence();
        if (state.caseId) loadCase(state.caseId, false);
      }
    });
  }

  // Direct intake wiring
  document.querySelectorAll(".intake-tab").forEach((button) => {
    button.addEventListener("click", () => switchIntakeTab(button.dataset.tab));
  });
  $("record-btn").addEventListener("click", toggleRecording);
  $("stop-record-btn").addEventListener("click", stopRecording);
  $("submit-audio-btn").addEventListener("click", submitAudioIntake);
  $("submit-doc-btn").addEventListener("click", submitDocumentIntake);
  $("submit-message-btn").addEventListener("click", submitMessageIntake);

  // Patient voice experience wiring
  $("patient-card-load").addEventListener("click", () => loadPatientCard(false));
  $("patient-card-listen").addEventListener("click", () => loadPatientCard(true, true));
  $("patient-detect-language").addEventListener("click", detectPatientLanguage);
  $("patient-language").addEventListener("change", async () => {
    state.patientFollowupCache = {};
    state.patientCardCache = {};
    if (state.caseId) {
      await loadPatientCard(false);
      await localizePatientFollowups();
    }
  });
  $("patient-followups").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const followupId = (
      button.dataset.patientPlay
      || button.dataset.patientSendText
      || button.dataset.patientRecord
      || button.dataset.patientStop
      || button.dataset.patientSendFile
    );
    if (!followupId) return;
    if (button.dataset.patientPlay) {
      playPatientFollowup(followupId);
    } else if (button.dataset.patientSendText) {
      submitPatientTextAnswer(followupId);
    } else if (button.dataset.patientRecord) {
      startPatientRecording(followupId, button);
    } else if (button.dataset.patientStop) {
      stopPatientRecording(followupId);
    } else if (button.dataset.patientSendFile) {
      const input = document.querySelector(`[data-patient-voice-file="${CSS.escape(followupId)}"]`);
      const file = input && input.files ? input.files[0] : null;
      if (!file) {
        setPatientAnswerStatus(followupId, "Choose an audio file first.", true);
        return;
      }
      submitPatientAudioAnswer(followupId, file);
    }
  });
}

loadIdentity();
applyRoleVisibility();
wire();
$("patient-id").style.display = state.identity.role === "patient" ? "" : "none";
checkApiHealth();
updateIntakeGate();
loadQueue();
loadIntelligence();

// Deep link support: If direct link was used (e.g. ?case_id=CASE_xxx or ?case=CASE_xxx), load it immediately
try {
  const urlParams = new URLSearchParams(window.location.search);
  const initialCaseId = urlParams.get("case_id") || urlParams.get("case");
  if (initialCaseId) {
    loadCase(initialCaseId, false);
  }
} catch (_) {}
