"use strict";

/* Client logic for the local requirements validator demo.
 * Security invariant: all document text, quotes, filenames, and report strings
 * are inserted strictly via textContent, never as raw HTML.
 */

(function () {
  const POLL_INTERVAL_MS = 1500;
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = new Set([".md", ".txt", ".pdf", ".docx"]);

  const SEVERITY_RANK = { critical: 0, major: 1, minor: 2 };

  const SEVERITY_META = {
    critical: { label: "Критичный", icon: "●", chipClass: "chip--critical" },
    major: { label: "Существенный", icon: "▲", chipClass: "chip--major" },
    minor: { label: "Незначительный", icon: "■", chipClass: "chip--minor" },
  };

  const READINESS_META = {
    yes: { label: "Да", chipClass: "chip--ok" },
    conditional: { label: "Условно", chipClass: "chip--major" },
    no: { label: "Нет", chipClass: "chip--critical" },
  };

  const RUN_STATUS_LABELS = {
    complete: "Полный отчёт",
    partial: "Частичный отчёт",
    rules_only: "Только проверки по правилам",
  };

  const STAGE_STATE_META = {
    pending: { label: "ожидает", icon: "○", chipClass: "chip--neutral" },
    running: { label: "выполняется", icon: "", chipClass: "chip--accent" },
    done: { label: "готово", icon: "✓", chipClass: "chip--ok" },
    cached: { label: "готово (кэш)", icon: "✓", chipClass: "chip--ok" },
    skipped: { label: "пропущено", icon: "—", chipClass: "chip--neutral" },
    failed: { label: "ошибка", icon: "✕", chipClass: "chip--critical" },
  };

  const REQ_KIND_LABELS = {
    functional: "Функциональное",
    nonfunctional: "Нефункциональное",
    data: "Требование к данным",
    interface: "Интерфейс / API",
    integration: "Интеграция",
    ui: "Интерфейс пользователя (UI)",
    business_goal: "Бизнес-цель",
    process: "Бизнес-процесс",
    constraint: "Ограничение",
    acceptance_criterion: "Критерий приёмки",
    other: "Прочее",
  };

  const SET_CHECK_LABELS = {
    contradiction: "Противоречие",
    cross_part: "Расхождение между частями документа",
    acceptance_mismatch: "Расхождение с критерием приёмки",
    acceptance_gap: "Пробел в покрытии критериями приёмки",
    missing_requirement: "Отсутствующее требование",
    missing_scenario: "Отсутствующий сценарий",
    empty_section: "Пустой или незаполненный раздел",
    terminology: "Расхождение терминологии",
    duplication: "Дублирование",
    traceability: "Нарушение трассируемости",
    other: "Прочее",
  };

  const QUOTE_STATUS_LABELS = {
    verified: "подтверждена дословно",
    repaired: "уточнена по тексту документа",
    anchored: "привязана к тексту требования",
  };

  const COVERAGE_STATUS_META = {
    covered: { label: "Покрыто", chipClass: "chip--ok" },
    partial: { label: "Частично", chipClass: "chip--major" },
    none: { label: "Не покрыто", chipClass: "chip--critical" },
  };

  const SKIPPED_REASON_LABELS = {
    meta: "служебная информация",
    template: "шаблон",
    empty: "пустой блок",
    other: "прочее",
  };

  const CRITERION_META = {
    necessary: { label: "Необходимость", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5" },
    appropriate: { label: "Уместность уровня", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5" },
    unambiguous: { label: "Однозначность", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5 · IEEE 830-1998, п. 4.3.2" },
    complete: { label: "Полнота", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5 · IEEE 830-1998, п. 4.3.3" },
    singular: { label: "Единичность", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5 · IEEE 830-1998, п. 4.3.7" },
    feasible: { label: "Выполнимость", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5" },
    verifiable: { label: "Проверяемость", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5 · IEEE 830-1998, п. 4.3.6" },
    correct: { label: "Корректность", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5 · IEEE 830-1998, п. 4.3.1" },
    conforming: { label: "Соответствие правилам и шаблону", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.5" },
    set_complete: { label: "Полнота набора", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.6 · IEEE 830-1998, п. 4.3.3" },
    set_consistent: { label: "Согласованность набора", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.6 · IEEE 830-1998, п. 4.3.4" },
    set_feasible: { label: "Выполнимость набора", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.6" },
    set_comprehensible: { label: "Понятность набора", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.6 · IEEE 830-1998, п. 4.3.2" },
    set_validatable: { label: "Валидируемость набора", ref: "ISO/IEC/IEEE 29148:2018, п. 5.2.6" },
    ranked: { label: "Ранжирование по важности", ref: "IEEE 830-1998, п. 4.3.5" },
    modifiable: { label: "Модифицируемость", ref: "IEEE 830-1998, п. 4.3.7" },
    traceable: { label: "Трассируемость", ref: "IEEE 830-1998, п. 4.3.8" },
  };

  // Application state.
  const state = {
    runId: null,
    pollTimer: null,
    elapsedTimer: null,
    startedAtMs: null,
    isSubmitting: false,
    report: null,
    markdown: "",
    allFindings: [],
    activeTab: "findings",
    severityFilter: "all",
    criterionFilter: "all",
    searchQuery: "",
    expandedReqIds: new Set(),
  };

  // DOM helpers: strictly DOM nodes and textContent.
  function $(id) {
    return document.getElementById(id);
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null || value === false) {
          continue;
        }
        if (key === "className") {
          node.className = value;
        } else if (key === "dataset") {
          for (const [dk, dv] of Object.entries(value)) {
            node.dataset[dk] = dv;
          }
        } else if (key.startsWith("on") && typeof value === "function") {
          node.addEventListener(key.slice(2).toLowerCase(), value);
        } else {
          node.setAttribute(key, value === true ? "" : String(value));
        }
      }
    }
    for (const child of children) {
      if (child === undefined || child === null || child === false) {
        continue;
      }
      if (typeof child === "string" || typeof child === "number") {
        node.appendChild(document.createTextNode(String(child)));
      } else {
        node.appendChild(child);
      }
    }
    return node;
  }

  function makeChip(text, extraClass) {
    return el("span", { className: `chip ${extraClass || "chip--neutral"}` }, text);
  }

  function criterionTitle(code) {
    return (CRITERION_META[code] && CRITERION_META[code].label) || code;
  }

  function criterionFullRef(code) {
    const meta = CRITERION_META[code];
    if (!meta) {
      return code;
    }
    return `${meta.label} · ${meta.ref}`;
  }

  function findingHeading(item) {
    if (item._isSet) {
      return SET_CHECK_LABELS[item.set_check] || criterionTitle(item.criterion);
    }
    return criterionTitle(item.criterion);
  }

  // Error banner controls.
  function showError(message) {
    $("error-banner-text").textContent = message;
    $("error-banner").classList.remove("hidden");
  }

  function hideError() {
    $("error-banner").classList.add("hidden");
    $("error-banner-text").textContent = "";
  }

  // Busy controls for upload actions.
  function setControlsDisabled(disabled) {
    $("select-file-btn").disabled = disabled;
    $("sample-btn").disabled = disabled;
    $("profile-select").disabled = disabled;
    $("file-input").disabled = disabled;
  }

  function stopPolling() {
    if (state.pollTimer !== null) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
    if (state.elapsedTimer !== null) {
      clearInterval(state.elapsedTimer);
      state.elapsedTimer = null;
    }
  }

  function startElapsedClock() {
    state.startedAtMs = Date.now();
    $("progress-elapsed").textContent = "0 с";
    if (state.elapsedTimer !== null) {
      clearInterval(state.elapsedTimer);
    }
    state.elapsedTimer = setInterval(() => {
      const sec = Math.max(0, Math.floor((Date.now() - state.startedAtMs) / 1000));
      $("progress-elapsed").textContent = `${sec} с`;
    }, 1000);
  }

  // Health check on load.
  async function fetchHealth() {
    const badge = $("health-badge");
    const warning = $("config-warning");
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data.key_configured) {
        badge.className = "status-pill status-pill--ok";
        badge.textContent = `Сервис готов · v${data.version}`;
        warning.classList.add("hidden");
      } else {
        badge.className = "status-pill status-pill--warn";
        badge.textContent = `Ключ не настроен · v${data.version}`;
        warning.classList.remove("hidden");
      }
    } catch (_err) {
      badge.className = "status-pill status-pill--warn";
      badge.textContent = "Нет связи с сервером";
    }
  }

  // API error extraction.
  async function extractErrorMessage(response, fallback) {
    try {
      const data = await response.json();
      if (data && typeof data.detail === "string" && data.detail.trim()) {
        return data.detail;
      }
    } catch (_err) {
      // Ignore non-JSON body.
    }
    return fallback;
  }

  // Client-side file pre-validation.
  function validateFile(file) {
    if (!file) {
      return "Файл не выбран.";
    }
    const name = file.name || "";
    const dot = name.lastIndexOf(".");
    const ext = dot >= 0 ? name.slice(dot).toLowerCase() : "";
    if (!ALLOWED_EXTENSIONS.has(ext)) {
      return "Неподдерживаемый формат файла. Допустимы: .md, .txt, .pdf, .docx.";
    }
    if (file.size === 0) {
      return "Выбранный файл пуст.";
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return "Размер файла превышает ограничение 10 МБ.";
    }
    return null;
  }

  function resetToUpload() {
    stopPolling();
    hideError();
    state.runId = null;
    state.report = null;
    state.markdown = "";
    state.allFindings = [];
    state.expandedReqIds.clear();
    state.severityFilter = "all";
    state.criterionFilter = "all";
    state.searchQuery = "";
    $("finding-search").value = "";
    $("file-input").value = "";
    $("selected-file-label").textContent = "";
    $("upload-section").classList.remove("hidden");
    $("progress-section").classList.add("hidden");
    $("result-section").classList.add("hidden");
    $("new-check-btn").classList.add("hidden");
    setControlsDisabled(false);
  }

  async function submitFile(file) {
    if (state.isSubmitting) {
      return;
    }
    hideError();
    const validationErr = validateFile(file);
    if (validationErr) {
      showError(validationErr);
      return;
    }

    state.isSubmitting = true;
    setControlsDisabled(true);
    $("selected-file-label").textContent = `Файл: ${file.name}`;

    const formData = new FormData();
    formData.append("file", file);
    const profile = $("profile-select").value;
    if (profile) {
      formData.append("profile", profile);
    }

    try {
      const res = await fetch("/api/runs", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const msg = await extractErrorMessage(res, `Не удалось запустить проверку (HTTP ${res.status}).`);
        showError(msg);
        setControlsDisabled(false);
        return;
      }
      const created = await res.json();
      beginTrackingRun(created.run_id, file.name);
    } catch (_err) {
      showError("Ошибка соединения с сервером при загрузке файла.");
      setControlsDisabled(false);
    } finally {
      state.isSubmitting = false;
    }
  }

  async function submitSample() {
    if (state.isSubmitting) {
      return;
    }
    hideError();
    state.isSubmitting = true;
    setControlsDisabled(true);
    $("selected-file-label").textContent = "Запуск встроенного примера…";

    const profile = $("profile-select").value;
    const url = profile ? `/api/runs/sample?profile=${encodeURIComponent(profile)}` : "/api/runs/sample";

    try {
      const res = await fetch(url, { method: "POST" });
      if (!res.ok) {
        const msg = await extractErrorMessage(res, `Не удалось запустить пример (HTTP ${res.status}).`);
        showError(msg);
        setControlsDisabled(false);
        return;
      }
      const created = await res.json();
      beginTrackingRun(created.run_id, "requirements.md");
    } catch (_err) {
      showError("Ошибка соединения с сервером при запуске примера.");
      setControlsDisabled(false);
    } finally {
      state.isSubmitting = false;
    }
  }

  function beginTrackingRun(runId, documentName) {
    stopPolling();
    state.runId = runId;
    state.report = null;
    state.markdown = "";
    $("upload-section").classList.add("hidden");
    $("result-section").classList.add("hidden");
    $("progress-section").classList.remove("hidden");
    $("new-check-btn").classList.add("hidden");
    $("progress-doc-name").textContent = documentName;
    $("progress-percent").textContent = "0%";
    $("progress-bar-fill").style.width = "0%";
    $("progress-bar-track").setAttribute("aria-valuenow", "0");
    clearNode($("stage-list"));
    startElapsedClock();
    pollStatus();
  }

  async function pollStatus() {
    if (!state.runId) {
      return;
    }
    const currentRunId = state.runId;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(currentRunId)}`, { cache: "no-store" });
      if (!res.ok) {
        const msg = await extractErrorMessage(res, `Не удалось получить статус проверки (HTTP ${res.status}).`);
        stopPolling();
        showError(msg);
        setControlsDisabled(false);
        $("upload-section").classList.remove("hidden");
        return;
      }
      const dto = await res.json();
      if (state.runId !== currentRunId) {
        return;
      }
      renderProgress(dto);

      if (dto.status === "done") {
        stopPolling();
        await loadFinalReport(currentRunId);
        return;
      }
      if (dto.status === "failed") {
        stopPolling();
        showError(dto.error || "Проверка завершилась с ошибкой.");
        setControlsDisabled(false);
        $("upload-section").classList.remove("hidden");
        $("new-check-btn").classList.remove("hidden");
        return;
      }

      state.pollTimer = setTimeout(pollStatus, POLL_INTERVAL_MS);
    } catch (_err) {
      if (state.runId === currentRunId) {
        state.pollTimer = setTimeout(pollStatus, POLL_INTERVAL_MS);
      }
    }
  }

  function renderProgress(dto) {
    $("progress-doc-name").textContent = dto.document_name || "";
    const pct = Math.round(Math.max(0, Math.min(1, dto.progress || 0)) * 100);
    $("progress-percent").textContent = `${pct}%`;
    $("progress-bar-fill").style.width = `${pct}%`;
    $("progress-bar-track").setAttribute("aria-valuenow", String(pct));

    const list = $("stage-list");
    clearNode(list);
    for (const stage of dto.stages || []) {
      const meta = STAGE_STATE_META[stage.state] || STAGE_STATE_META.pending;
      const iconWrap = el("span", {
        className: `stage-icon stage-icon--${stage.state}`,
        "aria-hidden": "true",
      });
      if (stage.state === "running") {
        iconWrap.appendChild(el("span", { className: "spinner" }));
      } else {
        iconWrap.textContent = meta.icon;
      }

      const main = el(
        "div",
        { className: "stage-main" },
        el("div", { className: "stage-name" }, stage.title || stage.key),
        stage.detail ? el("div", { className: "stage-detail" }, stage.detail) : null,
      );

      const rightChildren = [];
      if (typeof stage.seconds === "number" && stage.seconds > 0) {
        rightChildren.push(el("span", { className: "stage-time" }, `${stage.seconds.toFixed(1)} с`));
      }
      rightChildren.push(makeChip(meta.label, meta.chipClass));

      const rowClass =
        stage.state === "running"
          ? "stage-row stage-row--running"
          : stage.state === "failed"
            ? "stage-row stage-row--failed"
            : "stage-row";

      list.appendChild(
        el("li", { className: rowClass }, iconWrap, main, el("div", { className: "stage-right" }, ...rightChildren)),
      );
    }
  }

  async function loadFinalReport(runId) {
    try {
      const [jsonRes, mdRes] = await Promise.all([
        fetch(`/api/runs/${encodeURIComponent(runId)}/report.json`, { cache: "no-store" }),
        fetch(`/api/runs/${encodeURIComponent(runId)}/report.md`, { cache: "no-store" }),
      ]);
      if (!jsonRes.ok) {
        const msg = await extractErrorMessage(jsonRes, `Не удалось загрузить отчёт JSON (HTTP ${jsonRes.status}).`);
        showError(msg);
        setControlsDisabled(false);
        $("upload-section").classList.remove("hidden");
        return;
      }
      const report = await jsonRes.json();
      const markdown = mdRes.ok ? await mdRes.text() : "";

      state.report = report;
      state.markdown = markdown;
      state.allFindings = buildCombinedFindings(report);
      state.severityFilter = "all";
      state.criterionFilter = "all";
      state.searchQuery = "";
      state.expandedReqIds.clear();
      $("finding-search").value = "";

      renderResult(runId, report, markdown);
      $("progress-section").classList.add("hidden");
      $("result-section").classList.remove("hidden");
      $("new-check-btn").classList.remove("hidden");
      setControlsDisabled(false);
    } catch (_err) {
      showError("Ошибка при получении сформированного отчёта.");
      setControlsDisabled(false);
      $("upload-section").classList.remove("hidden");
    }
  }

  function buildCombinedFindings(report) {
    const reqItems = (report.findings || []).map((f) => Object.assign({}, f, { _isSet: false }));
    const setItems = (report.set_findings || []).map((sf) => Object.assign({}, sf, { _isSet: true }));
    return [...reqItems, ...setItems].sort((a, b) => {
      const ra = SEVERITY_RANK[a.severity] ?? 99;
      const rb = SEVERITY_RANK[b.severity] ?? 99;
      if (ra !== rb) {
        return ra - rb;
      }
      return String(a.id).localeCompare(String(b.id));
    });
  }

  function renderResult(runId, report, markdown) {
    renderPartialBanner(report);
    renderHeaderAndLinks(runId, report);
    renderVerdict(report.verdict);
    renderSummaryCards(report);
    populateCriterionFilter(state.allFindings);
    updateTabCounts(report);
    renderFindingsTab();
    renderRequirementsTab(report);
    renderSetCheckTab(report);
    renderQuestionsTab(report);
    renderMarkdownTab(markdown);
    renderPassportDetails(report);
    activateTab("findings");
  }

  function renderPartialBanner(report) {
    const banner = $("partial-banner");
    const run = report.run || {};
    if (run.status === "complete" && (!run.warnings || run.warnings.length === 0)) {
      banner.classList.add("hidden");
      return;
    }
    const statusText = RUN_STATUS_LABELS[run.status] || run.status || "Внимание";
    $("partial-banner-title").textContent =
      run.status !== "complete"
        ? `Внимание: сформирован неполный отчёт (${statusText.toLowerCase()}).`
        : "Предупреждения при разборе документа:";

    const ul = $("partial-banner-list");
    clearNode(ul);
    for (const warning of run.warnings || []) {
      ul.appendChild(el("li", null, warning));
    }
    banner.classList.remove("hidden");
  }

  function renderHeaderAndLinks(runId, report) {
    const run = report.run || {};
    $("result-doc-title").textContent = run.document_title || run.document_name || "Отчёт о проверке";

    const meta = $("result-meta");
    clearNode(meta);
    const items = [];
    if (run.document_name) {
      items.push(`Файл: ${run.document_name}`);
    }
    if (typeof run.duration_seconds === "number") {
      items.push(`Время: ${run.duration_seconds.toFixed(1)} с`);
    }
    if (run.profile) {
      items.push(`Профиль: ${run.profile}`);
    }
    if (Array.isArray(run.models_used) && run.models_used.length > 0) {
      items.push(`Модель: ${run.models_used.join(", ")}`);
    }
    if (run.status) {
      items.push(`Статус: ${RUN_STATUS_LABELS[run.status] || run.status}`);
    }
    for (const text of items) {
      meta.appendChild(el("span", null, text));
    }

    const mdUrl = `/api/runs/${encodeURIComponent(runId)}/report.md?download=1`;
    const jsonUrl = `/api/runs/${encodeURIComponent(runId)}/report.json?download=1`;
    $("download-md-btn").setAttribute("href", mdUrl);
    $("panel-download-md-btn").setAttribute("href", mdUrl);
    $("download-json-btn").setAttribute("href", jsonUrl);
  }

  function renderVerdict(verdict) {
    const v = verdict || { development: "no", testing: "no", headline: "", rationale: "", fix_first: [] };
    const overall =
      v.development === "no" || v.testing === "no"
        ? "no"
        : v.development === "conditional" || v.testing === "conditional"
          ? "conditional"
          : "yes";

    const banner = $("verdict-banner");
    banner.className = `card verdict-card verdict-card--${overall}`;
    $("verdict-heading").textContent = v.headline || "Итог проверки сформирован.";

    const devMeta = READINESS_META[v.development] || READINESS_META.no;
    const testMeta = READINESS_META[v.testing] || READINESS_META.no;
    const devChip = $("readiness-dev");
    devChip.className = `chip ${devMeta.chipClass}`;
    devChip.textContent = devMeta.label;
    const testChip = $("readiness-test");
    testChip.className = `chip ${testMeta.chipClass}`;
    testChip.textContent = testMeta.label;

    const rationaleEl = $("verdict-rationale");
    if (v.rationale && v.rationale.trim()) {
      rationaleEl.textContent = v.rationale;
      rationaleEl.classList.remove("hidden");
    } else {
      rationaleEl.textContent = "";
      rationaleEl.classList.add("hidden");
    }

    const fixBlock = $("fix-first-block");
    const fixList = $("fix-first-list");
    clearNode(fixList);
    if (Array.isArray(v.fix_first) && v.fix_first.length > 0) {
      for (const item of v.fix_first) {
        const jumpBtn = el(
          "button",
          {
            type: "button",
            className: "fix-link",
            onClick: () => jumpToFinding(item.finding_id),
          },
          item.finding_id,
        );
        fixList.appendChild(el("li", null, jumpBtn, el("span", null, item.title || "")));
      }
      fixBlock.classList.remove("hidden");
    } else {
      fixBlock.classList.add("hidden");
    }
  }

  function renderSummaryCards(report) {
    const s = report.summary || {};
    const bySev = s.findings_by_severity || {};
    const crit = bySev.critical || 0;
    const major = bySev.major || 0;
    const minor = bySev.minor || 0;
    const blockingQ = (report.questions || []).filter((q) => q.blocking).length;

    $("metric-req-total").textContent = String(s.requirements_total ?? (report.requirements || []).length);
    const reqParts = [];
    if (s.requirements_with_findings) {
      reqParts.push(`с замечаниями: ${s.requirements_with_findings}`);
    }
    if (s.requirements_not_checked) {
      reqParts.push(`не проверено: ${s.requirements_not_checked}`);
    }
    $("metric-req-sub").textContent = reqParts.length ? reqParts.join(" · ") : "все проверены";

    $("metric-critical").textContent = String(crit);
    $("metric-major").textContent = String(major);
    $("metric-minor").textContent = String(minor);
    $("metric-questions").textContent = String(s.questions_total ?? (report.questions || []).length);
    $("metric-questions-sub").textContent = `блокирующих: ${blockingQ}`;

    $("flt-cnt-all").textContent = String(state.allFindings.length);
    $("flt-cnt-critical").textContent = String(crit);
    $("flt-cnt-major").textContent = String(major);
    $("flt-cnt-minor").textContent = String(minor);
  }

  function updateTabCounts(report) {
    $("tab-count-findings").textContent = String(state.allFindings.length);
    $("tab-count-requirements").textContent = String((report.requirements || []).length);
    $("tab-count-set").textContent = String((report.set_findings || []).length);
    $("tab-count-questions").textContent = String((report.questions || []).length);
  }

  function populateCriterionFilter(items) {
    const select = $("criterion-filter");
    clearNode(select);
    select.appendChild(el("option", { value: "all" }, "Все критерии"));

    const counts = new Map();
    for (const item of items) {
      counts.set(item.criterion, (counts.get(item.criterion) || 0) + 1);
    }
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    for (const [code, count] of sorted) {
      select.appendChild(el("option", { value: code }, `${criterionTitle(code)} (${count})`));
    }
    select.value = "all";
  }

  // Shared finding card builder used in both "Нарушения" and "Проверка набора".
  function buildFindingCard(item, idPrefix) {
    const sevMeta = SEVERITY_META[item.severity] || SEVERITY_META.minor;
    const headingText = findingHeading(item);

    const tags = [];
    if (item._isSet) {
      tags.push(makeChip("Набор", "chip--neutral"));
    }
    if (item.cross_part) {
      tags.push(makeChip("Между частями", "chip--accent"));
    }
    for (const rid of item.requirement_ids || []) {
      tags.push(makeChip(rid, "chip--neutral"));
    }

    const header = el(
      "div",
      { className: "finding-head" },
      el(
        "div",
        { className: "finding-head-left" },
        makeChip(`${sevMeta.icon} ${sevMeta.label}`, sevMeta.chipClass),
        el("span", { className: "finding-id" }, item.id),
        el("h4", { className: "finding-title" }, headingText),
      ),
      tags.length ? el("div", { className: "finding-tags" }, ...tags) : null,
    );

    const criterionEl = el("div", { className: "finding-criterion" }, `Критерий: ${criterionFullRef(item.criterion)}`);

    const evidenceList = Array.isArray(item.evidence) && item.evidence.length > 0
      ? item.evidence
      : [{ quote: item.quote || "", section: item.section || "", requirement_id: null, quote_status: "verified" }];

    const sectionEl =
      evidenceList.length === 1 && evidenceList[0].section
        ? el("div", { className: "finding-section" }, `Раздел: ${evidenceList[0].section}`)
        : null;

    const evidenceWrap = el("div", { className: "evidence-list" });
    evidenceList.forEach((ev, idx) => {
      const metaParts = [];
      if (evidenceList.length > 1) {
        metaParts.push(el("span", null, `Цитата ${idx + 1}`));
      }
      if (evidenceList.length > 1 && ev.section) {
        metaParts.push(el("span", null, `Раздел: ${ev.section}`));
      }
      if (ev.requirement_id) {
        metaParts.push(makeChip(ev.requirement_id, "chip--neutral"));
      }
      if (ev.quote_status && QUOTE_STATUS_LABELS[ev.quote_status]) {
        metaParts.push(el("span", null, `(${QUOTE_STATUS_LABELS[ev.quote_status]})`));
      }

      evidenceWrap.appendChild(
        el(
          "div",
          { className: "evidence-item" },
          metaParts.length ? el("div", { className: "evidence-meta" }, ...metaParts) : null,
          el("blockquote", { className: `quote-box quote-box--${item.severity}` }, ev.quote || ""),
        ),
      );
    });

    const fields = el(
      "div",
      { className: "finding-fields" },
      el(
        "p",
        { className: "finding-field" },
        el("span", { className: "finding-field-label" }, "Объяснение: "),
        item.explanation || "",
      ),
      el(
        "p",
        { className: "finding-field" },
        el("span", { className: "finding-field-label" }, "Рекомендация: "),
        item.recommendation || "",
      ),
    );

    let questionBox = null;
    if (item.clarifying_question) {
      let optionsNode = null;
      if (Array.isArray(item.answer_options) && item.answer_options.length > 0) {
        optionsNode = el(
          "ul",
          { className: "answer-options" },
          ...item.answer_options.map((opt) => el("li", null, opt)),
        );
      }
      questionBox = el(
        "div",
        { className: "finding-question-box" },
        el(
          "p",
          { className: "finding-field" },
          el("span", { className: "finding-field-label" }, "Уточняющий вопрос: "),
          item.clarifying_question,
        ),
        optionsNode,
      );
    }

    const reviewNote = item.review_note
      ? el(
          "p",
          { className: "muted-note" },
          `Примечание пересмотра: ${item.review_note}`,
        )
      : null;

    const cardAttrs = { className: `finding-card finding-card--${item.severity}` };
    if (idPrefix) {
      cardAttrs.id = `${idPrefix}-${item.id}`;
    }

    return el(
      "article",
      cardAttrs,
      header,
      criterionEl,
      sectionEl,
      evidenceWrap,
      fields,
      questionBox,
      reviewNote,
    );
  }

  function renderFindingsTab() {
    const container = $("findings-list");
    clearNode(container);

    // Sync filter pills active styling.
    for (const pill of document.querySelectorAll(".filter-pill[data-severity]")) {
      pill.classList.toggle("is-active", pill.dataset.severity === state.severityFilter);
    }
    $("criterion-filter").value = state.criterionFilter;

    if (state.allFindings.length === 0) {
      container.appendChild(
        el("div", { className: "empty-state empty-state--ok" }, "✓ Нарушений в документе не выявлено."),
      );
      return;
    }

    const query = state.searchQuery.trim().toLowerCase();
    const filtered = state.allFindings.filter((item) => {
      if (state.severityFilter !== "all" && item.severity !== state.severityFilter) {
        return false;
      }
      if (state.criterionFilter !== "all" && item.criterion !== state.criterionFilter) {
        return false;
      }
      if (query) {
        const haystack = [
          item.id,
          item.criterion,
          criterionTitle(item.criterion),
          findingHeading(item),
          item.quote,
          item.section,
          item.explanation,
          item.recommendation,
          item.clarifying_question,
          ...(item.requirement_ids || []),
          ...(item.evidence || []).map((e) => `${e.quote} ${e.section}`),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        if (!haystack.includes(query)) {
          return false;
        }
      }
      return true;
    });

    if (filtered.length === 0) {
      const resetBtn = el(
        "button",
        {
          type: "button",
          className: "btn btn-secondary btn-sm",
          onClick: () => {
            state.severityFilter = "all";
            state.criterionFilter = "all";
            state.searchQuery = "";
            $("finding-search").value = "";
            renderFindingsTab();
          },
        },
        "Сбросить фильтры",
      );
      container.appendChild(
        el("div", { className: "empty-state" }, el("span", null, "Нарушений по выбранным фильтрам не найдено."), resetBtn),
      );
      return;
    }

    for (const item of filtered) {
      container.appendChild(buildFindingCard(item, "finding"));
    }
  }

  function renderRequirementsTab(report) {
    const tbody = $("requirements-tbody");
    clearNode(tbody);

    const reqs = report.requirements || [];
    const findingsByReq = new Map();
    for (const item of state.allFindings) {
      for (const rid of item.requirement_ids || []) {
        if (!findingsByReq.has(rid)) {
          findingsByReq.set(rid, []);
        }
        findingsByReq.get(rid).push(item);
      }
    }

    if (reqs.length === 0) {
      tbody.appendChild(
        el("tr", null, el("td", { colspan: "6", className: "muted-cell" }, "Требования в документе не выделены.")),
      );
    } else {
      for (const req of reqs) {
        const reqFindings = findingsByReq.get(req.id) || [];
        const statusChip = buildRequirementStatusChip(req, reqFindings);
        const docIdNode =
          req.id_assigned || !req.doc_id
            ? el("span", { className: "muted-cell" }, "присвоен агентом")
            : el("span", { className: "mono-cell" }, req.doc_id);

        const isExpanded = state.expandedReqIds.has(req.id);
        const tr = el(
          "tr",
          {
            className: "req-row",
            tabindex: "0",
            "aria-expanded": String(isExpanded),
            onClick: () => toggleRequirementRow(req.id),
            onKeydown: (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                toggleRequirementRow(req.id);
              }
            },
          },
          el("td", { className: "mono-cell" }, req.id),
          el("td", null, docIdNode),
          el("td", null, req.section || "—"),
          el("td", null, REQ_KIND_LABELS[req.kind] || req.kind || "—"),
          el("td", null, req.title || ""),
          el("td", null, statusChip),
        );
        tbody.appendChild(tr);

        if (isExpanded) {
          const links = reqFindings.map((f) =>
            el(
              "button",
              {
                type: "button",
                className: "fix-link",
                onClick: (ev) => {
                  ev.stopPropagation();
                  jumpToFinding(f.id);
                },
              },
              f.id,
            ),
          );
          const detailBox = el(
            "div",
            { className: "req-detail-box" },
            el("div", { className: "muted-note" }, "Дословный текст из документа:"),
            el("blockquote", { className: "quote-box" }, req.quote || ""),
            links.length
              ? el("div", { className: "muted-note" }, "Связанные нарушения: ", ...links)
              : el("div", { className: "muted-note" }, "Нарушений по этому требованию не зафиксировано."),
          );
          tbody.appendChild(el("tr", { className: "req-detail-row" }, el("td", { colspan: "6" }, detailBox)));
        }
      }
    }

    const skipped = report.skipped_blocks || [];
    const footnote = $("skipped-blocks-summary");
    if (skipped.length === 0) {
      footnote.textContent = "Служебных и пропущенных блоков: 0. Нажмите на строку требования, чтобы раскрыть полную цитату.";
    } else {
      const counts = new Map();
      for (const sb of skipped) {
        counts.set(sb.reason, (counts.get(sb.reason) || 0) + 1);
      }
      const parts = Array.from(counts.entries()).map(
        ([reason, cnt]) => `${SKIPPED_REASON_LABELS[reason] || reason} — ${cnt}`,
      );
      footnote.textContent = `Служебных и пропущенных блоков: ${skipped.length} (${parts.join(", ")}). Нажмите на строку требования, чтобы раскрыть полную цитату.`;
    }
  }

  function buildRequirementStatusChip(req, reqFindings) {
    if (req.check_status === "not_checked") {
      return makeChip("Не проверено", "chip--neutral");
    }
    if (reqFindings.length === 0) {
      return makeChip("✓ Без замечаний", "chip--ok");
    }
    const worst = reqFindings.reduce((acc, item) => {
      return (SEVERITY_RANK[item.severity] ?? 99) < (SEVERITY_RANK[acc] ?? 99) ? item.severity : acc;
    }, "minor");
    const meta = SEVERITY_META[worst] || SEVERITY_META.minor;
    return makeChip(`${meta.icon} ${meta.label} (${reqFindings.length})`, meta.chipClass);
  }

  function toggleRequirementRow(reqId) {
    if (state.expandedReqIds.has(reqId)) {
      state.expandedReqIds.delete(reqId);
    } else {
      state.expandedReqIds.add(reqId);
    }
    if (state.report) {
      renderRequirementsTab(state.report);
    }
  }

  function bucketSetFindings(setFindings) {
    const contradictions = [];
    const crossPart = [];
    const acceptance = [];
    const completeness = [];

    for (const raw of setFindings || []) {
      const sf = Object.assign({}, raw, { _isSet: true });
      if (sf.cross_part || sf.set_check === "cross_part") {
        crossPart.push(sf);
      } else if (sf.set_check === "acceptance_mismatch" || sf.set_check === "acceptance_gap") {
        acceptance.push(sf);
      } else if (
        sf.set_check === "contradiction" ||
        sf.set_check === "terminology" ||
        sf.set_check === "duplication" ||
        sf.criterion === "set_consistent"
      ) {
        contradictions.push(sf);
      } else {
        completeness.push(sf);
      }
    }

    const cmp = (a, b) => (SEVERITY_RANK[a.severity] ?? 99) - (SEVERITY_RANK[b.severity] ?? 99) || a.id.localeCompare(b.id);
    contradictions.sort(cmp);
    crossPart.sort(cmp);
    acceptance.sort(cmp);
    completeness.sort(cmp);

    return { contradictions, crossPart, acceptance, completeness };
  }

  function renderSetCheckTab(report) {
    const root = $("set-check-content");
    clearNode(root);

    const reqById = new Map((report.requirements || []).map((r) => [r.id, r]));
    const { contradictions, crossPart, acceptance, completeness } = bucketSetFindings(report.set_findings || []);

    root.appendChild(
      buildSetSubsection("Противоречия", contradictions, "Противоречий внутри набора не выявлено."),
    );
    root.appendChild(
      buildSetSubsection(
        "Согласованность между частями документа",
        crossPart,
        "Расхождений между частями документа не выявлено.",
      ),
    );

    // Acceptance criteria matrix + findings.
    const accSection = el("section", { className: "set-subsection" });
    accSection.appendChild(
      el(
        "div",
        { className: "subsection-head" },
        el("h3", { className: "subsection-title" }, "Требования и критерии приёмки"),
        makeChip(`строк матрицы: ${(report.coverage || []).length}`, "chip--neutral"),
      ),
    );

    const coverage = report.coverage || [];
    if (coverage.length > 0) {
      const tbody = el("tbody");
      for (const row of coverage) {
        const req = reqById.get(row.requirement_id);
        const reqLabel =
          req && req.doc_id && !req.id_assigned ? `${req.id} (${req.doc_id})` : row.requirement_id;
        const accText = Array.isArray(row.acceptance_ids) && row.acceptance_ids.length > 0
          ? row.acceptance_ids.join(", ")
          : "—";
        const covMeta = COVERAGE_STATUS_META[row.status] || { label: row.status, chipClass: "chip--neutral" };
        tbody.appendChild(
          el(
            "tr",
            null,
            el("td", { className: "mono-cell" }, reqLabel),
            el("td", { className: "mono-cell" }, accText),
            el("td", null, makeChip(covMeta.label, covMeta.chipClass)),
            el("td", null, row.note || "—"),
          ),
        );
      }
      accSection.appendChild(
        el(
          "div",
          { className: "table-wrap" },
          el(
            "table",
            { className: "data-table" },
            el(
              "thead",
              null,
              el(
                "tr",
                null,
                el("th", { scope: "col" }, "Требование"),
                el("th", { scope: "col" }, "Критерии приёмки"),
                el("th", { scope: "col" }, "Статус покрытия"),
                el("th", { scope: "col" }, "Примечание"),
              ),
            ),
            tbody,
          ),
        ),
      );
    } else {
      accSection.appendChild(
        el("div", { className: "empty-state" }, "Матрица покрытия критериями приёмки не сформирована."),
      );
    }

    if (acceptance.length > 0) {
      const list = el("div", { className: "findings-list" });
      for (const sf of acceptance) {
        list.appendChild(buildFindingCard(sf, "set"));
      }
      accSection.appendChild(list);
    }
    root.appendChild(accSection);

    root.appendChild(
      buildSetSubsection("Полнота набора", completeness, "Пробелов полноты на уровне набора не выявлено."),
    );
  }

  function buildSetSubsection(title, items, emptyMessage) {
    const section = el("section", { className: "set-subsection" });
    section.appendChild(
      el(
        "div",
        { className: "subsection-head" },
        el("h3", { className: "subsection-title" }, title),
        makeChip(String(items.length), items.length > 0 ? "chip--accent" : "chip--ok"),
      ),
    );
    if (items.length === 0) {
      section.appendChild(el("div", { className: "empty-state empty-state--ok" }, `✓ ${emptyMessage}`));
    } else {
      const list = el("div", { className: "findings-list" });
      for (const sf of items) {
        list.appendChild(buildFindingCard(sf, "set"));
      }
      section.appendChild(list);
    }
    return section;
  }

  function renderQuestionsTab(report) {
    const container = $("questions-list");
    clearNode(container);

    const questions = report.questions || [];
    if (questions.length === 0) {
      container.appendChild(
        el("div", { className: "empty-state empty-state--ok" }, "✓ Уточняющих вопросов к автору ТЗ нет."),
      );
      return;
    }

    const blocking = questions.filter((q) => q.blocking);
    const regular = questions.filter((q) => !q.blocking);

    if (blocking.length > 0) {
      container.appendChild(buildQuestionGroup(`Блокирующие вопросы (${blocking.length})`, blocking));
    }
    if (regular.length > 0) {
      container.appendChild(buildQuestionGroup(`Прочие вопросы (${regular.length})`, regular));
    }
  }

  function buildQuestionGroup(title, items) {
    const group = el("section", { className: "question-group" }, el("h3", { className: "question-group-title" }, title));
    for (const q of items) {
      const head = el(
        "div",
        { className: "question-head" },
        el("span", { className: "finding-id" }, q.id),
        q.blocking ? makeChip("● Блокирующий", "chip--critical") : makeChip("Уточнение", "chip--neutral"),
      );

      let optionsNode = null;
      if (Array.isArray(q.options) && q.options.length > 0) {
        optionsNode = el(
          "ul",
          { className: "answer-options" },
          ...q.options.map((opt) => el("li", null, opt)),
        );
      }

      const metaParts = [];
      if (q.section) {
        metaParts.push(el("span", null, `Раздел: ${q.section}`));
      }
      if (Array.isArray(q.requirement_ids) && q.requirement_ids.length > 0) {
        metaParts.push(el("span", null, `Требования: ${q.requirement_ids.join(", ")}`));
      }
      if (Array.isArray(q.finding_ids) && q.finding_ids.length > 0) {
        const findingLinks = el("span", null, "Дефекты: ");
        q.finding_ids.forEach((fid, idx) => {
          if (idx > 0) {
            findingLinks.appendChild(document.createTextNode(", "));
          }
          findingLinks.appendChild(
            el(
              "button",
              {
                type: "button",
                className: "fix-link",
                onClick: () => jumpToFinding(fid),
              },
              fid,
            ),
          );
        });
        metaParts.push(findingLinks);
      }

      group.appendChild(
        el(
          "article",
          { className: q.blocking ? "question-card question-card--blocking" : "question-card" },
          head,
          el("p", { className: "question-text" }, q.text || ""),
          optionsNode,
          metaParts.length ? el("div", { className: "question-meta" }, ...metaParts) : null,
        ),
      );
    }
    return group;
  }

  function renderMarkdownTab(markdown) {
    $("markdown-pre").textContent = markdown || "";
  }

  function renderPassportDetails(report) {
    const body = $("passport-details-body");
    clearNode(body);

    const run = report.run || {};
    const fields = [
      ["Документ", run.document_title || "—"],
      ["Файл", run.document_name || "—"],
      ["Формат", run.document_format || "—"],
      ["SHA-256", run.document_sha256 ? `${run.document_sha256.slice(0, 16)}…` : "—"],
      ["Версия", run.agent_version || "—"],
      ["Профиль", run.profile || "—"],
      ["Модели", Array.isArray(run.models_used) && run.models_used.length ? run.models_used.join(", ") : "—"],
      ["Стандарты", Array.isArray(run.standards) && run.standards.length ? run.standards.join(", ") : "—"],
      ["Отброшено цитат", String((report.summary && report.summary.dropped_unverified_quotes) ?? 0)],
      ["Отсеяно пересмотром", String((report.summary && report.summary.dropped_by_review) ?? 0)],
    ];

    const grid = el("div", { className: "passport-grid" });
    for (const [k, v] of fields) {
      grid.appendChild(
        el(
          "div",
          { className: "passport-item" },
          el("span", { className: "passport-key" }, k),
          el("span", { className: "passport-val" }, v),
        ),
      );
    }
    body.appendChild(grid);

    if (Array.isArray(run.stages) && run.stages.length > 0) {
      const tbody = el("tbody");
      for (const st of run.stages) {
        const meta = STAGE_STATE_META[st.state] || STAGE_STATE_META.pending;
        tbody.appendChild(
          el(
            "tr",
            null,
            el("td", null, st.title || st.key),
            el("td", null, makeChip(meta.label, meta.chipClass)),
            el("td", { className: "mono-cell" }, st.model || "—"),
            el("td", { className: "mono-cell tabular" }, `${(st.seconds || 0).toFixed(2)} с`),
            el("td", null, st.detail || "—"),
          ),
        );
      }
      body.appendChild(
        el(
          "div",
          { className: "table-wrap" },
          el(
            "table",
            { className: "data-table" },
            el(
              "thead",
              null,
              el(
                "tr",
                null,
                el("th", { scope: "col" }, "Этап"),
                el("th", { scope: "col" }, "Статус"),
                el("th", { scope: "col" }, "Модель"),
                el("th", { scope: "col" }, "Время"),
                el("th", { scope: "col" }, "Детали"),
              ),
            ),
            tbody,
          ),
        ),
      );
    }
  }

  // Tab navigation with keyboard support.
  const TAB_IDS = ["findings", "requirements", "set", "questions", "markdown"];

  function activateTab(tabName) {
    state.activeTab = tabName;
    for (const name of TAB_IDS) {
      const btn = $(`tab-${name}`);
      const panel = $(`panel-${name}`);
      const active = name === tabName;
      btn.setAttribute("aria-selected", String(active));
      btn.tabIndex = active ? 0 : -1;
      panel.classList.toggle("hidden", !active);
    }
  }

  function jumpToFinding(findingId) {
    state.severityFilter = "all";
    state.criterionFilter = "all";
    state.searchQuery = "";
    $("finding-search").value = "";
    activateTab("findings");
    renderFindingsTab();
    const target = $(`finding-${findingId}`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("is-highlighted");
      setTimeout(() => target.classList.remove("is-highlighted"), 2000);
    }
  }

  async function copyTextToClipboard(text, buttonEl, doneLabel) {
    const original = buttonEl.textContent;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = el("textarea", { className: "sr-only" }, text);
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      buttonEl.textContent = doneLabel;
      setTimeout(() => {
        buttonEl.textContent = original;
      }, 1800);
    } catch (_err) {
      showError("Не удалось скопировать текст в буфер обмена.");
    }
  }

  function formatQuestionsForCopy(questions) {
    const ordered = [...questions].sort((a, b) => Number(Boolean(b.blocking)) - Number(Boolean(a.blocking)));
    return ordered
      .map((q) => {
        const prefix = q.blocking ? "[Блокирующий] " : "";
        const lines = [`${q.id}. ${prefix}${q.text}`];
        if (q.section) {
          lines.push(`   Раздел: ${q.section}`);
        }
        if (Array.isArray(q.options) && q.options.length > 0) {
          lines.push(`   Варианты: ${q.options.join(" / ")}`);
        }
        return lines.join("\n");
      })
      .join("\n\n");
  }

  // Event bindings.
  function bindEvents() {
    const dropZone = $("drop-zone");
    const fileInput = $("file-input");

    $("select-file-btn").addEventListener("click", (ev) => {
      ev.stopPropagation();
      fileInput.click();
    });

    $("sample-btn").addEventListener("click", (ev) => {
      ev.stopPropagation();
      submitSample();
    });

    dropZone.addEventListener("click", (ev) => {
      if (ev.target.closest("button")) {
        return;
      }
      fileInput.click();
    });

    dropZone.addEventListener("keydown", (ev) => {
      if (ev.target !== dropZone) {
        return;
      }
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        fileInput.click();
      }
    });

    dropZone.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      dropZone.classList.add("is-dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("is-dragover");
    });

    dropZone.addEventListener("drop", (ev) => {
      ev.preventDefault();
      dropZone.classList.remove("is-dragover");
      const files = ev.dataTransfer && ev.dataTransfer.files;
      if (files && files.length > 0) {
        submitFile(files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        submitFile(fileInput.files[0]);
      }
    });

    $("error-dismiss-btn").addEventListener("click", hideError);
    $("new-check-btn").addEventListener("click", resetToUpload);

    // Tab clicks and arrow-key navigation.
    for (const name of TAB_IDS) {
      const btn = $(`tab-${name}`);
      btn.addEventListener("click", () => activateTab(name));
      btn.addEventListener("keydown", (ev) => {
        const idx = TAB_IDS.indexOf(name);
        let next = null;
        if (ev.key === "ArrowRight") {
          next = TAB_IDS[(idx + 1) % TAB_IDS.length];
        } else if (ev.key === "ArrowLeft") {
          next = TAB_IDS[(idx - 1 + TAB_IDS.length) % TAB_IDS.length];
        } else if (ev.key === "Home") {
          next = TAB_IDS[0];
        } else if (ev.key === "End") {
          next = TAB_IDS[TAB_IDS.length - 1];
        }
        if (next) {
          ev.preventDefault();
          activateTab(next);
          $(`tab-${next}`).focus();
        }
      });
    }

    // Severity filter pills.
    for (const pill of document.querySelectorAll(".filter-pill[data-severity]")) {
      pill.addEventListener("click", () => {
        state.severityFilter = pill.dataset.severity || "all";
        renderFindingsTab();
      });
    }

    // Criterion select and search input.
    $("criterion-filter").addEventListener("change", (ev) => {
      state.criterionFilter = ev.target.value || "all";
      renderFindingsTab();
    });

    $("finding-search").addEventListener("input", (ev) => {
      state.searchQuery = ev.target.value || "";
      renderFindingsTab();
    });

    // Clickable summary metric cards.
    for (const card of document.querySelectorAll(".metric-card[data-metric-action]")) {
      card.addEventListener("click", () => {
        const action = card.dataset.metricAction;
        if (action === "requirements") {
          activateTab("requirements");
        } else if (action === "questions") {
          activateTab("questions");
        } else if (action === "critical" || action === "major" || action === "minor") {
          state.severityFilter = action;
          state.criterionFilter = "all";
          activateTab("findings");
          renderFindingsTab();
        }
      });
    }

    // Copy buttons.
    $("copy-questions-btn").addEventListener("click", () => {
      const questions = (state.report && state.report.questions) || [];
      if (questions.length === 0) {
        return;
      }
      copyTextToClipboard(formatQuestionsForCopy(questions), $("copy-questions-btn"), "Скопировано ✓");
    });

    $("copy-markdown-btn").addEventListener("click", () => {
      if (!state.markdown) {
        return;
      }
      copyTextToClipboard(state.markdown, $("copy-markdown-btn"), "Скопировано ✓");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    fetchHealth();
  });
})();
