// Process page: upload (or use the sample), show the report, offer downloads.
// Uses the shared table helper from charts.js.
(() => {
  const form = document.getElementById("upload-form");
  const input = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const fileName = document.getElementById("file-name");
  const processButton = document.getElementById("process-button");
  const sampleButton = document.getElementById("sample-button");
  const errorBox = document.getElementById("error");
  const results = document.getElementById("results");
  const { integer, table } = HotelCharts;

  const setBusy = (button, busy, label) => {
    processButton.disabled = busy;
    sampleButton.disabled = busy;
    button.innerHTML = busy ? `<span class="spinner" aria-hidden="true"></span> Processing…` : label;
  };

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  };

  const render = ({ token, summary }) => {
    document.getElementById("result-title").textContent = `Result: ${summary.file}`;
    document.getElementById("result-note").textContent =
      `${integer.format(summary.rows_in - summary.rows_out)} rows removed during cleaning. ` +
      "The model-ready file is scaled and one-hot encoded; the cleaned file is human-readable.";
    document.getElementById("stat-rows").textContent = `${integer.format(summary.rows_in)} → ${integer.format(summary.rows_out)}`;
    document.getElementById("stat-columns").textContent = `${summary.columns_in} → ${summary.columns_out}`;
    document.getElementById("stat-missing").textContent = `${integer.format(summary.missing_in)} → ${integer.format(summary.missing_out)}`;
    document.getElementById("stat-target").textContent = summary.has_target ? "Kept" : "Not provided";
    document.getElementById("download-model").href = `/download/${token}/model-ready`;
    document.getElementById("download-clean").href = `/download/${token}/clean`;

    table(
      document.getElementById("steps-table"),
      [
        { key: "step", label: "Step" },
        { key: "rows_removed", label: "Rows removed", numeric: true, format: (v) => integer.format(v) },
        { key: "values_changed", label: "Values changed", numeric: true, format: (v) => integer.format(v) },
      ],
      summary.steps,
    );
    table(
      document.getElementById("preview-table"),
      summary.preview_columns.map((name) => ({ key: name, label: name.replace(/^[a-z]+__/, ""), numeric: true })),
      summary.preview,
    );
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const submit = async (url, body, button, label) => {
    showError("");
    setBusy(button, true, label);
    try {
      const response = await fetch(url, { method: "POST", body });
      const payload = await response.json().catch(() => ({ error: `Server error (${response.status}).` }));
      if (!response.ok) throw new Error(payload.error || `Server error (${response.status}).`);
      render(payload);
    } catch (error) {
      results.hidden = true;
      showError(error.message);
    } finally {
      setBusy(button, false, label);
    }
  };

  const showFile = () => {
    fileName.textContent = input.files.length ? input.files[0].name : "Drop a CSV here or click to browse";
  };

  input.addEventListener("change", showFile);

  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("is-dragging"));
  }
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      showFile();
    }
  });

  const processLabel = processButton.textContent;
  const sampleLabel = sampleButton.textContent;

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!input.files.length) {
      showError("Choose a CSV file first - or try the sample.");
      return;
    }
    const body = new FormData();
    body.append("file", input.files[0]);
    submit("/api/process", body, processButton, processLabel);
  });

  sampleButton.addEventListener("click", () => submit("/api/process-sample", null, sampleButton, sampleLabel));
})();
