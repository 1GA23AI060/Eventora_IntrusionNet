const apiStatus = document.querySelector("#apiStatus");
const jsonInput = document.querySelector("#jsonInput");
const csvInput = document.querySelector("#csvInput");
const fileName = document.querySelector("#fileName");
const predictJson = document.querySelector("#predictJson");
const predictCsv = document.querySelector("#predictCsv");
const normalCount = document.querySelector("#normalCount");
const attackCount = document.querySelector("#attackCount");
const totalCount = document.querySelector("#totalCount");
const riskScore = document.querySelector("#riskScore");
const normalBar = document.querySelector("#normalBar");
const attackBar = document.querySelector("#attackBar");
const confidenceBar = document.querySelector("#confidenceBar");
const attackRateLabel = document.querySelector("#attackRateLabel");
const confidenceLabel = document.querySelector("#confidenceLabel");
const thresholdStatus = document.querySelector("#thresholdStatus");
const resultsBody = document.querySelector("#resultsBody");
const resultSummary = document.querySelector("#resultSummary");
const message = document.querySelector("#message");
const setupPanel = document.querySelector("#setupPanel");
let modelReady = false;

function setLoading(isLoading) {
  predictJson.disabled = isLoading || !modelReady;
  predictCsv.disabled = isLoading || !modelReady;
}

function showMessage(text) {
  message.textContent = text;
  message.classList.toggle("show", Boolean(text));
}

function updateDashboard(summary = { normal: 0, attack: 0, total: 0 }) {
  const total = summary.total ?? 0;
  const normal = summary.normal ?? 0;
  const attack = summary.attack ?? 0;
  const attackRate = total ? attack / total : 0;
  const highConfidence = summary.high_confidence_attacks ?? 0;
  const highConfidenceRate = total ? highConfidence / total : 0;
  const avgRisk = summary.average_attack_probability ?? 0;

  normalCount.textContent = normal;
  attackCount.textContent = attack;
  totalCount.textContent = total;
  riskScore.textContent = `${Math.round(avgRisk * 100)}%`;
  normalBar.style.width = `${Math.max(0, Math.min(100, (1 - attackRate) * 100))}%`;
  attackBar.style.width = `${Math.max(0, Math.min(100, attackRate * 100))}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, highConfidenceRate * 100))}%`;
  attackRateLabel.textContent = `${Math.round(attackRate * 100)}% attack rate`;
  confidenceLabel.textContent = `${highConfidence} event${highConfidence === 1 ? "" : "s"}`;

  if (typeof summary.threshold === "number") {
    thresholdStatus.textContent = `Threshold ${summary.threshold.toFixed(3)}`;
  }
}

function renderResults(data) {
  const rows = data.results || [];
  updateDashboard(data.summary);
  resultSummary.textContent = rows.length ? `${rows.length} prediction(s)` : "No predictions";
  resultsBody.innerHTML = rows
    .slice(0, 500)
    .map((row) => {
      const kind = row.prediction === "Attack" ? "attack" : "normal";
      return `
        <tr>
          <td>${row.row}</td>
          <td><span class="badge ${kind}">${row.prediction}</span></td>
          <td><span class="severity ${String(row.severity).toLowerCase()}">${row.severity}</span></td>
          <td>${Number(row.attack_probability).toFixed(4)}</td>
        </tr>
      `;
    })
    .join("");

  if (rows.length > 500) {
    showMessage("Showing first 500 rows. The dashboard counts include all predictions.");
  }
}

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.details || "Request failed");
  }
  return data;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await parseResponse(response);
    modelReady = Boolean(data.model_ready);
    apiStatus.textContent = modelReady ? "Model ready" : "Train model first";
    apiStatus.className = `status-pill ${modelReady ? "ready" : "warning"}`;
    thresholdStatus.textContent =
      modelReady && typeof data.threshold === "number"
        ? `Threshold ${data.threshold.toFixed(3)}`
        : "Threshold --";
    setupPanel.classList.toggle("show", !modelReady);
    setLoading(false);

    if (!modelReady) {
      const missing = data.missing_artifacts?.join(", ") || "model artifacts";
      showMessage(`Training required before prediction. Missing: ${missing}.`);
    }
  } catch (error) {
    modelReady = false;
    apiStatus.textContent = "API unavailable";
    apiStatus.className = "status-pill warning";
    setupPanel.classList.add("show");
    setLoading(false);
  }
}

predictJson.addEventListener("click", async () => {
  showMessage("");
  if (!modelReady) {
    showMessage("Train the model before running predictions.");
    return;
  }
  setLoading(true);
  try {
    const features = JSON.parse(jsonInput.value);
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    renderResults(await parseResponse(response));
  } catch (error) {
    showMessage(error.message);
  } finally {
    setLoading(false);
  }
});

predictCsv.addEventListener("click", async () => {
  showMessage("");
  if (!modelReady) {
    showMessage("Train the model before running predictions.");
    return;
  }
  if (!csvInput.files.length) {
    showMessage("Choose a CSV file first.");
    return;
  }

  setLoading(true);
  try {
    const formData = new FormData();
    formData.append("file", csvInput.files[0]);
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    renderResults(await parseResponse(response));
  } catch (error) {
    showMessage(error.message);
  } finally {
    setLoading(false);
  }
});

csvInput.addEventListener("change", () => {
  fileName.textContent = csvInput.files[0]?.name || "No file selected";
});

checkHealth();
