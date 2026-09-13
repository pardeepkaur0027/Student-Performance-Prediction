const form = document.getElementById("predict-form");
const btn = document.getElementById("predict-btn");
const btnText = document.getElementById("predict-btn-text");
const resultBox = document.getElementById("result-box");
const resultCard = document.getElementById("result-card");
const resultEmpty = document.getElementById("result-empty");
const busy = document.getElementById("busy");

const MODEL_READY = document.body.dataset.modelReady === "true";

function setBusy(isBusy) {
  btn.disabled = isBusy;
  btnText.textContent = isBusy ? "Predicting..." : "Predict Student Performance";
  busy.classList.toggle("hidden", !isBusy);
  if (isBusy) {
    resultBox.classList.add("hidden");
    if (resultEmpty) resultEmpty.classList.remove("hidden");
  }
}

function renderResult(data) {
  resultBox.classList.remove("hidden");
  if (resultEmpty) resultEmpty.classList.add("hidden");

  const isPass = data.prediction === "PASS";
  const color = isPass ? "emerald" : "rose";
  const riskPalette = {
    "Low Risk": { bg: "bg-emerald-500/10", border: "border-emerald-500/40", text: "text-emerald-300", dot: "bg-emerald-400" },
    "Medium Risk": { bg: "bg-amber-500/10", border: "border-amber-500/40", text: "text-amber-300", dot: "bg-amber-400" },
    "High Risk": { bg: "bg-rose-500/10", border: "border-rose-500/40", text: "text-rose-300", dot: "bg-rose-400" },
  };
  const risk = riskPalette[data.risk_level];

  resultCard.className = `relative overflow-hidden rounded-2xl border p-6 md:p-8 backdrop-blur ${risk.bg} ${risk.border} bg-slate-900/60`;

  const pct = data.probability;
  const pctClass = pct >= 75 ? "text-emerald-300" : pct >= 45 ? "text-amber-300" : "text-rose-300";

  resultCard.innerHTML = `
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <div>
        <p class="text-xs uppercase tracking-widest text-slate-400">Predicted Result</p>
        <span class="mt-1 inline-flex items-center gap-2 text-3xl md:text-4xl font-bold text-${color}-400">
          <svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            ${isPass
              ? `<circle cx="12" cy="12" r="9"/><path stroke-linecap="round" stroke-linejoin="round" d="m9 12 2 2 4-4"/>`
              : `<circle cx="12" cy="12" r="9"/><path stroke-linecap="round" stroke-linejoin="round" d="m15 9-6 6M9 9l6 6"/>`}
          </svg>
          ${data.prediction}
        </span>
      </div>
      <div class="text-right">
        <p class="text-xs uppercase tracking-widest text-slate-400">Risk Level</p>
        <span class="mt-1 inline-flex items-center gap-2 text-xl font-semibold ${risk.text}">
          <span class="w-2.5 h-2.5 rounded-full ${risk.dot}"></span>
          ${data.risk_level}
        </span>
      </div>
    </div>

    <div class="mt-6">
      <div class="flex items-center justify-between text-sm mb-2">
        <span class="text-slate-400">Pass Probability</span>
        <span class="font-semibold ${pctClass}">${data.probability}%</span>
      </div>
      <div class="h-2.5 w-full rounded-full bg-slate-800 overflow-hidden">
        <div class="h-full rounded-full ${isPass ? "bg-emerald-400" : pct >= 45 ? "bg-amber-400" : "bg-rose-400"} transition-all duration-700"
             style="width: ${data.probability}%"></div>
      </div>
    </div>

    <div class="mt-6 ${risk.border} border rounded-xl p-4 text-sm ${risk.text}">
      ${data.risk_level === "High Risk"
        ? "This student is at <b>High Risk</b> of failing. Suggested actions: improve attendance, increase study hours, complete assignments on time, seek mentoring support."
        : data.risk_level === "Medium Risk"
        ? "This student is at <b>Medium Risk</b>. Regular monitoring is recommended."
        : "This student is at <b>Low Risk</b>. Performance looks good!"}
    </div>
  `;

  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderError(message) {
  resultBox.classList.remove("hidden");
  if (resultEmpty) resultEmpty.classList.add("hidden");
  resultCard.className = "rounded-2xl border border-rose-500/40 bg-rose-500/10 backdrop-blur p-6 text-rose-200 bg-slate-900/60";
  resultCard.innerHTML = `
    <div class="flex items-start gap-3">
      <svg class="w-6 h-6 shrink-0 text-rose-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="9"/><path stroke-linecap="round" d="M12 8v4M12 16h.01"/>
      </svg>
      <div>
        <p class="font-semibold">Prediction failed</p>
        <p class="mt-1 text-sm opacity-90">${message}</p>
      </div>
    </div>`;
  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!MODEL_READY) {
      renderError("Model files are not available on the server. Please check best_student_model.pkl and scaler.pkl.");
      return;
    }

    setBusy(true);
    try {
      const payload = Object.fromEntries(new FormData(form).entries());
      const resp = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!data.ok) {
        renderError(data.error || "Unknown error");
      } else {
        renderResult(data);
      }
    } catch (err) {
      renderError(err.message || String(err));
    } finally {
      setBusy(false);
    }
  });
}

document.querySelectorAll("#predict-form input[type='range']").forEach((slider) => {
  const output = document.getElementById(slider.id + "-out");
  const update = () => {
    if (output) output.textContent = slider.value;
  };
  slider.addEventListener("input", update);
  update();
});