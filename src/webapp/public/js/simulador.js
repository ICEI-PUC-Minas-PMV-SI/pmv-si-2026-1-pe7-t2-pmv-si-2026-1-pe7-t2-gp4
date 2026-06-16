(function () {
  const form = document.getElementById("form-simulador");
  const errorsBox = document.getElementById("form-errors");
  const resultBox = document.getElementById("resultado");
  const submitButton = document.getElementById("btn-calcular");
  const exampleButton = document.getElementById("btn-exemplo");

  const examplePayload = {
    age: 35,
    waiting_days: 20,
    gender: "F",
    scholarship: 0,
    hypertension: 0,
    diabetes: 0,
    alcoholism: 0,
    handicap: 0,
    sms_received: 1,
  };

  function showErrors(message) {
    errorsBox.hidden = false;
    errorsBox.textContent = message;
    resultBox.hidden = true;
  }

  function clearErrors() {
    errorsBox.hidden = true;
    errorsBox.textContent = "";
  }

  function readForm() {
    const formData = new FormData(form);
    return {
      age: Number(formData.get("age")),
      waiting_days: Number(formData.get("waiting_days")),
      gender: formData.get("gender"),
      scholarship: Number(formData.get("scholarship")),
      hypertension: Number(formData.get("hypertension")),
      diabetes: Number(formData.get("diabetes")),
      alcoholism: Number(formData.get("alcoholism")),
      handicap: Number(formData.get("handicap")),
      sms_received: Number(formData.get("sms_received")),
    };
  }

  function fillExample() {
    Object.entries(examplePayload).forEach(function ([key, value]) {
      const field = form.elements.namedItem(key);
      if (field) field.value = String(value);
    });
    clearErrors();
  }

  function renderResult(data) {
    document.getElementById("result-message").textContent = data.message;
    document.getElementById("result-percent").textContent = data.probability_percent + "%";
    document.getElementById("result-band").textContent = data.risk_band;
    document.getElementById("result-model").textContent = data.model_name || "—";
    document.getElementById("result-version").textContent = data.model_version || "—";
    document.getElementById("result-disclaimer").textContent = data.disclaimer || "";

    const fill = document.getElementById("risk-meter-fill");
    fill.style.width = data.probability_percent + "%";
    fill.className = "risk-meter-fill " + window.SUSApp.riskClass(data.risk_band);

    resultBox.hidden = false;
  }

  if (exampleButton) {
    exampleButton.addEventListener("click", fillExample);
  }

  if (form) {
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      clearErrors();
      window.SUSApp.setButtonLoading(submitButton, true);

      try {
        const payload = readForm();
        const data = await window.SUSApp.postJson("/api/predict", payload);
        renderResult(data);
      } catch (error) {
        showErrors(error.message);
      } finally {
        window.SUSApp.setButtonLoading(submitButton, false);
      }
    });

    form.addEventListener("reset", function () {
      clearErrors();
      resultBox.hidden = true;
    });
  }
})();
