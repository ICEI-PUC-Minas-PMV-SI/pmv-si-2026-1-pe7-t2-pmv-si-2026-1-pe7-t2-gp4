(function () {
  const MAX_PATIENTS = 100;
  const tbody = document.getElementById("agenda-body");
  const errorsBox = document.getElementById("agenda-errors");
  const resultBox = document.getElementById("batch-result");
  const calculateButton = document.getElementById("btn-calculate-batch");

  const examplePatients = [
    { age: 35, waiting_days: 20, gender: "F", scholarship: 0, hypertension: 0, diabetes: 0, alcoholism: 0, handicap: 0, sms_received: 1 },
    { age: 42, waiting_days: 8, gender: "M", scholarship: 1, hypertension: 1, diabetes: 0, alcoholism: 0, handicap: 0, sms_received: 0 },
    { age: 28, waiting_days: 30, gender: "F", scholarship: 0, hypertension: 0, diabetes: 0, alcoholism: 0, handicap: 0, sms_received: 1 },
    { age: 55, waiting_days: 5, gender: "M", scholarship: 0, hypertension: 1, diabetes: 1, alcoholism: 0, handicap: 1, sms_received: 0 },
    { age: 19, waiting_days: 18, gender: "F", scholarship: 1, hypertension: 0, diabetes: 0, alcoholism: 0, handicap: 0, sms_received: 1 },
  ];

  function defaultPatient() {
    return {
      age: 30,
      waiting_days: 10,
      gender: "F",
      scholarship: 0,
      hypertension: 0,
      diabetes: 0,
      alcoholism: 0,
      handicap: 0,
      sms_received: 0,
    };
  }

  function showErrors(message) {
    errorsBox.hidden = false;
    errorsBox.textContent = message;
    resultBox.hidden = true;
  }

  function clearErrors() {
    errorsBox.hidden = true;
    errorsBox.textContent = "";
  }

  function createRow(patient, index) {
    const row = document.createElement("tr");
    row.innerHTML =
      '<td>' + index + "</td>" +
      '<td><input type="number" min="0" max="120" data-field="age" value="' + patient.age + '"></td>' +
      '<td><input type="number" min="0" max="200" data-field="waiting_days" value="' + patient.waiting_days + '"></td>' +
      '<td><select data-field="gender"><option value="F"' + (patient.gender === "F" ? " selected" : "") + '>F</option><option value="M"' + (patient.gender === "M" ? " selected" : "") + ">M</option></select></td>" +
      fieldSelect("scholarship", patient.scholarship) +
      fieldSelect("hypertension", patient.hypertension) +
      fieldSelect("diabetes", patient.diabetes) +
      fieldSelect("alcoholism", patient.alcoholism) +
      '<td><input type="number" min="0" max="4" data-field="handicap" value="' + patient.handicap + '"></td>' +
      fieldSelect("sms_received", patient.sms_received) +
      '<td><button type="button" class="btn btn-secondary btn-remove">Remover</button></td>';

    row.querySelector(".btn-remove").addEventListener("click", function () {
      row.remove();
      renumberRows();
    });

    return row;
  }

  function fieldSelect(name, value) {
    return (
      '<td><select data-field="' + name + '">' +
      '<option value="0"' + (Number(value) === 0 ? " selected" : "") + ">0</option>" +
      '<option value="1"' + (Number(value) === 1 ? " selected" : "") + ">1</option>" +
      "</select></td>"
    );
  }

  function renumberRows() {
    Array.from(tbody.querySelectorAll("tr")).forEach(function (row, index) {
      row.children[0].textContent = String(index + 1);
    });
  }

  function addPatient(patient) {
    if (tbody.children.length >= MAX_PATIENTS) {
      showErrors("A agenda aceita no máximo 100 pacientes.");
      return;
    }
    clearErrors();
    tbody.appendChild(createRow(patient || defaultPatient(), tbody.children.length + 1));
  }

  function readPatients() {
    const patients = [];
    Array.from(tbody.querySelectorAll("tr")).forEach(function (row) {
      const getValue = function (field) {
        const element = row.querySelector('[data-field="' + field + '"]');
        return element ? element.value : "";
      };

      patients.push({
        age: Number(getValue("age")),
        waiting_days: Number(getValue("waiting_days")),
        gender: getValue("gender"),
        scholarship: Number(getValue("scholarship")),
        hypertension: Number(getValue("hypertension")),
        diabetes: Number(getValue("diabetes")),
        alcoholism: Number(getValue("alcoholism")),
        handicap: Number(getValue("handicap")),
        sms_received: Number(getValue("sms_received")),
      });
    });
    return patients;
  }

  function renderBatchResult(data) {
    const policy = data.policy;
    document.getElementById("batch-count").textContent = policy.patient_count;
    document.getElementById("batch-expected").textContent = policy.expected_absences.toFixed(2);
    document.getElementById("batch-preliminary").textContent = policy.preliminary_slots;
    document.getElementById("batch-cap").textContent = policy.hard_cap;
    document.getElementById("batch-recommendation").textContent = policy.recommended_extra_slots;
    document.getElementById("batch-formula").textContent = data.formula_explanation || "";
    document.getElementById("batch-disclaimer").textContent = data.disclaimer || "";

    const detailsBody = document.getElementById("batch-details-body");
    detailsBody.innerHTML = "";
    data.patients.forEach(function (item) {
      const row = document.createElement("tr");
      row.innerHTML =
        "<td>" + item.index + "</td>" +
        "<td>" + item.input.age + "</td>" +
        "<td>" + item.input.waiting_days + "</td>" +
        "<td>" + item.probability_percent + "%</td>" +
        "<td>" + item.risk_band + "</td>";
      detailsBody.appendChild(row);
    });

    resultBox.hidden = false;
  }

  document.getElementById("btn-add-patient").addEventListener("click", function () {
    addPatient();
  });

  document.getElementById("btn-load-example").addEventListener("click", function () {
    tbody.innerHTML = "";
    examplePatients.forEach(function (patient, index) {
      tbody.appendChild(createRow(patient, index + 1));
    });
    clearErrors();
    resultBox.hidden = true;
  });

  calculateButton.addEventListener("click", async function () {
    clearErrors();
    const patients = readPatients();
    if (!patients.length) {
      showErrors("Adicione ao menos um paciente à agenda.");
      return;
    }

    const safetyFactor = Number(document.getElementById("safety_factor").value);
    const maxExtraPercentage = Number(document.getElementById("max_extra_percentage").value) / 100;

    window.SUSApp.setButtonLoading(calculateButton, true);
    try {
      const data = await window.SUSApp.postJson("/api/predict-batch", {
        patients: patients,
        safety_factor: safetyFactor,
        max_extra_percentage: maxExtraPercentage,
      });
      renderBatchResult(data);
    } catch (error) {
      showErrors(error.message);
    } finally {
      window.SUSApp.setButtonLoading(calculateButton, false);
    }
  });

  addPatient();
})();
