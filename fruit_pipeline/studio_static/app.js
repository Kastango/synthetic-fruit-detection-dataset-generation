"use strict";
const $ = (s) => document.querySelector(s);
let spec,
  controls = {},
  current = null,
  index = 0,
  detail = false,
  revision = 0,
  busy = false,
  queued = false,
  timer,
  jobId = null,
  jobTimer;
function status(text, error = false) {
  $("#status").textContent = text;
  $("#status").classList.toggle("error", error);
}
function defaults() {
  return spec.defaults;
}
function buildControls() {
  const root = $("#controls");
  root.replaceChildren();
  for (const c of spec.controls) {
    const row = document.createElement("div");
    row.className = "control";
    const top = document.createElement("div");
    top.className = "control-top";
    const label = document.createElement("label");
    label.htmlFor = c.key;
    label.textContent = c.label;
    label.title = c.help;
    const out = document.createElement("output");
    out.htmlFor = c.key;
    out.textContent = controls[c.key];
    top.append(label, out);
    const input = document.createElement("input");
    Object.assign(input, {
      type: "range",
      id: c.key,
      min: c.min,
      max: c.max,
      step: c.step,
      value: controls[c.key],
    });
    input.setAttribute("aria-description", c.help);
    input.oninput = () => {
      controls[c.key] = Number(input.value);
      out.textContent = input.value;
      schedule();
    };
    row.append(top, input);
    root.append(row);
  }
}
function enable(value) {
  for (const id of ["export", "report", "generate"])
    $("#" + id).disabled = !value;
}
function schedule() {
  revision++;
  enable(false);
  status("Atualizando…");
  clearTimeout(timer);
  timer = setTimeout(request, 450);
}
async function api(url, body) {
  const response = await fetch(
    url,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const value = await response.json();
  if (!response.ok)
    throw Error(value.error || "Falha na geração. Tente novamente.");
  return value;
}
async function request() {
  if (busy) {
    queued = true;
    return;
  }
  busy = true;
  queued = false;
  const version = revision;
  const body = {
    controls: { ...controls },
    preset: "reference",
    seed: Number($("#seed").value),
    index,
  };
  status("Gerando imagens…");
  try {
    const value = await api("/api/preview", body);
    if (version === revision) {
      current = value;
      current.recipe = body;
      paint();
      enable(true);
      status("Imagens atualizadas");
    }
  } catch (error) {
    if (version === revision) status(error.message, true);
  } finally {
    busy = false;
    if (queued || version !== revision) request();
  }
}
function paint() {
  const boxes = $("#boxes").checked,
    background = $("#background").checked;
  $("#scenes").replaceChildren(
    ...current.synthetic.map((scene, i) => {
      const card = document.createElement("article");
      card.className = "image-card";
      const head = document.createElement("div");
      head.className = "card-head";
      const title = document.createElement("h2");
      title.textContent = `Cena ${index + i + 1}`;
      const count = document.createElement("span");
      count.className = "small";
      count.textContent = background
        ? "Fundo original"
        : `${scene.count} caixas`;
      head.append(title, count);
      card.append(head);
      if (detail && !background) {
        const crops = document.createElement("div");
        crops.className = "crops";
        for (const [j, src] of scene.crops.entries()) {
          const im = document.createElement("img");
          im.src = src;
          im.alt = `Fruta ${j + 1} da cena ${i + 1}`;
          crops.append(im);
        }
        if (!scene.crops.length)
          crops.textContent = "Nenhuma caixa nesta cena.";
        card.append(crops);
      } else {
        const button = document.createElement("button");
        button.className = "image-button";
        button.setAttribute("aria-label", `Ampliar cena ${index + i + 1}`);
        const img = document.createElement("img");
        img.src = background
          ? scene.background
          : boxes
            ? scene.annotated
            : scene.image;
        img.alt = background
          ? "Árvore antes da inserção das poncãs"
          : `Árvore com ${scene.count} poncãs anotadas`;
        button.append(img);
        button.onclick = () => {
          $("#zoom-image").src = img.src;
          $("#zoom").showModal();
        };
        card.append(button);
      }
      return card;
    }),
  );
  $("#metrics").replaceChildren(
    ...current.metrics.map((row) => {
      const tr = document.createElement("tr");
      for (const text of [
        row.label,
        ...(row.quantiles
          ? row.quantiles.map((v) => v.toFixed(2))
          : ["—", "—", "—"]),
      ]) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }),
  );
  $("#sample-note").textContent =
    `${current.sample_images} imagens, ${current.sample_boxes} caixas. Tamanho, posição e aparência em %; contraste em log₂.`;
  $("#fingerprint").textContent =
    `Semente ${current.recipe.seed}. Receita ${current.config_hash.slice(0, 12)}.`;
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("#export").onclick = () => {
  if (current)
    download("studio_candidate.yaml", current.yaml, "application/yaml");
};
$("#report").onclick = () => {
  if (current) {
    const { synthetic, ...report } = current;
    download(
      "studio_report.json",
      JSON.stringify(report, null, 2),
      "application/json",
    );
  }
};
$("#reset").onclick = () => {
  controls = { ...defaults() };
  buildControls();
  schedule();
};
$("#seed").onchange = () => {
  index = 0;
  schedule();
};
$("#next").onclick = () => {
  index += current?.sample_images || 8;
  schedule();
};
$("#boxes").onchange = () => current && paint();
$("#background").onchange = () => current && paint();
function mode(v) {
  detail = v;
  $("#scene-tab").classList.toggle("selected", !v);
  $("#crop-tab").classList.toggle("selected", v);
  if (current) paint();
}
$("#scene-tab").onclick = () => mode(false);
$("#crop-tab").onclick = () => mode(true);
$("#close-zoom").onclick = () => $("#zoom").close();
$("#mobile-controls").onclick = () => {
  const open = document.body.classList.toggle("controls-open");
  $("#mobile-controls").textContent = open
    ? "Fechar ajustes"
    : "Ajustar receita";
  $("#mobile-controls").setAttribute("aria-expanded", String(open));
};
$("#generate").onclick = () => $("#generation").showModal();
$("#close-generation").onclick = () => $("#generation").close();
async function pollJob() {
  clearTimeout(jobTimer);
  try {
    const job = await api("/api/jobs/" + jobId);
    $("#job-progress").hidden = false;
    $("#job-progress").max = job.total;
    $("#job-progress").value = job.completed;
    if (job.status === "complete") {
      $("#job-status").textContent = `Dataset pronto: ${job.total} imagens.`;
      $("#download-dataset").href = job.download;
      $("#download-dataset").hidden = false;
      $("#start-generation").disabled = false;
    } else if (job.status === "error") {
      $("#job-status").textContent = job.error;
      $("#start-generation").disabled = false;
    } else {
      $("#job-status").textContent =
        job.completed === job.total
          ? "Preparando o ZIP…"
          : `Gerando ${job.completed} de ${job.total} imagens…`;
      jobTimer = setTimeout(pollJob, 2000);
    }
  } catch (e) {
    $("#job-status").textContent = e.message;
    $("#start-generation").disabled = false;
  }
}
$("#generation-form").onsubmit = async (event) => {
  event.preventDefault();
  if (!current) return;
  $("#start-generation").disabled = true;
  $("#download-dataset").hidden = true;
  $("#job-status").textContent = "Iniciando geração…";
  try {
    const job = await api("/api/generate", {
      ...current.recipe,
      total: Number($("#total").value),
      train_ratio: Number($("#train-ratio").value) / 100,
    });
    jobId = job.id;
    pollJob();
  } catch (e) {
    $("#job-status").textContent = e.message;
    $("#start-generation").disabled = false;
  }
};
(async () => {
  try {
    spec = await api("/api/controls");
    controls = { ...defaults() };
    buildControls();
    await request();
  } catch (e) {
    status(
      "Não foi possível iniciar. Verifique o servidor e recarregue a página.",
      true,
    );
  }
})();
