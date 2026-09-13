const estado = { indice: 0, total: 0, marcas: [], selecionadas: new Set(), previsoes: true };

const $ = (id) => document.getElementById(id);

async function iniciar() {
  const setup = await (await fetch("/api/setup")).json();
  estado.marcas = setup.marcas;
  estado.total = setup.total;
  $("legenda-pred").hidden = !setup.tem_detector;
  desenharMarcas();
  await carregar(0);
}

function desenharMarcas() {
  $("marcas").innerHTML = "";
  estado.marcas.forEach((marca, i) => {
    const item = document.createElement("li");
    item.dataset.chave = marca.chave;
    item.innerHTML = `<b>${i + 1}</b> ${marca.chave}<small>${marca.rotulo}</small>`;
    item.onclick = () => alternar(marca.chave);
    $("marcas").appendChild(item);
  });
}

function alternar(chave) {
  // "ok" quer dizer "nada a reparar": conviver com outra marca seria mentira.
  if (chave === "ok") {
    const tinha = estado.selecionadas.has("ok");
    estado.selecionadas.clear();
    if (!tinha) estado.selecionadas.add("ok");
  } else {
    estado.selecionadas.delete("ok");
    estado.selecionadas.has(chave)
      ? estado.selecionadas.delete(chave)
      : estado.selecionadas.add(chave);
  }
  pintarMarcas();
}

function pintarMarcas() {
  for (const item of $("marcas").children) {
    item.classList.toggle("ativa", estado.selecionadas.has(item.dataset.chave));
  }
}

async function carregar(indice) {
  if (indice < 0 || indice >= estado.total) return;
  const sufixo = estado.previsoes ? "" : "?previsoes=0";
  const item = await (await fetch(`/api/item/${indice}${sufixo}`)).json();
  estado.indice = indice;
  $("figura").src = item.figura;
  $("identidade").innerHTML =
    `<div class="nome">${item.imagem}</div>` +
    `<div class="meta">${item.caixas} caixas · ${item.resolucao}</div>`;
  const d = item.discordancia;
  $("discordancia").innerHTML = d
    ? `<div class="meta">detector: ${d.previstas} caixas · ` +
      `${d.sem_par} do gabarito sem par · ${d.sobrando} previstas sem par</div>`
    : "";
  estado.selecionadas = new Set(item.veredicto ? item.veredicto.marcas : []);
  $("nota").value = item.veredicto ? item.veredicto.nota : "";
  pintarMarcas();
  atualizarProgresso();
}

function atualizarProgresso() {
  $("progresso").textContent = `${estado.indice + 1} de ${estado.total}`;
}

async function salvarESeguir() {
  const item = $("identidade").querySelector(".nome").textContent;
  await fetch("/api/veredicto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      imagem: item,
      marcas: [...estado.selecionadas],
      nota: $("nota").value,
    }),
  });
  await carregar(estado.indice + 1);
}

$("proxima").onclick = salvarESeguir;
$("anterior").onclick = () => carregar(estado.indice - 1);

document.addEventListener("keydown", (evento) => {
  if (evento.target.tagName === "TEXTAREA" && evento.key !== "Enter") return;
  if (evento.key >= "1" && evento.key <= String(estado.marcas.length)) {
    alternar(estado.marcas[Number(evento.key) - 1].chave);
  } else if (evento.key === "Enter") {
    evento.preventDefault();
    salvarESeguir();
  } else if (evento.key === "ArrowLeft") {
    carregar(estado.indice - 1);
  } else if (evento.key === "ArrowRight") {
    carregar(estado.indice + 1);
  } else if (evento.key === "p") {
    estado.previsoes = !estado.previsoes;
    carregar(estado.indice);
  }
});

iniciar();
