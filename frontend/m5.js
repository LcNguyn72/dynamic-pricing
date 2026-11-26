const $ = (id) => document.getElementById(id);
const API = "/api/m5";
const categories = {ALL: "Tất cả", FOODS: "Thực phẩm", HOUSEHOLD: "Đồ gia dụng", HOBBIES: "Sở thích"};
const region = {CA: "California", TX: "Texas", WI: "Wisconsin"};
const moneyFormat = new Intl.NumberFormat("vi-VN", {minimumFractionDigits: 2, maximumFractionDigits: 4});
const quantityFormat = new Intl.NumberFormat("vi-VN", {maximumFractionDigits: 0});
const estimatedQuantityFormat = new Intl.NumberFormat("vi-VN", {maximumFractionDigits: 1});
const dateFormat = new Intl.DateTimeFormat("vi-VN", {day: "2-digit", month: "2-digit", year: "numeric", timeZone: "UTC"});
const timeFormat = new Intl.DateTimeFormat("vi-VN", {dateStyle: "short", timeStyle: "short"});
const chartNumber = new Intl.NumberFormat("vi-VN", {maximumFractionDigits: 2});
const historyMetrics = {price: "Giá bán", qty: "Số lượng đã bán", revenue: "Doanh thu"};
const historyColors = ["#204bd7", "#087366"];
const historyWeeks = 12;
const weekMs = 7 * 24 * 60 * 60 * 1000;
const knownErrors = {
  missing_previous_week: "Tuần này chưa có đủ lịch sử. Hãy thử tuần mặc định.",
  week_in_training_period: "Chọn một tuần sau ngày 06/10/2014.",
  model_unavailable: "Dịch vụ dự đoán giá chưa sẵn sàng. Hãy thử lại sau.",
  price_model_unavailable: "Sản phẩm này chưa có giá đề xuất cho cửa hàng đã chọn.",
  price_outside_train_range: "Giá mới nằm ngoài khoảng mô hình hỗ trợ để ước tính lượng bán.",
  m5_database_unavailable: "Chưa tải được dữ liệu cửa hàng. Hãy thử lại sau.",
};
const state = {category: "ALL", catalog: null, detail: null, edit: null, route: 0};

function safe(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

const money = (value) => moneyFormat.format(Number(value)) + " USD";
const units = (value) => quantityFormat.format(value) + " sản phẩm";
const estimatedUnits = (value) => value > 0 && value < 1 ? "Dưới 1 sản phẩm" :
  estimatedQuantityFormat.format(value) + " sản phẩm";
const day = (value) => dateFormat.format(new Date(value + "T00:00:00Z"));
const plain = (text) => text.toLocaleLowerCase("vi").normalize("NFD").replace(/\p{M}/gu, "").replace(/đ/g, "d");
const detailLink = (sku) => "#san-pham/" + encodeURIComponent(sku);
const predictLink = (sku, store) => "#du-doan?" + new URLSearchParams({sku, store});

async function request(path, options) {
  const response = await fetch(API + path, options);
  if (response.ok) return response.json();
  const data = await response.json().catch(() => ({}));
  const detail = data.detail ?? data;
  const error = new Error(knownErrors[detail.code] || detail.message || "Đã có lỗi. Hãy thử lại.");
  error.code = detail.code;
  throw error;
}

async function getCatalog() {
  if (!state.catalog) {
    state.catalog = request("/catalog").then(({items}) => items).catch((error) => {
      state.catalog = null;
      throw error;
    });
  }
  return state.catalog;
}

function picture(item, className = "") {
  const index = Number(item.position) - 1;
  const x = (index % 5) * 25;
  const y = Math.floor(index / 5) * (100 / 3);
  return `<div class="product-image ${className}" role="img" aria-label="${safe(item.name)}" style="--x:${x}%;--y:${y}%"></div>`;
}

function card(item) {
  return `<article class="product-card">
    <a href="${detailLink(item.sku)}" aria-label="So sánh giá ${safe(item.name)} ở ${item.store_count} cửa hàng">
      ${picture(item)}
      <div class="card-copy">
        <span class="muted small">${safe(categories[item.category])}</span>
        <h3>${safe(item.name)}</h3>
        <div class="card-price"><span>Từ <strong>${money(item.min_price)}</strong></span><span>${item.store_count} cửa hàng</span></div>
      </div>
    </a>
  </article>`;
}

async function loadHomeHighlights(route) {
  const basketNote = $("basket-note"), sellersNote = $("sellers-note");
  const basketList = $("home-basket"), sellersList = $("home-sellers");
  basketNote.textContent = sellersNote.textContent = "Đang tải số liệu…";
  basketList.replaceChildren();
  sellersList.replaceChildren();
  try {
    const {basket, best_sellers: sellers, week} = await request("/home-highlights");
    if (route !== state.route) return;
    const basketMax = Math.max(...basket.map((row) => row.total), 1);
    const salesMax = Math.max(...sellers.map((row) => row.qty), 1);
    basketNote.textContent = basket.length ?
      `Giá đang áp dụng cho ${basket[0].item_count} sản phẩm, mỗi sản phẩm một đơn vị.` :
      "Chưa có cửa hàng đủ sản phẩm cho giỏ mẫu.";
    sellersNote.textContent = week ? `Trong danh mục tại 10 cửa hàng, tuần ${day(week)} (M5).` :
      "Chưa có số lượng bán được ghi nhận.";
    basketList.innerHTML = basket.map((row) => {
      const lowest = row.total === basket[0].total;
      return `<li class="basket-row${lowest ? " basket-lowest" : ""}">
        <span>${safe(row.store_id)}${lowest ? '<span class="basket-tag">Thấp nhất</span>' : ""}</span>
        <span class="bar-track" aria-hidden="true"><span style="--bar:${100 * row.total / basketMax}%"></span></span>
        <strong>${money(row.total)}</strong>
      </li>`;
    }).join("");
    sellersList.innerHTML = sellers.map((row) => `<li class="seller-row">
      <a class="text-link" href="${detailLink(row.sku)}">${safe(row.name)}</a><strong>${units(row.qty)}</strong>
      <span class="bar-track" aria-hidden="true"><span style="--bar:${100 * row.qty / salesMax}%"></span></span>
    </li>`).join("");
  } catch (error) {
    if (route === state.route) basketNote.textContent = sellersNote.textContent = error.message;
  }
}

function renderCatalog(items) {
  const query = plain($("catalog-search").value.trim());
  const matching = items.filter((item) =>
    (state.category === "ALL" || item.category === state.category) &&
    (!query || plain(item.name + " " + item.sku).includes(query))
  );
  $("category-filters").innerHTML = Object.entries(categories).map(([code, name]) =>
    `<button class="filter" type="button" data-category="${code}" aria-pressed="${state.category === code}">${name}</button>`
  ).join("");
  $("catalog-count").textContent = matching.length + " sản phẩm";
  $("catalog-grid").innerHTML = matching.map(card).join("") ||
    '<p class="empty">Hãy thử từ khóa khác hoặc chọn nhóm khác.</p>';
}

function offerRows(item) {
  const lowest = Math.min(...item.offers.map((offer) => offer.price));
  return item.offers.map((offer, i) => `<tr>
    <th scope="row"><strong>${safe(offer.store_id)}</strong><span class="store-region">${safe(region[offer.state])}</span></th>
    <td class="offer-price">${money(offer.price)}
      ${i === 0 && Number(offer.price) === lowest ? '<span class="lowest">Thấp nhất</span>' : ""}
      <span class="offer-sales-mobile">Đã bán ${quantityFormat.format(offer.observed_qty)}</span>
    </td>
    <td class="offer-sales">${units(offer.observed_qty)}</td>
    <td class="offer-action"><button class="text-button" type="button" data-edit="${safe(offer.store_id)}">Chỉnh giá<span class="visually-hidden"> ${safe(offer.store_id)}</span></button></td>
  </tr>`).join("");
}

function editHistory(item) {
  if (!item.changes.length) return "";
  return `<details class="change-history">
    <summary>Các lần chỉnh giá</summary>
    <ul>${item.changes.map((change) => `<li>
      <strong>${safe(change.store_id)}</strong>
      <span>${money(change.old_price)} → ${money(change.new_price)}</span>
      <time datetime="${safe(change.changed_at)}">${timeFormat.format(new Date(change.changed_at))}</time>
    </li>`).join("")}</ul>
  </details>`;
}

function priceComparison(offers) {
  const highest = Math.max(...offers.map((offer) => Number(offer.price)));
  return `<details class="price-comparison">
    <summary>Biểu đồ so sánh giá</summary>
    <ul aria-label="Giá bán theo cửa hàng">
      ${offers.map((offer) => `<li><span>${safe(offer.store_id)}</span>
        <span class="bar-track" aria-hidden="true"><span style="--bar:${100 * Number(offer.price) / highest}%"></span></span>
        <strong>${money(offer.price)}</strong></li>`).join("")}
    </ul>
  </details>`;
}

function renderDetail(item, preferredStore) {
  state.detail = item;
  const firstStore = item.offers[0]?.store_id;
  const store = item.offers.some((offer) => offer.store_id === preferredStore) ? preferredStore : firstStore;
  $("detail").innerHTML = `
    <a class="back-link" href="#san-pham">‹ Danh sách sản phẩm</a>
    <div class="product-detail">
      <div class="product-summary">
        ${picture(item, "detail-image")}
        <p class="muted small">${safe(categories[item.category])}</p>
        <h1>${safe(item.name)}</h1>
        <p class="detail-start">Từ <strong>${money(item.offers[0].price)}</strong></p>
        <a class="button button-primary" href="${predictLink(item.sku, store)}">Xem giá đề xuất</a>
      </div>
      <section class="offers" aria-labelledby="offers-title">
        <div class="section-heading">
          <h2 id="offers-title">Giá tại ${item.offers.length} cửa hàng</h2>
          <button class="button button-neutral batch-edit" type="button" data-edit-many>Chỉnh giá nhiều cửa hàng</button>
        </div>
        <p class="muted small">Giá ban đầu và số đã bán ghi nhận trong tuần ${day(item.week)} của M5.</p>
        <table class="offers-table">
          <thead><tr><th scope="col">Cửa hàng</th><th scope="col">Giá bán</th><th class="offer-sales" scope="col">Đã bán</th><th scope="col"><span class="visually-hidden">Thao tác</span></th></tr></thead>
          <tbody>${offerRows(item)}</tbody>
        </table>
        ${priceComparison(item.offers)}
      </section>
    </div>
    <section class="history" aria-labelledby="history-title">
      <div class="section-heading history-heading">
        <div><h2 id="history-title">Xu hướng bán hàng</h2><p class="muted small">12 tuần gần nhất trong dữ liệu M5</p></div>
        <div class="history-stores">
          <label class="field">Cửa hàng<select id="history-store">${historyStoreOptions(item.offers, store)}</select></label>
          <label class="field">So sánh với<select id="history-compare">
            <option value="">Chọn cửa hàng</option>${historyStoreOptions(item.offers)}
          </select></label>
        </div>
      </div>
      <div class="history-metrics">
        <div class="filters" role="group" aria-label="Chỉ số bán hàng">
          ${Object.entries(historyMetrics).map(([key, label]) =>
            `<button class="filter" data-history-metric="${key}" type="button" aria-pressed="${key === "price"}">${label}</button>`
          ).join("")}
        </div>
        <span id="history-unit" class="muted small">USD</span>
      </div>
      <p id="chart-status" class="muted small" role="status"></p>
      <canvas id="history-chart" role="img" tabindex="0" aria-describedby="chart-reading" hidden></canvas>
      <div id="chart-reading" class="chart-reading small" aria-live="polite" hidden></div>
      ${editHistory(item)}
    </section>`;
  setupHistoryChart(item);
}

function historyStoreOptions(offers, selected) {
  return offers.map(({store_id}) =>
    `<option value="${safe(store_id)}"${store_id === selected ? " selected" : ""}>${safe(store_id)}</option>`
  ).join("");
}

function setupHistoryChart(item) {
  const canvas = $("history-chart");
  const status = $("chart-status");
  const reading = $("chart-reading");
  const primary = $("history-store"), compare = $("history-compare");
  const buttons = [...$("detail").querySelectorAll("[data-history-metric]")];
  const anchor = Date.parse(item.week + "T00:00:00Z");
  const isoDay = (time) => new Date(time).toISOString().slice(0, 10);
  const weeks = Array.from({length: historyWeeks}, (_, i) => isoDay(anchor - (historyWeeks - 1 - i) * weekMs));
  const before = isoDay(anchor + weekMs);
  let metric = "price", selected = weeks.length - 1, series = [], plot, pending = true, lastReading = "";
  const available = new Map();

  function draw() {
    if (pending || !canvas.isConnected || $("detail").hidden) return;
    const values = series.map(({store, rows}, index) => ({
      store, color: historyColors[index], points: weeks.map((week) => rows.get(week)?.[metric] ?? null),
    }));
    const hasData = values.some(({points}) => points.some((value) => value !== null));
    canvas.hidden = reading.hidden = !hasData;
    if (!hasData) {
      status.textContent = "Chưa có số liệu cho khoảng thời gian này.";
      return;
    }
    status.textContent = "";
    canvas.setAttribute("aria-label", `${historyMetrics[metric]} theo tuần. Dùng phím mũi tên để chọn tuần.`);
    plot = drawHistoryChart(canvas, weeks, values, metric, selected);
    const renderValue = metric === "qty" ? units : money;
    const nextReading = `<strong>Tuần ${day(weeks[selected])}</strong><ul>${values.map(({store, color, points}) =>
      `<li><span class="chart-key" style="--line:${color}" aria-hidden="true"></span>${safe(store)}
      <strong>${points[selected] === null ? "Chưa ghi nhận" : renderValue(points[selected])}</strong></li>`
    ).join("")}</ul>`;
    if (lastReading !== nextReading) {
      reading.innerHTML = nextReading;
      lastReading = nextReading;
    }
  }

  async function load() {
    state.chartRequest?.abort();
    const controller = new AbortController();
    state.chartRequest = controller;
    if (compare.value === primary.value) compare.value = "";
    for (const option of compare.options) option.disabled = option.value === primary.value;
    const stores = [primary.value, compare.value].filter(Boolean);
    $("detail").querySelector(".button-primary").href = predictLink(item.sku, primary.value);
    pending = true;
    status.textContent = "Đang tải lịch sử…";
    canvas.hidden = reading.hidden = true;
    try {
      const nextSeries = await Promise.all(stores.map(async (store) => {
        if (!available.has(store)) {
          const params = new URLSearchParams({store_id: store, limit: String(historyWeeks), before});
          const {items} = await request(`/products/${item.sku}/weeks?${params}`, {signal: controller.signal});
          available.set(store, new Map(items.filter((row) => row.observed_sales).map((row) => [row.week, row])));
        }
        return {store, rows: available.get(store)};
      }));
      if (controller.signal.aborted || !canvas.isConnected) return;
      series = nextSeries;
      pending = false;
      draw();
    } catch (error) {
      if (!controller.signal.aborted && canvas.isConnected) status.textContent = error.message;
    }
  }

  function selectWeek(index) {
    const next = Math.max(0, Math.min(weeks.length - 1, index));
    if (next === selected) return;
    selected = next;
    draw();
  }
  function selectPoint(event) {
    if (!plot || pending) return;
    selectWeek(Math.round((event.offsetX - plot.left) / plot.step));
  }

  primary.addEventListener("change", load);
  compare.addEventListener("change", load);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      metric = button.dataset.historyMetric;
      for (const control of buttons) control.setAttribute("aria-pressed", control === button);
      $("history-unit").textContent = metric === "qty" ? "Sản phẩm" : "USD";
      draw();
    });
  }
  canvas.addEventListener("click", selectPoint);
  canvas.addEventListener("pointermove", (event) => {
    if (event.pointerType === "mouse") selectPoint(event);
  });
  canvas.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    selectWeek(selected + (event.key === "ArrowLeft" ? -1 : 1));
  });
  state.chartObserver?.disconnect();
  state.chartObserver = new ResizeObserver(draw);
  state.chartObserver.observe(canvas);
  load();
}

function drawHistoryChart(canvas, weeks, series, metric, selected) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (!width || !height) return;
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const values = series.flatMap(({points}) => points.filter((value) => value !== null));
  const min = Math.min(...values), max = Math.max(...values);
  const margin = Math.max((max - min) * 0.15, 0.05);
  const lo = metric === "price" ? Math.max(0, min - margin) : 0;
  const targetTick = Math.max(max / 3, metric === "qty" ? 1 : 0.01);
  const tickScale = 10 ** Math.floor(Math.log10(targetTick));
  const tick = [1, 2, 5, 10].map((value) => value * tickScale).find((value) => value >= targetTick);
  const hi = metric === "price" ? max + margin : tick * 3;
  const left = 62, right = width - 15, top = 16, bottom = height - 33;
  const step = (right - left) / (weeks.length - 1);
  const x = (i) => left + step * i;
  const y = (price) => bottom - (bottom - top) * (price - lo) / (hi - lo);
  ctx.font = '12px "Be Vietnam Pro", sans-serif';
  ctx.textBaseline = "middle";
  for (let i = 0; i < 4; i++) {
    const price = lo + (hi - lo) * i / 3;
    const height = y(price);
    ctx.strokeStyle = "#dfe7f3";
    ctx.beginPath(); ctx.moveTo(left, height); ctx.lineTo(right, height); ctx.stroke();
    ctx.fillStyle = "#516483";
    ctx.textAlign = "right";
    ctx.fillText(chartNumber.format(price), left - 8, height);
  }
  ctx.strokeStyle = "#91a5c4";
  ctx.beginPath(); ctx.moveTo(x(selected), top); ctx.lineTo(x(selected), bottom); ctx.stroke();
  series.forEach(({color, points}, level) => {
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.setLineDash(level === 0 ? [] : [5, 4]);
    let connected = false;
    ctx.beginPath();
    points.forEach((value, i) => {
      if (value === null) connected = false;
      else {
        if (connected) ctx.lineTo(x(i), y(value));
        else ctx.moveTo(x(i), y(value));
        connected = true;
      }
    });
    ctx.stroke();
    points.forEach((value, i) => {
      if (value === null) return;
      ctx.beginPath(); ctx.arc(x(i), y(value), i === selected ? 4.5 : 2.5, 0, Math.PI * 2); ctx.fill();
    });
  });
  ctx.fillStyle = "#516483";
  ctx.textAlign = "left";
  ctx.fillText(day(weeks[0]), left, bottom + 20);
  ctx.textAlign = "right";
  ctx.fillText(day(weeks.at(-1)), right, bottom + 20);
  return {left, step};
}

async function renderPredict(params, route) {
  const [items, {stores}] = await Promise.all([getCatalog(), request("/stores")]);
  if (route !== state.route) return;
  const item = items.find((entry) => entry.sku === params.get("sku")) ?? items[0];
  const store = stores.find((entry) => entry.id === params.get("store"))?.id ?? stores[0]?.id;
  $("predict").innerHTML = `
    <div class="page-heading"><h1>Dự đoán giá</h1><p>Chọn sản phẩm, cửa hàng và tuần để xem giá đề xuất.</p></div>
    <div class="predict-layout">
      <form id="predict-form" class="predict-form">
        <label class="field">Sản phẩm<select name="sku">${items.map((entry) =>
          `<option value="${safe(entry.sku)}"${item.sku === entry.sku ? " selected" : ""}>${safe(entry.name)}</option>`
        ).join("")}</select></label>
        <label class="field">Cửa hàng<select name="store_id">${stores.map((entry) =>
          `<option value="${safe(entry.id)}"${store === entry.id ? " selected" : ""}>${safe(entry.id)} · ${safe(region[entry.state])}</option>`
        ).join("")}</select></label>
        <label class="field">Tuần dự đoán<input type="date" name="week" value="${safe(item.week)}" required>
          <span class="muted small">Chọn ngày thứ Hai đầu tuần.</span>
        </label>
        <button class="button button-primary" type="submit">Xem giá đề xuất</button>
        <p id="predict-error" class="inline-error" role="alert"></p>
      </form>
      <section class="result" id="predict-result" aria-live="polite">
        <h2>Giá đề xuất</h2>
        <p class="muted">Kết quả sẽ xuất hiện sau khi tính.</p>
      </section>
    </div>`;
  const form = $("predict-form");
  form.addEventListener("submit", runPrediction);
  form.addEventListener("change", () => {
    $("predict-error").textContent = "";
    $("predict-result").innerHTML = '<h2>Giá đề xuất</h2><p class="muted">Bấm Xem giá đề xuất để tính lại.</p>';
  });
}

function demandChart(current, proposed, {label = "Giá đề xuất", metric = "qty", applied = false} = {}) {
  const revenue = metric === "revenue";
  const field = revenue ? "expected_revenue" : "predicted_qty";
  const title = revenue ? "Doanh thu dự kiến theo giá" : "Lượng bán dự kiến theo giá";
  const format = revenue ? (value) => money(Number(value.toFixed(2))) : estimatedUnits;
  const expected = proposed[field];
  const baseline = current?.[field];
  const baselineLabel = applied ? "Giá trước khi áp dụng" : "Giá đang áp dụng";
  const max = Math.max(baseline ?? 0, expected, 1);
  const rows = current ? [[baselineLabel, baseline], [label, expected]] : [[label, expected]];
  let summary = baselineLabel + " nằm ngoài khoảng giá mô hình hỗ trợ.";
  if (current) {
    const delta = expected - baseline;
    const direction = delta > 0 ? "Tăng" : "Giảm";
    if (revenue) {
      summary = Math.abs(delta) < 0.005 ? "Doanh thu dự kiến gần bằng nhau." :
        direction + " khoảng " + format(Math.abs(delta)) + ".";
    } else {
      summary = Math.abs(delta) < 1e-8 ? "Lượng bán dự kiến ngang nhau." :
        Math.abs(delta) < 1 ? direction + " dưới 1 sản phẩm." :
          direction + " khoảng " + units(Math.abs(delta)) + ".";
    }
  }
  return `<figure class="demand-chart" aria-label="${title}">
    <figcaption>${title}</figcaption>
    <ul>${rows.map(([label, quantity], index) => `<li>
      <span${current && index === 0 ? " data-price-baseline" : ""}>${label}</span><strong>${format(quantity)}</strong>
      <span class="bar-track" aria-hidden="true"><span style="--bar:${100 * quantity / max}%"></span></span>
    </li>`).join("")}</ul>
    <p class="small">${summary}</p>
  </figure>`;
}

async function runPrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const errorNode = $("predict-error"), result = $("predict-result");
  const data = Object.fromEntries(new FormData(form));
  if (new Date(data.week + "T00:00:00Z").getUTCDay() !== 1) {
    errorNode.textContent = "Chọn ngày thứ Hai đầu tuần.";
    form.elements.week.focus();
    return;
  }
  errorNode.textContent = "";
  result.innerHTML = '<h2>Giá đề xuất</h2><p class="muted">Đang tính giá…</p>';
  for (const field of form.elements) field.disabled = true;
  try {
    const item = await request("/catalog/" + data.sku);
    const offer = item.offers.find((entry) => entry.store_id === data.store_id);
    const price = await request("/recommend-price", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...data, current_price: offer.price, strategy: "grid"}),
    });
    const proposed = Number(price.recommendation.price);
    const applied = proposed.toFixed(2);
    const rounded = Math.abs(proposed - Number(applied)) > 1e-8;
    const difference = proposed - Number(offer.price);
    const comparison = Math.abs(difference) < 0.005 ? "Gần bằng giá đang áp dụng" :
      (difference > 0 ? "Cao hơn " : "Thấp hơn ") + money(Math.abs(difference));
    result.innerHTML = `<h2>Giá đề xuất</h2><strong class="result-price">${money(proposed)}</strong>
      <p class="result-comparison">${comparison}</p>
      <dl class="result-data"><div><dt data-price-baseline>Giá đang áp dụng</dt><dd>${money(offer.price)}</dd></div>
      <div><dt>${safe(item.name)} · ${safe(offer.store_id)}</dt><dd>Tuần ${day(data.week)}</dd></div></dl>
      <div class="filters prediction-metrics" role="group" aria-label="Chỉ số dự kiến">
        <button class="filter" data-prediction-metric="qty" type="button" aria-pressed="true">Lượng bán dự kiến</button>
        <button class="filter" data-prediction-metric="revenue" type="button" aria-pressed="false">Doanh thu dự kiến</button>
      </div>
      <div class="prediction-chart">${demandChart(price.current_price_point, price.recommendation)}</div>
      ${rounded ? '<p class="small muted">Giá lưu dùng hai chữ số thập phân.</p>' : ""}
      <div class="prediction-actions">
        <button class="button button-primary" id="apply-recommendation" type="button">Áp dụng ${money(applied)}</button>
        <a class="text-link" href="${detailLink(item.sku)}">Xem trang sản phẩm</a>
      </div>
      <p id="apply-status" class="small apply-status" role="status"></p>`;
    const chart = result.querySelector(".prediction-chart");
    const metrics = [...result.querySelectorAll("[data-prediction-metric]")];
    for (const button of metrics) {
      button.addEventListener("click", () => {
        for (const control of metrics) control.setAttribute("aria-pressed", control === button);
        chart.innerHTML = demandChart(price.current_price_point, price.recommendation, {
          metric: button.dataset.predictionMetric,
          applied: chart.dataset.applied === "true",
        });
      });
    }
    result.querySelector("#apply-recommendation").addEventListener("click", (event) => {
      applyRecommendation(result, event.currentTarget, item.sku, offer.store_id, applied);
    });
  } catch (error) {
    errorNode.textContent = error.message;
    result.innerHTML = '<h2>Giá đề xuất</h2><p class="muted">Hãy kiểm tra lựa chọn và thử lại.</p>';
  } finally {
    for (const field of form.elements) field.disabled = false;
  }
}

async function saveCatalogPrices(sku, stores, price) {
  const saved = await request(`/catalog/${sku}/price`, {
    method: "PUT", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({price, store_ids: stores}),
  });
  state.catalog = null;
  return saved;
}

async function applyRecommendation(result, button, sku, store, price) {
  if (button.disabled) return;
  const status = result.querySelector("#apply-status");
  button.disabled = true;
  status.classList.remove("inline-error");
  status.textContent = "Đang áp dụng giá…";
  try {
    await saveCatalogPrices(sku, [store], price);
    button.textContent = "Đã áp dụng";
    status.textContent = `Giá tại ${store} hiện là ${money(price)}.`;
    if (button.isConnected) {
      result.querySelector(".prediction-chart").dataset.applied = "true";
      for (const label of result.querySelectorAll("[data-price-baseline]")) {
        label.textContent = "Giá trước khi áp dụng";
      }
    }
  } catch (error) {
    status.classList.add("inline-error");
    status.textContent = error.message;
    button.disabled = false;
  }
}

function openEdit(store) {
  const item = state.detail, offer = item?.offers.find((entry) => entry.store_id === store);
  if (!item || (store && !offer)) return;
  state.edit = {item, store};
  $("dialog-title").textContent = store ? "Chỉnh giá bán" : "Chỉnh giá nhiều cửa hàng";
  $("edit-context").textContent = store ? item.name + " · " + store : item.name;
  $("edit-store-picker").hidden = Boolean(store);
  $("edit-store-options").innerHTML = store ? "" : [...item.offers].sort((a, b) => a.store_id.localeCompare(b.store_id))
    .map((entry) => `<label><input type="checkbox" value="${safe(entry.store_id)}">${safe(entry.store_id)}</label>`).join("");
  $("edit-price").value = offer ? Number(offer.price).toFixed(2) : "";
  $("edit-error").textContent = "";
  syncEditStores();
  $("price-dialog").showModal();
  if (store) $("edit-price").select();
  else $("edit-store-options").querySelector("input")?.focus();
}

function selectedEditStores() {
  if (state.edit.store) return [state.edit.store];
  return [...$("edit-store-options").querySelectorAll("input:checked")].map((input) => input.value);
}

function syncEditStores() {
  const stores = selectedEditStores();
  const total = state.edit.item.offers.length;
  $("edit-store-count").textContent = `Đã chọn ${stores.length}/${total} cửa hàng`;
  $("edit-toggle-stores").textContent = stores.length === total ? "Bỏ chọn tất cả" : "Chọn tất cả";
  $("edit-save").textContent = stores.length > 1 ? `Lưu cho ${stores.length} cửa hàng` : "Lưu giá mới";
  $("edit-save").disabled = stores.length === 0;
  const select = $("edit-preview-store");
  const previous = select.value;
  select.replaceChildren(...stores.map((store) => new Option(store, store)));
  if (stores.includes(previous)) select.value = previous;
  $("edit-preview-store-field").hidden = stores.length < 2;
  queueEditPreview();
}

function cancelEditPreview() {
  clearTimeout(state.editPreviewTimer);
  state.editPreview?.abort();
}

function queueEditPreview() {
  cancelEditPreview();
  const input = $("edit-price");
  const preview = $("edit-preview");
  const price = input.valueAsNumber;
  const store = $("edit-preview-store").value;
  const offer = state.edit.item.offers.find((entry) => entry.store_id === store);
  $("edit-current").hidden = !offer;
  if (!offer) {
    preview.textContent = "Chọn cửa hàng và nhập giá mới để xem lượng bán dự kiến.";
    return;
  }
  $("edit-current").textContent = "Giá đang áp dụng" + (state.edit.store ? " " : " tại " + store + " ") + money(offer.price);
  if (!input.validity.valid || price === Number(offer.price)) {
    preview.textContent = "Nhập giá mới để xem lượng bán dự kiến.";
    return;
  }
  const controller = new AbortController();
  state.editPreview = controller;
  preview.textContent = "Đang ước tính lượng bán…";
  state.editPreviewTimer = setTimeout(() => previewEditedPrice(state.edit.item, offer, price, controller.signal), 350);
}

async function previewEditedPrice(item, offer, price, signal) {
  const preview = $("edit-preview");
  const forecast = (candidate_price) => request("/forecast", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sku: item.sku, store_id: offer.store_id, week: item.week, candidate_price}),
    signal,
  });
  try {
    const [current, edited] = await Promise.all([
      forecast(Number(offer.price)).catch((error) => {
        if (error.code === "price_outside_train_range") return null;
        throw error;
      }),
      forecast(price),
    ]);
    if (signal.aborted || !$("price-dialog").open) return;
    preview.innerHTML = `<p>Ước tính cho tuần ${day(item.week)}</p>${demandChart(current, edited, {label: "Giá mới"})}`;
  } catch (error) {
    if (!signal.aborted && $("price-dialog").open) preview.textContent = error.message;
  }
}

async function savePrice(event) {
  event.preventDefault();
  const {item, store} = state.edit;
  const stores = selectedEditStores();
  const button = $("edit-save");
  if (!stores.length || button.disabled) return;
  const previewStore = $("edit-preview-store").value;
  button.disabled = true;
  $("edit-store-picker").disabled = true;
  $("edit-price").readOnly = true;
  $("edit-preview-store").disabled = true;
  $("edit-cancel").disabled = true;
  $("edit-error").textContent = "";
  try {
    const result = await saveCatalogPrices(item.sku, stores, $("edit-price").value);
    const refreshed = await request("/catalog/" + item.sku).catch(() => null);
    $("price-dialog").close();
    if (state.detail?.sku === item.sku && !$("detail").hidden) {
      if (refreshed) renderDetail(refreshed, previewStore);
      else {
        for (const offer of item.offers) {
          if (stores.includes(offer.store_id)) offer.price = result.price;
        }
        item.offers.sort((a, b) => Number(a.price) - Number(b.price) || a.store_id.localeCompare(b.store_id));
        renderDetail(item, previewStore);
      }
      $("detail").querySelector(store ? `[data-edit="${store}"]` : "[data-edit-many]")?.focus({preventScroll: true});
    }
    const changed = result.stores.filter((entry) => entry.changed).length;
    showToast(changed ? `Đã cập nhật giá tại ${changed} cửa hàng.` : "Các cửa hàng đã chọn đang áp dụng mức giá này.");
  } catch (error) {
    $("edit-error").textContent = error.message;
  } finally {
    button.disabled = false;
    $("edit-store-picker").disabled = false;
    $("edit-price").readOnly = false;
    $("edit-preview-store").disabled = false;
    $("edit-cancel").disabled = false;
  }
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(state.toast);
  state.toast = setTimeout(() => { toast.hidden = true; }, 4500);
}

async function navigate() {
  const [path, search] = location.hash.slice(1).split("?");
  if (path === "workspace") return;
  const route = ++state.route;
  const itemKey = /^san-pham\/([A-Z0-9_]+)$/.exec(path)?.[1];
  const view = itemKey ? "detail" : path === "san-pham" ? "catalog" : path === "du-doan" ? "predict" : "home";
  document.querySelectorAll(".view").forEach((section) => { section.hidden = section.id !== view; });
  for (const link of document.querySelectorAll("[data-nav]")) {
    if (link.dataset.nav === (view === "detail" ? "catalog" : view)) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  state.chartObserver?.disconnect();
  state.chartRequest?.abort();
  if (itemKey || view === "predict") $(view).replaceChildren();
  const status = $("page-status");
  status.textContent = "Đang tải…";
  window.scrollTo(0, 0);
  if (view === "home") loadHomeHighlights(route);
  try {
    if (itemKey) {
      const item = await request("/catalog/" + itemKey);
      if (route === state.route) renderDetail(item);
    } else if (view === "predict") {
      await renderPredict(new URLSearchParams(search), route);
    } else {
      const items = await getCatalog();
      if (route !== state.route) return;
      if (view === "home") {
        const featured = [1, 9, 16, 2].map((position) => items.find((item) => item.position === position)).filter(Boolean);
        $("featured-grid").innerHTML = featured.map(card).join("");
      } else renderCatalog(items);
    }
    if (route === state.route) status.textContent = "";
  } catch (error) {
    if (route === state.route) status.textContent = error.message;
  }
}

$("catalog-search").addEventListener("input", async () => {
  if (state.catalog) renderCatalog(await state.catalog);
});
$("category-filters").addEventListener("click", async (event) => {
  const filter = event.target.closest("[data-category]");
  if (!filter) return;
  state.category = filter.dataset.category;
  renderCatalog(await getCatalog());
  $("category-filters").querySelector(`[data-category="${state.category}"]`).focus();
});
$("detail").addEventListener("click", (event) => {
  const edit = event.target.closest("[data-edit]");
  if (edit) openEdit(edit.dataset.edit);
  if (event.target.closest("[data-edit-many]")) openEdit();
});
$("edit-store-options").addEventListener("change", syncEditStores);
$("edit-toggle-stores").addEventListener("click", () => {
  const inputs = [...$("edit-store-options").querySelectorAll("input")];
  const selectAll = inputs.some((input) => !input.checked);
  inputs.forEach((input) => { input.checked = selectAll; });
  syncEditStores();
});
$("edit-preview-store").addEventListener("change", queueEditPreview);
$("edit-cancel").addEventListener("click", () => $("price-dialog").close());
$("edit-price").addEventListener("input", queueEditPreview);
$("price-dialog").addEventListener("close", cancelEditPreview);
$("price-dialog").addEventListener("cancel", (event) => {
  if ($("edit-save").disabled) event.preventDefault();
});
$("edit-form").addEventListener("submit", savePrice);
document.querySelector(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  $("workspace").focus();
});
window.addEventListener("hashchange", navigate);
navigate();
