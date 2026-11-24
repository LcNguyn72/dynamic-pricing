const $ = (id) => document.getElementById(id);
const money = new Intl.NumberFormat("vi-VN", { style: "currency", currency: "USD" });
const unitPrice = new Intl.NumberFormat("vi-VN", {
  style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 4,
});
const decimalMoney = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const decimalPrice = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 2, maximumFractionDigits: 4 });
const units = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 });
const dateFormat = new Intl.DateTimeFormat("vi-VN", {
  day: "2-digit", month: "2-digit", year: "numeric", timeZone: "UTC",
});
const categories = { FOODS: "Thực phẩm", HOBBIES: "Sở thích", HOUSEHOLD: "Gia dụng" };
const names = { grid: "Lưới giá", ppo_feedback: "PPO có phản hồi" };
const pageSize = 12;
const historyPageSize = 52;
const state = {
  store: "", stats: null, query: "", offset: 0, catalog: [], detail: null, model: null,
  view: "", storeRevision: 0, catalogRevision: 0, detailRevision: 0, inputRevision: 0,
  pricingBusy: false, feedbackBusy: false,
};

function formatted(formatter, value) {
  return Number.isFinite(value) ? formatter.format(value) : "—";
}

function formatDate(week) {
  const date = new Date(week + "T00:00:00Z");
  return Number.isNaN(date.valueOf()) ? "—" : dateFormat.format(date);
}

function nextWeek(week) {
  if (!week) return "";
  const date = new Date(week + "T00:00:00Z");
  date.setUTCDate(date.getUTCDate() + 7);
  return date.toISOString().slice(0, 10);
}

function monday(week) {
  return new Date(week + "T00:00:00Z").getUTCDay() === 1;
}

function showStatus(id, message, tone = "") {
  const element = $(id);
  element.textContent = message;
  element.hidden = !message;
  element.dataset.tone = tone;
}

const errorMessages = {
  ppo_policy_unavailable: "PPO chưa khả dụng cho sản phẩm và cửa hàng này.",
  ppo_series_unavailable: "PPO chưa khả dụng cho sản phẩm và cửa hàng này.",
  price_model_unavailable: "Sản phẩm này chưa có mô hình ARIMAX tại cửa hàng.",
  price_outside_train_range: "Giá nhập nằm ngoài khoảng giá mô hình đã học.",
  missing_previous_week: "Cần có dữ liệu bán hàng của tuần ngay trước đó.",
};

async function api(path, options) {
  let response;
  try {
    response = await fetch("/api/m5" + path, options);
  } catch {
    throw new Error("Không kết nối được với hệ thống. Hãy thử lại.");
  }
  const body = await response.json().catch(() => null);
  if (response.ok && body) return body;
  const detail = body?.detail ?? body;
  if (Array.isArray(detail)) throw new Error("Kiểm tra lại mã sản phẩm, ngày và số liệu đã nhập.");
  throw new Error(errorMessages[detail?.code] || detail?.message
    || (typeof detail === "string" ? detail : "Yêu cầu thất bại. Hãy thử lại."));
}

function get(path, parameters) {
  const query = parameters ? "?" + new URLSearchParams(parameters).toString() : "";
  return api(path + query);
}

function post(path, body) {
  return api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function row(values) {
  const tr = element("tr");
  for (const item of values) tr.append(typeof item === "string" ? element("td", item) : item);
  return tr;
}

function emptyRow(target, columns, message) {
  const cell = element("td", message, "table-empty");
  cell.colSpan = columns;
  $(target).replaceChildren(row([cell]));
}

function drawChart(id, values, dark = false) {
  const canvas = $(id);
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const points = values.filter(Number.isFinite);
  if (points.length < 2) return;

  const padding = dark ? 5 : 12;
  const minimum = dark ? Math.min(...points) : 0;
  const maximum = Math.max(...points);
  const range = maximum - minimum || 1;
  const baseline = height - padding;
  const coords = points.map((value, index) => [
    padding + index / (points.length - 1) * (width - 2 * padding),
    baseline - (value - minimum) / range * (height - 2 * padding),
  ]);
  if (!dark) {
    context.strokeStyle = "#e1e8ef";
    for (let index = 0; index <= 4; index += 1) {
      const y = padding + index * (height - 2 * padding) / 4;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }
  }
  context.beginPath();
  coords.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
  context.lineWidth = 2.4;
  context.lineJoin = "round";
  context.strokeStyle = dark ? "#ffc85b" : "#216a9b";
  context.stroke();
  context.lineTo(coords.at(-1)[0], baseline);
  context.lineTo(coords[0][0], baseline);
  context.closePath();
  context.fillStyle = dark ? "rgba(255, 200, 91, .11)" : "rgba(33, 106, 155, .10)";
  context.fill();
  const [lastX, lastY] = coords.at(-1);
  context.beginPath();
  context.arc(lastX, lastY, 3.5, 0, 2 * Math.PI);
  context.fillStyle = context.strokeStyle;
  context.fill();
}

function orderedWeeks(data) {
  return [...(data?.weekly || [])].sort((a, b) => a.week.localeCompare(b.week));
}

function updateCharts() {
  const weeks = orderedWeeks(state.stats);
  const metric = $("metric").value;
  const selected = weeks.slice(-26);
  drawChart("overview-chart", weeks.slice(-13).map((week) => week.revenue), true);
  drawChart("stats-chart", selected.map((week) => week[metric]));
  $("trend-start").textContent = selected.length ? formatDate(selected[0].week) : "";
  $("trend-end").textContent = selected.length ? formatDate(selected.at(-1).week) : "";
  $("stats-chart").setAttribute("aria-label", metric === "revenue" ? "Doanh thu theo tuần" : "Lượng bán theo tuần");
}

function renderStats() {
  const weeks = orderedWeeks(state.stats);
  const latest = weeks.at(-1);
  const period = weeks.length
    ? formatDate(weeks[0].week) + " – " + formatDate(latest.week)
    : "";
  $("overview-period").textContent = period ? "M5 · " + period : "Dữ liệu bán hàng M5";
  $("stats-period").textContent = period ? "Dữ liệu lịch sử " + period : "Số liệu lịch sử của cửa hàng.";
  $("overview-week").textContent = latest ? "Tuần " + formatDate(latest.week) : "";
  $("overview-latest").textContent = formatted(money, latest?.revenue);
  for (const prefix of ["overview", "stats"]) {
    $(prefix + "-revenue").textContent = formatted(money, state.stats?.total_revenue);
    $(prefix + "-qty").textContent = formatted(units, state.stats?.total_qty);
    $(prefix + "-count").textContent = formatted(units, state.stats?.product_count);
    const rows = weeks.slice(prefix === "overview" ? -5 : -26).reverse().map((week) => row([
      formatDate(week.week), formatted(units, week.qty), formatted(money, week.revenue),
    ]));
    if (rows.length) $(prefix + "-weeks").replaceChildren(...rows);
    else emptyRow(prefix + "-weeks", 3, "Chưa có dữ liệu theo tuần.");
  }
  updateCharts();
}

async function loadStats(revision) {
  try {
    const result = await get("/stats", { store_id: state.store });
    if (revision !== state.storeRevision) return;
    state.stats = result;
    renderStats();
    setDefaultWeek();
    showStatus("app-status", "");
  } catch (error) {
    if (revision !== state.storeRevision) return;
    showStatus("app-status", "Không tải được thống kê. " + error.message, "error");
  }
}

function productRow(product) {
  const sku = element("span", undefined, "sku-tag");
  const dot = element("span", undefined, "sku-dot");
  dot.dataset.category = product.category;
  dot.setAttribute("aria-hidden", "true");
  sku.append(dot, product.sku);
  const identifier = element("td");
  identifier.append(sku);
  const detail = element("button", "Xem lịch sử", "link-button");
  detail.type = "button";
  detail.setAttribute("aria-label", "Xem lịch sử " + product.sku);
  detail.addEventListener("click", () => loadDetail(product));
  const action = element("td");
  action.append(detail);
  return row([
    identifier, categories[product.category] || product.category || "—", formatted(unitPrice, product.last_price), action,
  ]);
}

async function loadCatalog() {
  const revision = ++state.catalogRevision;
  const { store, query, offset } = state;
  $("products-previous").disabled = true;
  $("products-next").disabled = true;
  $("product-count").textContent = "";
  emptyRow("product-list", 4, "Đang tải sản phẩm…");
  try {
    const result = await get("/products", { store_id: store, q: query, limit: pageSize, offset });
    if (revision !== state.catalogRevision) return;
    state.catalog = result.items;
    const rows = result.items.map(productRow);
    if (rows.length) $("product-list").replaceChildren(...rows);
    else emptyRow("product-list", 4, "Không tìm thấy sản phẩm phù hợp.");
    $("product-count").textContent = formatted(units, result.total) + " sản phẩm";
    $("products-previous").disabled = offset === 0;
    $("products-next").disabled = offset + pageSize >= result.total;
    renderPricingOptions();
  } catch (error) {
    if (revision !== state.catalogRevision) return;
    emptyRow("product-list", 4, error.message);
  }
}

function loadDetail(product) {
  state.detailRevision += 1;
  state.detail = { sku: product.sku, before: "" };
  $("product-detail").hidden = false;
  $("detail-heading").textContent = product.sku;
  $("detail-description").textContent = (categories[product.category] || product.category) + " · " + state.store;
  $("detail-pricing").dataset.sku = product.sku;
  $("history-pagination").hidden = true;
  emptyRow("product-weeks", 4, "Đang tải lịch sử…");
  $("detail-heading").focus();
  void loadProductWeeks();
}

async function loadProductWeeks() {
  const revision = state.detailRevision;
  const { sku, before } = state.detail;
  const params = { store_id: state.store, limit: historyPageSize };
  if (before) params.before = before;
  const button = $("load-older-weeks");
  button.disabled = true;
  showStatus("product-history-status", "");
  try {
    const result = await get("/products/" + encodeURIComponent(sku) + "/weeks", params);
    if (revision !== state.detailRevision) return;
    const rows = result.items.map((week) => row([
      formatDate(week.week), formatted(unitPrice, week.price),
      week.observed_sales ? formatted(units, week.qty) : "—", formatted(money, week.revenue),
    ]));
    if (before) $("product-weeks").append(...rows);
    else if (rows.length) $("product-weeks").replaceChildren(...rows);
    else emptyRow("product-weeks", 4, "Sản phẩm chưa có lịch sử bán hàng.");
    if (rows.length) state.detail.before = result.items.at(-1).week;
    $("history-pagination").hidden = rows.length < historyPageSize;
  } catch (error) {
    if (revision !== state.detailRevision) return;
    if (before) showStatus("product-history-status", error.message, "error");
    else emptyRow("product-weeks", 4, error.message);
  } finally {
    if (revision === state.detailRevision) button.disabled = false;
  }
}

function modelScope() {
  return state.model?.m5_pricing_policies?.ppo_feedback;
}

function storePolicySkus() {
  return (modelScope()?.scope || []).filter((item) => item.store_id === state.store).map((item) => item.sku);
}

function setDefaultWeek() {
  const dataset = state.model?.datasets?.m5;
  const week = $("pricing-week");
  week.min = nextWeek(dataset?.train_end);
  if (week.value && !(week.dataset.source === "stats" && dataset?.test_start)) {
    showWeekNote("pricing-week", "pricing-date-note");
    return;
  }
  const defaultWeek = dataset?.test_start || nextWeek(state.stats?.latest_week);
  const previousDefault = week.value;
  week.value = defaultWeek || "";
  week.dataset.source = dataset?.test_start ? "model" : "stats";
  if (!$("feedback-week").value || $("feedback-week").value === previousDefault) {
    $("feedback-week").value = week.value;
  }
  showWeekNote("pricing-week", "pricing-date-note");
  showWeekNote("feedback-week", "feedback-week-note");
}

function showWeekNote(input, note) {
  const date = $(input).value;
  $(note).textContent = monday(date)
    ? "Thứ Hai, " + formatDate(date)
    : "Chọn ngày thứ Hai đầu tuần.";
}

function seedPricingExample() {
  const sku = $("pricing-sku");
  if (sku.value || sku.dataset.edited === "true" || !modelScope()?.available) return;
  const available = storePolicySkus();
  if (available.length) {
    sku.value = available.includes("FOODS_3_090") ? "FOODS_3_090" : available[0];
    $("strategy").options[2].disabled = false;
    $("strategy").value = "compare";
  }
}

function renderPricingOptions() {
  const availableSkus = storePolicySkus();
  const list = new Set([...availableSkus, ...state.catalog.map((product) => product.sku)]);
  $("pricing-skus").replaceChildren(...[...list].map((sku) => new Option(sku, sku)));
  const sku = $("pricing-sku").value.trim().toUpperCase();
  const ppoReady = Boolean(modelScope()?.available && availableSkus.includes(sku));
  const strategy = $("strategy");
  strategy.options[1].disabled = !ppoReady;
  strategy.options[2].disabled = !ppoReady;
  if (!ppoReady) strategy.value = "grid";
  let status = state.model ? "" : "Chưa đọc được trạng thái chính sách PPO.";
  if (state.model && !modelScope()?.available) status = "Máy chủ chưa tải chính sách PPO.";
  else if (state.model && !ppoReady && sku) status = "PPO chưa có cho mã sản phẩm này tại cửa hàng.";
  showStatus("policy-status", status);
  $("feedback-context").textContent = sku
    ? sku + " · " + state.store + ". Phản hồi được xét cho các tuần dự báo sau đó."
    : "Chọn mã sản phẩm ở phần định giá.";
  $("calculate").disabled = !state.store || !sku || state.pricingBusy || state.feedbackBusy;
  $("save-feedback").disabled = !state.store || !sku || state.feedbackBusy || state.pricingBusy;
}

function clearComparison(message = "Chọn sản phẩm và tuần để tính giá.") {
  $("comparison").hidden = true;
  $("comparison-summary").hidden = true;
  $("pricing-candidates").hidden = true;
  $("comparison-context").textContent = "";
  showStatus("pricing-status", message);
}

function resetInputs() {
  state.inputRevision += 1;
  state.pricingBusy = false;
  clearComparison();
  showStatus("feedback-status", "");
  renderPricingOptions();
}

function recommendationHeaders(points) {
  const tr = row([]);
  const metric = element("th", "Chỉ số");
  metric.scope = "col";
  tr.append(metric);
  for (const point of points) {
    const th = element("th", point.label);
    th.scope = "col";
    tr.append(th);
  }
  return tr;
}

function renderComparison(results, originalForecast, failures, payload) {
  const primary = results[0]?.value;
  const recommendations = results.map(({ strategy, value }) => ({
    label: names[strategy], point: value.recommendation || value.grid_result,
  })).filter((item) => item.point);

  const reference = primary?.current_price_point || (originalForecast?.candidate_price != null ? {
    price: originalForecast.candidate_price,
    predicted_qty: originalForecast.predicted_qty,
    expected_revenue: originalForecast.candidate_price * originalForecast.predicted_qty,
  } : null);
  const points = [{ label: "Giá hiện tại", point: reference }, ...recommendations];
  const best = Math.max(...points.map((item) => item.point?.expected_revenue ?? -Infinity));
  const metrics = [
    ["Giá bán (USD)", "price", decimalPrice],
    ["Lượng bán dự báo", "predicted_qty", units],
    ["Doanh thu dự kiến (USD)", "expected_revenue", decimalMoney],
  ];
  $("comparison-head").replaceChildren(recommendationHeaders(points));
  $("comparison-body").replaceChildren(...metrics.map(([label, key, formatter]) => {
    const heading = element("th", label);
    heading.scope = "row";
    return row([heading, ...points.map(({ point }) => {
      const value = key === "price" ? point?.price ?? primary?.current_price : point?.[key];
      const cell = element("td", formatted(formatter, value));
      if (recommendations.length > 1 && point?.expected_revenue === best) cell.dataset.best = "true";
      return cell;
    })]);
  }));
  $("comparison").hidden = false;
  $("comparison-context").textContent = payload.sku + " · " + formatDate(payload.week);
  const candidates = results.find((result) => result.strategy === "grid")?.value.candidates || [];
  $("candidate-list").replaceChildren(...candidates.map((point) => row([
    formatted(unitPrice, point.price), formatted(units, point.predicted_qty), formatted(money, point.expected_revenue),
  ])));
  $("pricing-candidates").hidden = !candidates.length;
  const notes = [];
  if (!reference && !primary?.current_price_in_train_range) notes.push("Giá hiện tại nằm ngoài khoảng giá đã học.");
  const count = primary?.feedback_count ?? originalForecast?.feedback_count;
  if (Number.isInteger(count)) notes.push("Phản hồi dùng để cập nhật ARIMAX: " + count + ".");
  if (primary?.train_price_range) notes.push("Khoảng giá đã học " + formatted(unitPrice, primary.train_price_range.min)
    + " – " + formatted(unitPrice, primary.train_price_range.max) + ".");
  $("comparison-summary").textContent = notes.join(" ");
  $("comparison-summary").hidden = !notes.length;
  showStatus("pricing-status", failures.join(" "), "error");
}

async function calculate(event) {
  event.preventDefault();
  const sku = $("pricing-sku").value.trim().toUpperCase();
  const week = $("pricing-week").value;
  if (!monday(week)) {
    clearComparison();
    showStatus("pricing-status", "Chọn ngày thứ Hai đầu tuần.", "error");
    $("pricing-week").focus();
    return;
  }
  const revision = ++state.inputRevision;
  const reference = $("pricing-current").valueAsNumber;
  const base = { sku, store_id: state.store, week };
  const price = Number.isFinite(reference) ? { current_price: reference } : {};
  const forecastPrice = Number.isFinite(reference) ? { candidate_price: reference } : {};
  const value = $("strategy").value;
  const methods = value === "compare" ? ["grid", "ppo_feedback"] : [value];
  state.pricingBusy = true;
  renderPricingOptions();
  clearComparison("Đang tính giá đề xuất…");
  const [forecast, ...recommendations] = await Promise.allSettled([
    post("/forecast", { ...base, ...forecastPrice }),
    ...methods.map((strategy) => post("/recommend-price", { ...base, ...price, strategy })),
  ]);
  if (revision !== state.inputRevision) return;
  state.pricingBusy = false;
  renderPricingOptions();
  const results = [];
  const failures = [];
  recommendations.forEach((result, index) => {
    const strategy = methods[index];
    if (result.status === "fulfilled") results.push({ strategy, value: result.value });
    else failures.push(names[strategy] + ": " + result.reason.message);
  });
  if (results.length || forecast.status === "fulfilled") {
    if (!results.length && forecast.value.candidate_price == null) {
      showStatus("pricing-status", failures.join(" "), "error");
      return;
    }
    renderComparison(results, forecast.value, failures, base);
  } else {
    showStatus("pricing-status", [...new Set(failures)].join(" "), "error");
  }
}

function setFeedbackSource() {
  const manual = $("feedback-source").value === "manual";
  for (const label of document.querySelectorAll("[data-manual-feedback]")) {
    label.hidden = !manual;
    label.control.disabled = !manual;
  }
  showStatus("feedback-status", "");
}

async function saveFeedback(event) {
  event.preventDefault();
  const sku = $("pricing-sku").value.trim().toUpperCase();
  const store_id = state.store;
  const week = $("feedback-week").value;
  if (!monday(week)) {
    showStatus("feedback-status", "Chọn ngày thứ Hai của tuần đã bán.", "error");
    $("feedback-week").focus();
    return;
  }
  const fields = $("feedback-source").value === "manual"
    ? { applied_price: $("feedback-price").valueAsNumber, observed_qty: $("feedback-qty").valueAsNumber }
    : { source: "m5_history" };
  state.feedbackBusy = true;
  renderPricingOptions();
  showStatus("feedback-status", "Đang lưu phản hồi…");
  const sameContext = () => state.store === store_id && $("pricing-sku").value.trim().toUpperCase() === sku;
  try {
    const result = await post("/feedback", { sku, store_id, week, ...fields });
    if (sameContext()) {
      resetInputs();
      const observation = Number.isFinite(result.observed_qty) && Number.isFinite(result.applied_price)
        ? " · " + formatted(units, result.observed_qty) + " sản phẩm tại " + formatted(unitPrice, result.applied_price)
        : "";
      showStatus("feedback-status", "Đã lưu phản hồi cho tuần " + formatDate(week)
        + observation + ". Hãy tính lại nếu đang kiểm thử một tuần sau đó.", "ok");
    }
  } catch (error) {
    if (sameContext()) showStatus("feedback-status", error.message, "error");
  } finally {
    state.feedbackBusy = false;
    renderPricingOptions();
  }
}

function activateStore() {
  state.store = $("store").value;
  state.storeRevision += 1;
  state.detailRevision += 1;
  state.stats = null;
  state.catalog = [];
  state.detail = null;
  state.offset = 0;
  state.query = "";
  $("product-search").value = "";
  $("product-detail").hidden = true;
  $("pricing-sku").value = "";
  delete $("pricing-sku").dataset.edited;
  $("pricing-current").value = "";
  $("pricing-week").value = "";
  $("feedback-week").value = "";
  $("feedback-price").value = "";
  $("feedback-qty").value = "";
  seedPricingExample();
  resetInputs();
  setDefaultWeek();
  renderStats();
  showStatus("app-status", "Đang tải số liệu cửa hàng…");
  void loadStats(state.storeRevision);
  void loadCatalog();
}

function navigate(focus = false) {
  const page = location.hash.slice(1);
  const allowed = ["overview", "products", "statistics", "pricing"];
  state.view = allowed.includes(page) ? page : "overview";
  for (const view of allowed) {
    $("view-" + view).hidden = view !== state.view;
    const link = document.querySelector('nav a[href="#' + view + '"]');
    if (view === state.view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  updateCharts();
  if (focus) {
    window.scrollTo(0, 0);
    $(state.view + "-heading").focus({ preventScroll: true });
  }
}

async function loadStores() {
  try {
    const { stores } = await get("/stores");
    if (!stores.length) {
      showStatus("app-status", "Cơ sở dữ liệu chưa có cửa hàng M5.", "error");
      $("store").replaceChildren(new Option("Chưa có dữ liệu"));
      return;
    }
    $("store").replaceChildren(...stores.map(({ id }) => new Option(id, id)));
    $("store").disabled = false;
    activateStore();
  } catch (error) {
    $("store").replaceChildren(new Option("Mất kết nối"));
    showStatus("app-status", error.message, "error");
  }
}

async function loadModelInfo() {
  try {
    state.model = await get("/model-info");
    seedPricingExample();
    renderPricingOptions();
    setDefaultWeek();
  } catch {
    $("policy-status").textContent = "Chưa đọc được trạng thái mô hình. Hãy tải lại trang để kiểm tra PPO.";
  }
}

$("store").addEventListener("change", activateStore);
$("metric").addEventListener("change", updateCharts);
$("product-search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.store) return;
  state.query = $("product-search").value.trim();
  state.offset = 0;
  void loadCatalog();
});
$("products-previous").addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - pageSize);
  void loadCatalog();
});
$("products-next").addEventListener("click", () => {
  state.offset += pageSize;
  void loadCatalog();
});
$("detail-pricing").addEventListener("click", (event) => {
  $("pricing-sku").value = event.currentTarget.dataset.sku;
  $("pricing-current").value = "";
  resetInputs();
});
$("load-older-weeks").addEventListener("click", loadProductWeeks);
for (const id of ["pricing-sku", "pricing-week", "pricing-current", "strategy"]) {
  $(id).addEventListener(id === "pricing-sku" ? "input" : "change", (event) => {
    if (id === "pricing-sku") event.currentTarget.dataset.edited = "true";
    if (id === "pricing-week") {
      event.currentTarget.dataset.source = "user";
      showWeekNote("pricing-week", "pricing-date-note");
    }
    resetInputs();
  });
}
$("pricing-form").addEventListener("submit", calculate);
$("feedback-source").addEventListener("change", setFeedbackSource);
$("feedback-week").addEventListener("change", () => showWeekNote("feedback-week", "feedback-week-note"));
$("feedback-form").addEventListener("submit", saveFeedback);
window.addEventListener("hashchange", () => navigate(true));
window.addEventListener("resize", updateCharts);
navigate();
void loadStores();
void loadModelInfo();
