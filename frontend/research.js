const $ = (id) => document.getElementById(id);
const countFormat = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 0 });
const numberFormat = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 });
const priceFormat = new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const state = {
  forecast: null, primary: 'song_transformer_price', series: null, forecastRevision: 0,
  pricing: null, policy: 'ppo_feedback_gradient', replay: null, replayRevision: 0,
  scenario: null, scenarioLoading: false, scenarioRevision: 0,
  step: 0, timer: null
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

function format(value, digits = 2) {
  return new Intl.NumberFormat('vi-VN', { maximumFractionDigits: digits }).format(Number(value));
}

function signed(value, digits = 2) {
  const numeric = Number(value);
  return `${numeric > 0 ? '+' : ''}${format(numeric, digits)}%`;
}

function shortDate(value) {
  const [year, month, day] = String(value).slice(0, 10).split('-');
  return `${day}/${month}/${year}`;
}

async function api(path) {
  const response = await fetch(`/api/research${path}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `API trả về mã ${response.status}.`);
  $('api-status').textContent = 'Đã kết nối';
  return payload;
}

async function apiPost(path, body) {
  const response = await fetch(`/api/research${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg).join('; ') : payload.detail;
    throw new Error(detail || `API trả về mã ${response.status}.`);
  }
  $('api-status').textContent = 'Đã kết nối';
  return payload;
}

function showStatus(id, message, retry) {
  const node = $(id);
  node.hidden = false;
  node.innerHTML = retry
    ? `${escapeHtml(message)} <button type="button" data-retry="${retry}">Tải lại</button>`
    : escapeHtml(message);
}

function fillSelect(node, items, current) {
  node.innerHTML = items.map(({ value, label }) =>
    `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join('');
  if (items.some((item) => String(item.value) === String(current))) node.value = String(current);
}

function modelLabel(id) {
  return state.forecast?.models.find((model) => model.model === id)?.label || id;
}

function policyLabel(id) {
  return state.pricing?.policies.find((policy) => policy.id === id)?.label || id;
}

function setTab(which) {
  const selected = ['forecast', 'pricing', 'scenario'].includes(which) ? which : 'forecast';
  for (const tab of ['forecast', 'pricing', 'scenario']) {
    $(tab + '-tab').setAttribute('aria-selected', String(tab === selected));
    $(tab + '-tab').tabIndex = tab === selected ? 0 : -1;
    $(tab + '-view').hidden = tab !== selected;
  }
  if (selected !== 'pricing') stopReplay();
  if (selected === 'scenario' && !state.scenario && !state.scenarioLoading) loadScenarioOptions();
  history.replaceState(null, '', `#${selected}`);
}

function chartSvg(series, selectedIndex = -1, axisDigits = 0) {
  const all = series.flatMap((item) => item.values.filter((value) => Number.isFinite(value)));
  if (!all.length) return '<p>Không có dữ liệu để vẽ.</p>';
  const width = 760, height = 235, left = 47, right = 14, top = 17, bottom = 34;
  const low = Math.max(0, Math.min(...all) - (Math.max(...all) - Math.min(...all) || 1) * .09);
  const high = Math.max(...all) + (Math.max(...all) - Math.min(...all) || 1) * .12;
  const n = series[0].values.length;
  const x = (index) => left + index * (width - left - right) / Math.max(1, n - 1);
  const y = (value) => top + (high - value) / (high - low) * (height - top - bottom);
  const grid = Array.from({ length: 4 }, (_, index) => {
    const lineY = top + index * (height - top - bottom) / 3;
    const label = high - index * (high - low) / 3;
    return `<line x1="${left}" y1="${lineY.toFixed(1)}" x2="${width - right}" y2="${lineY.toFixed(1)}" stroke="#dce8e5"/><text x="${left - 7}" y="${(lineY + 4).toFixed(1)}" text-anchor="end" fill="#70888b" font-size="10">${format(label, axisDigits)}</text>`;
  }).join('');
  const lines = series.map((item) => {
    const points = item.values.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(' ');
    const circles = selectedIndex >= 0 && item.values[selectedIndex] !== undefined
      ? `<circle cx="${x(selectedIndex).toFixed(1)}" cy="${y(item.values[selectedIndex]).toFixed(1)}" r="5" fill="${item.color}" stroke="white" stroke-width="2"/>` : '';
    return `<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="${item.width || 2.5}" stroke-linecap="round" stroke-linejoin="round" ${item.dash ? 'stroke-dasharray="6 5"' : ''}/>${circles}`;
  }).join('');
  const cursor = selectedIndex >= 0
    ? `<line x1="${x(selectedIndex).toFixed(1)}" y1="${top}" x2="${x(selectedIndex).toFixed(1)}" y2="${height - bottom}" stroke="#a4b9ba" stroke-dasharray="3 4"/>` : '';
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">${grid}${cursor}${lines}</svg>`;
}

function renderRanking() {
  const models = state.forecast.models;
  const largest = Math.max(...models.map((model) => model.rmse_mean));
  $('ranking-meta').textContent = `${models.length} mô hình`;
  $('model-list').innerHTML = models.map((model, index) => `
    <button type="button" class="model-row ${model.model === state.primary ? 'selected' : ''}" data-model="${escapeHtml(model.model)}" aria-pressed="${model.model === state.primary}">
      <span class="model-rank">${index + 1}</span>
      <span class="model-body"><span class="model-name">${escapeHtml(model.label)}</span><span class="model-track"><i style="width:${(model.rmse_mean / largest * 100).toFixed(1)}%"></i></span></span>
      <span class="model-score"><strong>${format(model.rmse_mean)}</strong><small>${model.runs > 1 ? `± ${format(model.rmse_sd)}` : '1 lượt'}</small></span>
    </button>`).join('');
}

function syncForecastControls() {
  const overview = state.forecast;
  const skuSelect = $('forecast-sku');
  const oldSku = skuSelect.value;
  const preferred = [...overview.skus].sort((a, b) => b.test_weeks - a.test_weeks || a.sku.localeCompare(b.sku))[0]?.sku;
  fillSelect(skuSelect, overview.skus.map((item) => ({
    value: item.sku, label: `${item.sku} · ${item.test_weeks} tuần test`
  })), oldSku || preferred);
  const comparison = $('forecast-compare');
  const compareOptions = overview.models.filter((model) => model.model !== state.primary)
    .map((model) => ({ value: model.model, label: model.label }));
  const compare = compareOptions.some((option) => option.value === comparison.value)
    ? comparison.value : (compareOptions.find((option) => option.value === 'naive_last')?.value || compareOptions[0].value);
  fillSelect(comparison, compareOptions, compare);
  const first = overview.models.find((model) => model.model === state.primary);
  const second = overview.models.find((model) => model.model === comparison.value);
  const seeds = first.seeds.filter((seed) => second.seeds.includes(seed));
  fillSelect($('forecast-seed'), seeds.map((seed) => ({ value: seed, label: `Seed ${seed}` })),
             seeds.includes(Number($('forecast-seed').value)) ? $('forecast-seed').value : seeds[0]);
}

function renderPriceEffects() {
  $('price-effects').innerHTML = state.forecast.price_effects.map((effect) => {
    const model = effect.model_pair.split(' / ')[0];
    const label = { xgboost: 'XGBoost', lstm: 'LSTM', transformer: 'Transformer' }[model] || model;
    const delta = Number(effect.overall_rmse_change_price_minus_plain);
    return `<div class="price-effect-row"><strong>${escapeHtml(label)}</strong><span>Giá giúp ${effect.skus_improved_with_price}/${effect.skus} SKU</span><b class="${delta < 0 ? 'improved' : 'worsened'}">Δ RMSE ${delta > 0 ? '+' : ''}${format(delta)}</b></div>`;
  }).join('');
}

async function loadForecastOverview() {
  const revision = ++state.forecastRevision;
  const dataset = $('forecast-dataset').value;
  showStatus('forecast-status', 'Đang tải bảng dự báo…');
  $('forecast-content').hidden = true;
  try {
    const overview = await api(`/forecast/overview?dataset=${encodeURIComponent(dataset)}`);
    if (revision !== state.forecastRevision) return;
    state.forecast = overview;
    state.primary = overview.models.some((model) => model.model === state.primary)
      ? state.primary : 'song_transformer_price';
    $('forecast-test-count').textContent = `${countFormat.format(overview.test_rows)} SKU–tuần test`;
    $('forecast-source').textContent = `${overview.label} · ${overview.cohort}`;
    syncForecastControls();
    renderRanking();
    renderPriceEffects();
    $('forecast-status').hidden = true;
    $('forecast-content').hidden = false;
    await loadForecastSeries();
  } catch (error) {
    if (revision !== state.forecastRevision) return;
    $('api-status').textContent = 'Chưa kết nối';
    showStatus('forecast-status', error.message, 'forecast');
  }
}

async function loadForecastSeries() {
  if (!state.forecast) return;
  const revision = ++state.forecastRevision;
  const dataset = $('forecast-dataset').value;
  const sku = $('forecast-sku').value;
  const seed = $('forecast-seed').value;
  const compare = $('forecast-compare').value;
  $('forecast-chart').textContent = 'Đang tải các tuần test…';
  try {
    const query = new URLSearchParams({ dataset, sku, seed, models: `${state.primary},${compare}` });
    const series = await api(`/forecast/series?${query}`);
    if (revision !== state.forecastRevision) return;
    state.series = series;
    renderForecastSeries();
  } catch (error) {
    if (revision !== state.forecastRevision) return;
    $('forecast-chart').textContent = error.message;
    $('forecast-table').innerHTML = '';
  }
}

function renderForecastSeries() {
  const rows = state.series.rows;
  const main = state.primary, compare = $('forecast-compare').value;
  const metric = state.series.metrics.find((item) => item.model === main);
  $('forecast-primary-name').textContent = modelLabel(main);
  $('forecast-primary-rmse').textContent = metric ? `RMSE của SKU: ${format(metric.rmse_mean)}` : '—';
  $('forecast-th-primary').textContent = modelLabel(main);
  $('forecast-th-compare').textContent = modelLabel(compare);
  $('forecast-chart').innerHTML = chartSvg([
    { values: rows.map((row) => Number(row.actual)), color: '#183b45', width: 3.4 },
    { values: rows.map((row) => Number(row[main])), color: '#147d71', width: 2.7 },
    { values: rows.map((row) => Number(row[compare])), color: '#c77948', width: 2.4, dash: true }
  ]);
  $('forecast-chart').setAttribute('aria-label', `${rows.length} tuần test của ${state.series.sku}; so ${modelLabel(main)} và ${modelLabel(compare)} với lượng bán thực tế.`);
  $('forecast-range').textContent = `${shortDate(rows[0].week)} – ${shortDate(rows.at(-1).week)} · ${rows.length} tuần có nhãn`;
  $('forecast-table').innerHTML = rows.map((row) => `<tr>
    <td>${shortDate(row.week)}</td><td>${priceFormat.format(row.price)}</td>
    <td>${format(row.actual, 0)}</td><td>${format(row[main])}</td><td>${format(row[compare])}</td>
  </tr>`).join('');
}

function pricingSummaryRows() {
  return state.pricing.summary.filter((item) => item.simulator_variant === $('pricing-simulator').value)
    .sort((a, b) => b.uplift_vs_fixed_mean_pct - a.uplift_vs_fixed_mean_pct);
}

function renderPolicyList() {
  $('policy-list').innerHTML = pricingSummaryRows().map((item) => {
    const boundary = Number(item.at_train_min_mean_pct) + Number(item.at_train_max_mean_pct);
    return `<button type="button" class="policy-row ${item.policy === state.policy ? 'selected' : ''}" data-policy="${escapeHtml(item.policy)}" aria-pressed="${item.policy === state.policy}">
      <span><strong>${escapeHtml(policyLabel(item.policy))}</strong><small>${item.seeds} seed · chạm biên ${format(boundary, 1)}%</small></span>
      <span class="policy-score ${item.uplift_vs_fixed_mean_pct < 0 ? 'negative' : ''}">${signed(item.uplift_vs_fixed_mean_pct)}<small>± ${format(item.uplift_vs_fixed_sd_pct)} điểm</small></span>
    </button>`;
  }).join('');
}

function checkValue(split, model) {
  return state.pricing.observed_price_checks.find((row) => row.split === split && row.model === model);
}

function renderSimulatorEvidence() {
  const simulator = $('pricing-simulator').value;
  const model = simulator === 'lag_adjusted_kernel' ? 'selected_lag_kernel' : 'song_kernel';
  const validation = checkValue('validation', model), test = checkValue('test', model);
  const naiveValidation = checkValue('validation', 'naive_last'), naiveTest = checkValue('test', 'naive_last');
  $('simulator-evidence').innerHTML = `<div><span>RMSE validation tại giá đã bán</span><strong>${format(validation.rmse)} <small>so với naive ${format(naiveValidation.rmse)}</small></strong></div>
    <div><span>RMSE test tại giá đã bán</span><strong>${format(test.rmse)} <small>so với naive ${format(naiveTest.rmse)}</small></strong></div>
    <div><span>Đường cầu tăng theo giá</span><strong>${state.pricing.positive_price_response_skus}/${state.pricing.price_response_checked_skus} <small>chuỗi trên kernel hiệu chỉnh</small></strong></div>`;
}

async function loadPricingOverview() {
  showStatus('pricing-status', 'Đang tải kết quả định giá…');
  $('pricing-content').hidden = true;
  try {
    state.pricing = await api('/pricing/overview');
    fillSelect($('pricing-seed'), state.pricing.seeds.map((seed) => ({ value: seed, label: `Seed ${seed}` })), 42);
    fillSelect($('pricing-sku'), state.pricing.skus.map((sku) => ({ value: sku, label: sku })), state.pricing.skus[0]);
    renderSimulatorEvidence();
    renderPolicyList();
    $('pricing-status').hidden = true;
    $('pricing-content').hidden = false;
    await loadReplay();
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối';
    showStatus('pricing-status', error.message, 'pricing');
  }
}

async function loadReplay() {
  if (!state.pricing) return;
  stopReplay();
  const revision = ++state.replayRevision;
  const query = new URLSearchParams({
    simulator: $('pricing-simulator').value, policy: state.policy, seed: $('pricing-seed').value
  });
  $('replay-chart').textContent = 'Đang tải quyết định giá…';
  try {
    const result = await api(`/pricing/replay?${query}`);
    if (revision !== state.replayRevision) return;
    state.replay = result;
    state.step = 0;
    renderReplay();
  } catch (error) {
    if (revision !== state.replayRevision) return;
    $('replay-chart').textContent = error.message;
    $('replay-table').innerHTML = '';
  }
}

function replayRows() {
  return state.replay?.rows.filter((row) => row.sku === $('pricing-sku').value) || [];
}

function renderReplay() {
  const rows = replayRows();
  if (!rows.length) return;
  state.step = Math.min(state.step, rows.length - 1);
  const current = rows[state.step];
  $('replay-title').textContent = `Phát lại ${policyLabel(state.policy)}`;
  $('replay-subtitle').textContent = `${$('pricing-sku').value} · ${rows.length} tuần test đã lưu`;
  $('replay-total').textContent = numberFormat.format(state.replay.totals.simulated_revenue);
  $('replay-uplift').textContent = signed(state.replay.totals.uplift_pct);
  $('replay-uplift').classList.toggle('negative', state.replay.totals.uplift_pct < 0);
  $('replay-slider').max = String(rows.length - 1);
  $('replay-slider').value = String(state.step);
  $('replay-position').textContent = `${state.step + 1}/${rows.length}`;
  $('replay-prev').disabled = state.step === 0;
  $('replay-next').disabled = state.step === rows.length - 1;
  $('replay-chart').innerHTML = chartSvg([
    { values: rows.map((row) => Number(row.chosen_price)), color: '#147d71', width: 3 },
    { values: rows.map((row) => Number(row.fixed_price)), color: '#c77948', width: 2.4, dash: true }
  ], state.step, 2);
  $('replay-chart').setAttribute('aria-label', `Giá ${policyLabel(state.policy)} và giá cố định trong ${rows.length} tuần của ${$('pricing-sku').value}; đang ở tuần ${state.step + 1}.`);
  $('replay-week').innerHTML = `<strong>${shortDate(current.week)}</strong>
    <span>Giá chọn <b>${priceFormat.format(current.chosen_price)}</b></span>
    <span>Giá cố định <b>${priceFormat.format(current.fixed_price)}</b></span>
    <span>Bán mô phỏng <b>${format(current.simulated_units, 0)}</b></span>
    <span>Doanh thu mô phỏng <b>${numberFormat.format(current.simulated_revenue)}</b></span>`;
  $('replay-table').innerHTML = rows.map((row, index) => `<tr class="${index === state.step ? 'current' : ''}">
    <td><button type="button" data-step="${index}">${shortDate(row.week)}</button></td>
    <td>${priceFormat.format(row.chosen_price)}</td><td>${priceFormat.format(row.fixed_price)}</td>
    <td>${format(row.simulated_units, 0)}</td><td>${numberFormat.format(row.simulated_revenue)}</td>
  </tr>`).join('');
}

function stopReplay() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $('replay-play').textContent = 'Phát';
}

function toggleReplay() {
  if (state.timer) { stopReplay(); return; }
  const rows = replayRows();
  if (!rows.length) return;
  if (state.step === rows.length - 1) state.step = 0;
  $('replay-play').textContent = 'Tạm dừng';
  renderReplay();
  state.timer = setInterval(() => {
    if (state.step >= rows.length - 1) { stopReplay(); return; }
    state.step += 1;
    renderReplay();
  }, 850);
}

function selectedScenarioItem() {
  return state.scenario?.items.find((item) => item.sku === $('scenario-sku').value);
}

function invalidateScenario() {
  state.scenarioRevision += 1;
  $('scenario-result').hidden = true;
  $('scenario-details').hidden = true;
  $('scenario-empty').hidden = false;
  $('scenario-empty').textContent = 'Thông số đã thay đổi. Bấm “Tính giá bằng PPO” để xem quyết định mới.';
}

function syncScenarioWeek(resetWeek = false) {
  const item = selectedScenarioItem();
  if (!item) return;
  const previous = resetWeek ? null : $('scenario-week').value;
  fillSelect($('scenario-week'), item.weeks.map((week) => ({
    value: week.week, label: shortDate(week.week)
  })), previous || item.weeks[0].week);
  syncScenarioPrice();
}

function syncScenarioPrice() {
  const item = selectedScenarioItem();
  const week = item?.weeks.find((row) => row.week === $('scenario-week').value);
  if (!item || !week) return;
  const input = $('scenario-price');
  const online = $('scenario-policy').value === 'ppo_feedback_gradient';
  input.min = String(item.price_min);
  input.max = String(item.price_max);
  input.value = online ? '' : String(week.previous_price);
  input.placeholder = online ? 'Theo chuỗi mô phỏng' : 'Giá tuần trước';
  $('scenario-price-range').textContent = online
    ? `Miền train: ${format(item.price_min, 4)}–${format(item.price_max, 4)} USD · để trống để phát lại đúng giá đầu vào của chính sách`
    : `Miền train: ${format(item.price_min, 4)}–${format(item.price_max, 4)} USD · giá quan sát tuần trước: ${format(week.previous_price, 4)} USD`;
  $('scenario-form-note').textContent = online
    ? 'Các tuần test trước được mô phỏng nối tiếp. Transformer cập nhật sau mỗi 10 quyết định mô phỏng.'
    : 'Dùng 12 tuần lịch sử đã quan sát. Transformer được giữ cố định trong lần thử này.';
  invalidateScenario();
}

async function loadScenarioOptions() {
  if (state.scenarioLoading) return;
  state.scenarioLoading = true;
  showStatus('scenario-status', 'Đang đọc các tuần test M5…');
  $('scenario-content').hidden = true;
  try {
    const options = await api('/pricing/scenario/options');
    state.scenario = options;
    fillSelect($('scenario-policy'), options.policy_variants.map((item) => ({
      value: item.id, label: item.label
    })), 'ppo_feedback_gradient');
    fillSelect($('scenario-sku'), options.items.map((item) => ({ value: item.sku, label: item.sku })), options.items[0]?.sku);
    fillSelect($('scenario-seed'), options.seeds.map((seed) => ({ value: seed, label: `Seed ${seed}` })), 42);
    syncScenarioWeek(true);
    $('scenario-status').hidden = true;
    $('scenario-content').hidden = false;
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối';
    showStatus('scenario-status', error.message, 'scenario');
  } finally {
    state.scenarioLoading = false;
  }
}

function scenarioChartSvg(result) {
  const rows = result.curve;
  const revenues = rows.flatMap((row) => [row.forecast_revenue, row.simulator_revenue]);
  const minimum = Math.min(...revenues), maximum = Math.max(...revenues);
  const spread = maximum - minimum || 1;
  const low = Math.max(0, minimum - spread * .1), high = maximum + spread * .15;
  const width = 760, height = 245, left = 54, right = 18, top = 17, bottom = 35;
  const x = (price) => left + (price - result.price_min) / (result.price_max - result.price_min || 1) * (width - left - right);
  const y = (revenue) => top + (high - revenue) / (high - low || 1) * (height - top - bottom);
  const grid = Array.from({ length: 4 }, (_, index) => {
    const lineY = top + index * (height - top - bottom) / 3;
    const value = high - index * (high - low) / 3;
    return `<line x1="${left}" y1="${lineY.toFixed(1)}" x2="${width - right}" y2="${lineY.toFixed(1)}" stroke="#dce8e5"/><text x="${left - 7}" y="${(lineY + 4).toFixed(1)}" text-anchor="end" fill="#60787c" font-size="10">${format(value, 0)}</text>`;
  }).join('');
  const line = (key, color, dash = false) => {
    const points = rows.map((row) => `${x(row.price).toFixed(1)},${y(row[key]).toFixed(1)}`).join(' ');
    return `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="${dash ? 2.3 : 3}" stroke-linecap="round" stroke-linejoin="round" ${dash ? 'stroke-dasharray="6 5"' : ''}/>`;
  };
  const marker = (price, revenue, color, dash = false) => `<line x1="${x(price).toFixed(1)}" y1="${top}" x2="${x(price).toFixed(1)}" y2="${height - bottom}" stroke="${color}" stroke-width="1.6" ${dash ? 'stroke-dasharray="3 4"' : ''}/><circle cx="${x(price).toFixed(1)}" cy="${y(revenue).toFixed(1)}" r="5.5" fill="${color}" stroke="white" stroke-width="2"/>`;
  const ticks = [result.price_min, (result.price_min + result.price_max) / 2, result.price_max]
    .map((price) => `<text x="${x(price).toFixed(1)}" y="${height - 10}" text-anchor="middle" fill="#60787c" font-size="10">${format(price, 2)}</text>`).join('');
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">${grid}${line('simulator_revenue', '#c77948', true)}${line('forecast_revenue', '#147d71')}${marker(result.current_price, result.forecast.revenue_current, '#36535a', true)}${marker(result.recommended_price, result.forecast.revenue_recommended, '#0a806a')}${ticks}</svg>`;
}

function renderScenario(result) {
  const online = result.policy === 'ppo_feedback_gradient';
  $('scenario-decision-meta').textContent = `${result.sku} · ${shortDate(result.week)} · seed ${result.seed}`;
  $('scenario-current-price').textContent = `${format(result.current_price, 4)} USD`;
  $('scenario-recommended-price').textContent = `${format(result.recommended_price, 4)} USD`;
  const priceDirection = result.price_change_pct > 0 ? 'Tăng' : result.price_change_pct < 0 ? 'Giảm' : 'Giữ nguyên';
  $('scenario-price-change').textContent = `${priceDirection} ${format(Math.abs(result.price_change_pct), 2)}% so với giá đầu vào`;
  $('scenario-policy-context').textContent = online
    ? `${result.policy_label}: phát lại ${result.prior_simulated_decisions} tuần, cập nhật Transformer ${result.forecast_updates_before_week} lần trước tuần chọn. ${result.current_price_source === 'user_input' ? 'Giá đầu vào do bạn nhập.' : result.current_price_source === 'train_mean_initial' ? 'Giá khởi đầu lấy từ trung bình train.' : 'Giá đầu vào từ chuỗi mô phỏng.'}`
    : 'PPO dự báo cố định: dùng lịch sử quan sát, không cập nhật Transformer.';
  $('scenario-revenue-current').textContent = `${format(result.forecast.revenue_current)} USD`;
  $('scenario-revenue-recommended').textContent = `${format(result.forecast.revenue_recommended)} USD`;
  $('scenario-units-current').textContent = `Lượng bán dự báo: ${format(result.forecast.units_current)} đơn vị`;
  $('scenario-units-recommended').textContent = `Lượng bán dự báo: ${format(result.forecast.units_recommended)} đơn vị`;
  const delta = result.forecast.revenue_change;
  const deltaPct = result.forecast.revenue_change_pct;
  $('scenario-revenue-change').textContent = `Chênh lệch doanh thu dự báo: ${delta > 0 ? '+' : ''}${format(delta)} USD${deltaPct === null ? '' : ` (${signed(deltaPct)})`}`;
  $('scenario-revenue-change').classList.toggle('negative', delta < 0);
  $('scenario-chart').innerHTML = scenarioChartSvg(result);
  $('scenario-chart').setAttribute('aria-label', `Doanh thu ước tính theo 41 mức giá từ ${format(result.price_min, 2)} đến ${format(result.price_max, 2)} USD của ${result.sku}. Hai đường là Transformer và simulator; đánh dấu giá hiện tại và giá PPO chọn.`);
  $('scenario-chart-range').textContent = `Giá trong tập train: ${format(result.price_min, 4)}–${format(result.price_max, 4)} USD/đơn vị`;
  const simulated = result.simulator_expected;
  const simulatedPct = simulated.revenue_current > 0
    ? 100 * (simulated.revenue_recommended / simulated.revenue_current - 1) : null;
  $('scenario-simulator-summary').textContent = `Simulator kỳ vọng: ${format(simulated.revenue_current)} → ${format(simulated.revenue_recommended)} USD${simulatedPct === null ? '' : ` (${signed(simulatedPct)})`}. Đây là kết quả mô phỏng.`;
  $('scenario-simulator-summary').classList.toggle('negative', simulated.revenue_change < 0);
  const context = result.context;
  $('scenario-history-note').textContent = online
    ? `${context.simulated_history_weeks} tuần gần nhất là kết quả mô phỏng; các tuần trước đó là lịch sử quan sát. Không dùng lượng bán thật của tuần đang thử.`
    : 'Cả 12 tuần là lịch sử đã quan sát trước tuần đang thử.';
  const previousLabel = context.simulated_history_weeks > 0 ? 'Tuần trước · mô phỏng' : 'Tuần trước · quan sát';
  $('scenario-context-meta').innerHTML = `<div><span>${previousLabel}</span><strong>${format(context.previous_week_price, 4)} USD</strong><small>${format(context.previous_week_units)} đơn vị bán</small></div><div><span>Tuần đang thử</span><strong>${format(context.event_days, 0)} ngày sự kiện</strong><small>${format(context.snap_days, 0)} ngày SNAP</small></div><div><span>Độ nhạy giá dự báo</span><strong>${format(result.forecast.demand_gradient_at_current_price, 2)}</strong><small>đơn vị cho mỗi 1 USD tăng giá</small></div>`;
  $('scenario-history').innerHTML = context.history.map((row) => `<tr><td>${row.weeks_before} tuần</td><td>${format(row.price, 4)}</td><td>${format(row.units)}</td></tr>`).join('');
  $('scenario-empty').hidden = true;
  $('scenario-result').hidden = false;
  $('scenario-details').hidden = false;
}

async function runScenario() {
  const revision = ++state.scenarioRevision;
  $('scenario-submit').disabled = true;
  $('scenario-submit').textContent = 'Đang tính giá…';
  $('scenario-result').hidden = true;
  $('scenario-details').hidden = true;
  $('scenario-empty').hidden = false;
  $('scenario-empty').textContent = 'Đang tính một quyết định giá từ lịch sử đã quan sát…';
  try {
    const result = await apiPost('/pricing/scenario', {
      sku: $('scenario-sku').value, week: $('scenario-week').value,
      current_price: $('scenario-price').value === '' ? null : Number($('scenario-price').value),
      seed: Number($('scenario-seed').value), policy_variant: $('scenario-policy').value
    });
    if (revision !== state.scenarioRevision) return;
    renderScenario(result);
  } catch (error) {
    if (revision !== state.scenarioRevision) return;
    $('scenario-empty').textContent = error.message;
  } finally {
    $('scenario-submit').disabled = false;
    $('scenario-submit').textContent = 'Tính giá bằng PPO';
  }
}

function init() {
  $('forecast-tab').addEventListener('click', () => setTab('forecast'));
  $('pricing-tab').addEventListener('click', () => setTab('pricing'));
  $('scenario-tab').addEventListener('click', () => setTab('scenario'));
  document.querySelector('.research-switch').addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const tabs = ['forecast', 'pricing', 'scenario'];
    const active = tabs.findIndex((tab) => $(tab + '-tab').getAttribute('aria-selected') === 'true');
    const target = tabs[(active + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    setTab(target);
    $(target + '-tab').focus();
  });
  $('forecast-dataset').addEventListener('change', loadForecastOverview);
  $('model-list').addEventListener('click', (event) => {
    const row = event.target.closest('[data-model]');
    if (!row) return;
    state.primary = row.dataset.model;
    syncForecastControls();
    renderRanking();
    loadForecastSeries();
  });
  ['forecast-sku', 'forecast-seed', 'forecast-compare'].forEach((id) =>
    $(id).addEventListener('change', () => { syncForecastControls(); loadForecastSeries(); }));
  $('pricing-simulator').addEventListener('change', () => {
    renderPolicyList(); renderSimulatorEvidence(); loadReplay();
  });
  $('policy-list').addEventListener('click', (event) => {
    const row = event.target.closest('[data-policy]');
    if (!row) return;
    state.policy = row.dataset.policy;
    renderPolicyList();
    loadReplay();
  });
  $('pricing-seed').addEventListener('change', loadReplay);
  $('pricing-sku').addEventListener('change', () => { stopReplay(); state.step = 0; renderReplay(); });
  $('replay-slider').addEventListener('input', (event) => { stopReplay(); state.step = Number(event.target.value); renderReplay(); });
  $('replay-prev').addEventListener('click', () => { stopReplay(); state.step -= 1; renderReplay(); });
  $('replay-next').addEventListener('click', () => { stopReplay(); state.step += 1; renderReplay(); });
  $('replay-play').addEventListener('click', toggleReplay);
  $('replay-table').addEventListener('click', (event) => {
    const row = event.target.closest('[data-step]');
    if (!row) return;
    stopReplay(); state.step = Number(row.dataset.step); renderReplay();
  });
  $('scenario-sku').addEventListener('change', () => syncScenarioWeek(true));
  $('scenario-policy').addEventListener('change', syncScenarioPrice);
  $('scenario-week').addEventListener('change', syncScenarioPrice);
  $('scenario-price').addEventListener('input', invalidateScenario);
  $('scenario-seed').addEventListener('change', invalidateScenario);
  $('scenario-form').addEventListener('submit', (event) => { event.preventDefault(); runScenario(); });
  ['forecast-status', 'pricing-status', 'scenario-status'].forEach((id) => $(id).addEventListener('click', (event) => {
    const target = event.target.closest('[data-retry]');
    if (target?.dataset.retry === 'forecast') loadForecastOverview();
    if (target?.dataset.retry === 'pricing') loadPricingOverview();
    if (target?.dataset.retry === 'scenario') loadScenarioOptions();
  }));
  setTab(['#forecast', '#pricing', '#scenario'].includes(location.hash) ? location.hash.slice(1) : 'scenario');
  loadForecastOverview();
  loadPricingOverview();
}

document.addEventListener('DOMContentLoaded', init);
