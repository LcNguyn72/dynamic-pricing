const state = {
  products: [], selectedSku: null, activeTier: 'all',
  lastResult: null, loadToken: 0, scenarioRevision: 0,
  rankingBySku: new Map(), rankingStatus: 'idle', rankingError: '', rankingToken: 0
};

const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 0 });
const price = new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const signed = (value) => `${value > 0 ? '+' : ''}${new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 }).format(value)}%`;
const labels = { re: 'Rẻ', vua_thap: 'Vừa thấp', vua_cao: 'Vừa cao', dat: 'Đắt' };

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
}

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail) ? payload.detail.map((d) => d.msg).join('; ') : payload.detail;
    throw new Error(detail || `Yêu cầu thất bại (${response.status})`);
  }
  return payload;
}

async function selectedProducts() {
  const response = await fetch('/selected-products.json');
  if (!response.ok) throw new Error('Không tải được danh sách sản phẩm demo.');
  return response.json();
}

let toastTimer;
function toast(message, isError = false) {
  const node = $('toast');
  node.textContent = message;
  node.classList.toggle('error', isError);
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 4600);
}

function productBySku(sku) { return state.products.find((item) => item.sku === sku); }

function renderOverview(health) {
  $('metric-products').textContent = number.format(state.products.length);
  $('metric-weeks').textContent = number.format(state.products.reduce((total, item) => total + item.weeks, 0));
  $('metric-model-products').textContent = number.format(health.n_products);
}

function renderProductOptions() {
  $('product-select').innerHTML = state.products.map((item) =>
    `<option value="${escapeHtml(item.sku)}">${escapeHtml(item.sku)} · ${escapeHtml(item.description)}</option>`
  ).join('');
}

function referencePrice(item) { return Number(Number(item.reference_price).toFixed(2)); }

function updatePriceSource() {
  const item = productBySku(state.selectedSku);
  if (!item) return;
  const reference = referencePrice(item);
  const usingReference = Number($('current-price').value) === reference;
  $('price-reference').textContent = usingReference
    ? `Điền sẵn từ giá tham chiếu mô hình: ${price.format(reference)}`
    : `Giá tham chiếu mô hình: ${price.format(reference)}`;
  $('reset-price').hidden = usingReference;
}

function rankingValue(result, sort) {
  if (!result || result.error) return -Infinity;
  if (sort === 'gain') return Number(result.confidence?.expected_gain_pct) || 0;
  if (sort === 'probability') return Number(result.confidence?.prob_gain) || 0;
  return Number(result.priority_score) || 0;
}

function renderCatalog() {
  const ready = state.rankingStatus === 'ready';
  $('ranking-table').hidden = !ready;
  $('ranking-state').hidden = ready;
  $('catalog-empty').hidden = true;
  if (!ready) {
    $('ranking-count').textContent = '';
    $('ranking-state').innerHTML = state.rankingStatus === 'error'
      ? `${escapeHtml(state.rankingError)} <button type="button" class="text-button" data-retry>Thử lại</button>`
      : 'Đang tính đề xuất cho 20 sản phẩm…';
    return;
  }
  const query = $('product-search').value.trim().toLocaleLowerCase('vi');
  const sort = $('ranking-sort').value;
  const ranked = state.products.map((item, index) => ({ item, index,
    result: state.rankingBySku.get(item.sku) })).sort((a, b) =>
    rankingValue(b.result, sort) - rankingValue(a.result, sort) ||
    rankingValue(b.result, 'priority') - rankingValue(a.result, 'priority') || a.index - b.index
  ).map((row, index) => ({ ...row, rank: row.result?.error ? '—' : index + 1 }));
  const visible = ranked.filter(({ item }) =>
    (state.activeTier === 'all' || item.tier === state.activeTier) &&
    (`${item.sku} ${item.description}`.toLocaleLowerCase('vi').includes(query))
  );
  $('ranking-count').textContent = `${visible.length} / ${state.products.length} sản phẩm`;
  $('product-table').innerHTML = visible.map(({ item, result, rank }) => {
    const failed = !result || result.error;
    const probability = result?.confidence?.prob_gain;
    return `<tr class="${item.sku === state.selectedSku ? 'selected' : ''} ${rank <= 3 ? 'top-rank' : ''}">
      <td class="rank-cell" data-label="Hạng">${rank}</td>
      <td class="product-cell"><strong title="${escapeHtml(item.description)}">${escapeHtml(item.description)}</strong><small>SKU ${escapeHtml(item.sku)} <span class="tier-tag ${escapeHtml(item.tier)}">${escapeHtml(labels[item.tier] || item.tier)}</span></small>${failed ? `<small class="row-error" title="${escapeHtml(result?.error || '')}">Không tính được đề xuất</small>` : ''}</td>
      <td class="numeric" data-label="Giá tham chiếu">${price.format(referencePrice(item))}</td>
      <td class="numeric recommendation-cell" data-label="Giá đề xuất">${failed ? '—' : price.format(result.recommended_price)}</td>
      <td class="numeric gain-cell ${!failed && result.confidence.expected_gain_pct > 0 ? 'positive' : ''}" data-label="Tăng doanh thu dự kiến">${failed ? '—' : signed(result.confidence.expected_gain_pct)}</td>
      <td class="numeric" data-label="Xác suất tăng">${failed || probability == null ? '—' : `${Math.round(probability * 100)}%`}</td>
      <td class="action-cell"><span class="row-actions"><button type="button" class="table-action" data-sku="${escapeHtml(item.sku)}" aria-label="Thử giá ${escapeHtml(item.description)}, SKU ${escapeHtml(item.sku)}">Thử giá</button><a class="table-action" href="/product.html?sku=${encodeURIComponent(item.sku)}" aria-label="Xem lịch sử ${escapeHtml(item.description)}, SKU ${escapeHtml(item.sku)}">Lịch sử</a></span></td>
    </tr>`;
  }).join('');
  $('catalog-empty').hidden = visible.length > 0;
}

async function refreshRanking() {
  const token = ++state.rankingToken;
  state.rankingStatus = 'loading';
  state.rankingBySku = new Map();
  const day = $('scenario-date').value;
  const maxChange = Number($('max-change').value);
  const displayDate = day ? day.split('-').reverse().join('/') : '—';
  $('ranking-context').textContent = `Dùng giá tham chiếu của từng sản phẩm, ngày ${displayDate}, giới hạn ±${Math.round(maxChange * 100)}%.`;
  renderCatalog();
  if (!$('scenario-date').checkValidity()) {
    state.rankingStatus = 'error';
    state.rankingError = 'Chọn ngày mô phỏng hợp lệ để so sánh.';
    renderCatalog();
    return;
  }
  try {
    const items = state.products.map((item) => ({ sku: item.sku,
      current_price: referencePrice(item), date: day, max_change: maxChange }));
    const payload = await api('/recommend/batch', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items }) });
    if (token !== state.rankingToken) return;
    const requestedSkus = new Set(items.map((item) => item.sku));
    if (!Array.isArray(payload.results) || payload.results.length !== items.length ||
        new Set(payload.results.map((result) => String(result.sku))).size !== items.length ||
        payload.results.some((result) => !requestedSkus.has(String(result.sku)))) {
      throw new Error('API trả về danh sách so sánh không đầy đủ.');
    }
    state.rankingBySku = new Map(payload.results.map((result) => [String(result.sku), result]));
    state.rankingStatus = 'ready';
  } catch (error) {
    if (token !== state.rankingToken) return;
    state.rankingStatus = 'error';
    state.rankingError = `Không tính được bảng so sánh: ${error.message}`;
  }
  renderCatalog();
}

function gridLines(width, left, right, top, bottom, rows = 3) {
  const innerHeight = bottom - top;
  return Array.from({ length: rows + 1 }, (_, i) => {
    const y = top + innerHeight * i / rows;
    return `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="#e9eef4" stroke-width="1"/>`;
  }).join('');
}

function linePath(values, min, max, left, top, width, height) {
  const span = max - min || 1;
  return values.map((value, i) => {
    const x = left + i * width / Math.max(1, values.length - 1);
    const y = top + height - (value - min) / span * height;
    return `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function chartMarker(rows, values, min, max, target, color) {
  if (target == null) return '';
  const left = 12, right = 12, top = 20, bottom = 158;
  const index = rows.reduce((best, row, i) => Math.abs(row.price - target) < Math.abs(rows[best].price - target) ? i : best, 0);
  const x = left + index * (480 - left - right) / Math.max(1, rows.length - 1);
  const y = top + (bottom - top) - (values[index] - min) / (max - min || 1) * (bottom - top);
  return `<line x1="${x.toFixed(1)}" y1="${top}" x2="${x.toFixed(1)}" y2="${bottom}" stroke="${color}" stroke-dasharray="4 4" opacity=".65"/>
    <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${color}" stroke="white" stroke-width="2"/>`;
}

function renderDemandChart(curve) {
  const rows = curve.points || [];
  if (!rows.length) { $('demand-chart').textContent = 'Chưa có đường cầu.'; return; }
  const values = rows.map((row) => row.units);
  const min = Math.min(...values) * .97, max = Math.max(...values) * 1.03;
  const chartWidth = 480, left = 12, right = 12, top = 20, bottom = 158;
  const path = linePath(values, min, max, left, top, chartWidth - left - right, bottom - top);
  $('demand-chart').setAttribute('aria-label', `Lượng bán dự báo từ ${number.format(Math.min(...values))} đến ${number.format(Math.max(...values))} đơn vị mỗi tuần theo giá.`);
  $('demand-chart').innerHTML = `<svg viewBox="0 0 480 178" preserveAspectRatio="none" aria-hidden="true">
    ${gridLines(chartWidth, left, right, top, bottom)}
    <path d="${path}" fill="none" stroke="#8998d5" stroke-width="3" stroke-linejoin="round"/>
    ${chartMarker(rows, values, min, max, curve.current_price, '#91a2b3')}
    ${chartMarker(rows, values, min, max, state.lastResult?.recommended_price, '#e5a35d')}
  </svg>`;
  $('demand-scale').textContent = `Dự báo ${number.format(Math.min(...values))}–${number.format(Math.max(...values))} đơn vị mỗi tuần`;
  $('demand-start').textContent = `Giá ${price.format(rows[0].price)}`;
  $('demand-end').textContent = `Giá ${price.format(rows.at(-1).price)}`;
  $('demand-range').textContent = `±${Math.round(Number($('max-change').value) * 100)}%`;
}

function renderCurveChart(curve) {
  const rows = curve.points || [];
  if (!rows.length) { $('curve-chart').textContent = 'Chưa có đường doanh thu.'; return; }
  const values = rows.map((row) => row.revenue);
  const min = Math.min(...values) * .97, max = Math.max(...values) * 1.03;
  const chartWidth = 480, left = 12, right = 12, top = 20, bottom = 158;
  const innerWidth = chartWidth - left - right, innerHeight = bottom - top;
  const path = linePath(values, min, max, left, top, innerWidth, innerHeight);
  const area = `${path} L${chartWidth - right},${bottom} L${left},${bottom} Z`;
  const pMin = rows[0].price, pMax = rows.at(-1).price;
  $('curve-chart').innerHTML = `<svg viewBox="0 0 480 178" preserveAspectRatio="none" aria-hidden="true">
    <defs><linearGradient id="curveFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#3cbfa0" stop-opacity=".23"/><stop offset="100%" stop-color="#3cbfa0" stop-opacity="0"/></linearGradient></defs>
    ${gridLines(chartWidth, left, right, top, bottom)}<path d="${area}" fill="url(#curveFill)"/>
    <path d="${path}" fill="none" stroke="#29ac90" stroke-width="3" stroke-linejoin="round"/>
    ${chartMarker(rows, values, min, max, curve.current_price, '#91a2b3')}
    ${chartMarker(rows, values, min, max, state.lastResult?.recommended_price, '#e5a35d')}
  </svg>`;
  $('curve-start').textContent = `Giá ${price.format(pMin)}`;
  $('curve-end').textContent = `Giá ${price.format(pMax)}`;
  $('curve-range').textContent = `±${Math.round(Number($('max-change').value) * 100)}%`;
}

async function updateCurve() {
  if (!state.selectedSku) return;
  const sku = state.selectedSku;
  const token = state.loadToken;
  const revision = state.scenarioRevision;
  const day = $('scenario-date').value;
  const params = new URLSearchParams({ date: day, max_change: $('max-change').value });
  const current = Number($('current-price').value);
  if (!(current > 0)) return;
  params.set('current_price', String(current));
  const curve = await api(`/demand_curve/${encodeURIComponent(sku)}?${params}`);
  if (token !== state.loadToken || revision !== state.scenarioRevision || sku !== state.selectedSku ||
      day !== $('scenario-date').value || current !== Number($('current-price').value) ||
      params.get('max_change') !== $('max-change').value) return;
  renderDemandChart(curve);
  renderCurveChart(curve);
}

function clearResult() {
  state.scenarioRevision += 1;
  state.lastResult = null;
  $('result-empty').hidden = false;
  $('result-content').hidden = true;
  $('result-badge').hidden = true;
  $('result-badge').className = 'result-badge';
  document.querySelector('.result-detail').open = false;
}

async function selectProduct(sku, batchResult = null) {
  const item = productBySku(sku);
  if (!item) return;
  ++state.loadToken;
  state.selectedSku = sku;
  $('product-select').value = sku;
  $('current-price').value = referencePrice(item).toFixed(2);
  updatePriceSource();
  $('chart-product-name').textContent = `${item.description} · SKU ${item.sku}`;
  renderCatalog();
  clearResult();
  if (batchResult && !batchResult.error) renderResult(batchResult);
  $('demand-chart').textContent = 'Đang tải đường cầu…';
  $('curve-chart').textContent = 'Đang tải đường doanh thu…';
  try { await updateCurve(); } catch (error) { toast(error.message, true); }
}

function renderResult(result) {
  state.lastResult = result;
  $('result-empty').hidden = true;
  $('result-content').hidden = false;
  const action = { lower: 'Giảm giá', raise: 'Tăng giá', keep: 'Giữ nguyên' }[result.action] || result.action;
  $('result-badge').textContent = action;
  $('result-badge').hidden = false;
  $('result-badge').className = `result-badge ${result.action}`;
  $('result-current').textContent = price.format(result.current_price);
  $('result-recommended').textContent = price.format(result.recommended_price);
  $('result-action').textContent = action;
  $('result-change').textContent = signed(result.price_change_pct);
  $('result-change').classList.toggle('negative', result.price_change_pct < 0);
  $('result-units-current').textContent = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 }).format(result.expected.units_current);
  $('result-units-new').textContent = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 }).format(result.expected.units_recommended);
  $('result-revenue-current').textContent = price.format(result.expected.revenue_current);
  $('result-revenue-new').textContent = price.format(result.expected.revenue_recommended);
  $('result-gain').textContent = signed(result.confidence.expected_gain_pct);
  $('result-probability').textContent = result.confidence.prob_gain == null
    ? '—' : `${Math.round(result.confidence.prob_gain * 100)}%`;
  $('result-reason').textContent = String(result.reason)
    .replace(/Elasticity/g, 'Độ co giãn giá')
    .replace(/\bsd\b/g, 'độ bất định')
    .replace(/mức strong/g, 'mức mạnh')
    .replace(/mức moderate/g, 'mức trung bình')
    .replace(/mức weak/g, 'mức thận trọng')
    .replace(/10-90%:/g, 'khoảng 10–90%:')
    .replace(/ \.\. /g, ' đến ');
  $('result-warnings').innerHTML = result.warnings
    .filter((warning) => !warning.startsWith('Không có unit_cost:'))
    .map((warning) => `<div class="warning">${escapeHtml(warning)}</div>`).join('');
  $('saved-simulation').hidden = !result.recommendation_id;
  if (result.recommendation_id) {
    $('saved-simulation').href = `/product.html?sku=${encodeURIComponent(result.sku)}#activity`;
  }
}

async function submitScenario(event) {
  event.preventDefault();
  const sku = state.selectedSku;
  const current = Number($('current-price').value);
  if (!(current > 0)) { toast('Giá hiện tại phải lớn hơn 0.', true); return; }
  const body = { sku, date: $('scenario-date').value, current_price: current,
    max_change: Number($('max-change').value) };
  const token = state.loadToken;
  const revision = state.scenarioRevision;
  const stillCurrent = () => token === state.loadToken && revision === state.scenarioRevision && sku === state.selectedSku &&
    body.date === $('scenario-date').value && body.max_change === Number($('max-change').value) &&
    current === Number($('current-price').value);
  const button = $('submit-button');
  button.disabled = true;
  button.firstElementChild.textContent = 'Đang tính toán…';
  try {
    const result = await api('/demo/simulations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (stillCurrent()) {
      renderResult(result);
      await updateCurve();
    } else {
      toast('Kịch bản đã thay đổi trong lúc tính. Hãy tính lại để xem kết quả mới.');
    }
  } catch (error) { if (stillCurrent()) toast(error.message, true); }
  finally { button.disabled = false; button.firstElementChild.textContent = 'Tính giá đề xuất'; }
}

function bindEvents() {
  document.querySelectorAll('#tier-filters .chip').forEach((node) =>
    node.setAttribute('aria-pressed', String(node.dataset.tier === state.activeTier)));
  $('product-select').addEventListener('change', (event) => selectProduct(event.target.value));
  $('scenario-date').addEventListener('change', () => {
    clearResult();
    updateCurve().catch((error) => toast(error.message, true));
    refreshRanking();
  });
  $('current-price').addEventListener('input', () => {
    clearResult();
    updatePriceSource();
  });
  $('current-price').addEventListener('change', () => updateCurve().catch((error) => toast(error.message, true)));
  $('reset-price').addEventListener('click', () => {
    const item = productBySku(state.selectedSku);
    if (!item) return;
    $('current-price').value = referencePrice(item).toFixed(2);
    updatePriceSource();
    clearResult();
    updateCurve().catch((error) => toast(error.message, true));
  });
  $('max-change').addEventListener('change', () => {
    clearResult();
    updateCurve().catch((error) => toast(error.message, true));
    refreshRanking();
  });
  $('scenario-form').addEventListener('submit', submitScenario);
  $('product-search').addEventListener('input', renderCatalog);
  $('ranking-sort').addEventListener('change', renderCatalog);
  $('ranking-state').addEventListener('click', (event) => {
    if (event.target.closest('[data-retry]')) refreshRanking();
  });
  $('tier-filters').addEventListener('click', (event) => {
    const chip = event.target.closest('[data-tier]');
    if (!chip) return;
    state.activeTier = chip.dataset.tier;
    document.querySelectorAll('#tier-filters .chip').forEach((node) => {
      node.classList.toggle('active', node === chip);
      node.setAttribute('aria-pressed', String(node === chip));
    });
    renderCatalog();
  });
  $('product-table').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-sku]');
    if (!button) return;
    selectProduct(button.dataset.sku, state.rankingBySku.get(button.dataset.sku));
    document.querySelector('#simulator').scrollIntoView({ behavior: 'smooth' });
  });
  document.querySelectorAll('.nav-link').forEach((link) => link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((node) => node.classList.toggle('active', node === link));
  }));
}

async function initialize() {
  bindEvents();
  try {
    const [health, modelProducts, selected] = await Promise.all([
      api('/health'), api('/products'), selectedProducts()
    ]);
    if (health.status !== 'ok') throw new Error('API mô hình chưa sẵn sàng.');
    if (!Array.isArray(selected) || selected.length !== 20 ||
        new Set(selected.map((item) => String(item.sku))).size !== 20) {
      throw new Error('Danh sách chọn mẫu cần đúng 20 SKU không trùng nhau.');
    }
    const bySku = new Map(modelProducts.map((item) => [String(item.sku), item]));
    state.products = selected.map((row) => {
      const modelProduct = bySku.get(String(row.sku));
      if (!modelProduct) throw new Error(`SKU ${row.sku} chưa có tham số trong mô hình.`);
      return { ...modelProduct, description: String(modelProduct.description).trim(),
        tier: row.tier, weeks: Number(row.weeks), n_levels: Number(row.n_levels) };
    });
    $('scenario-date').min = '2009-11-30';
    $('scenario-date').max = '2011-12-05';
    $('scenario-date').value = '2011-11-15';
    $('api-status').textContent = 'Đã kết nối';
    renderOverview(health);
    renderProductOptions();
    const requestedSku = new URLSearchParams(window.location.search).get('sku');
    const initialSku = requestedSku && productBySku(requestedSku) ? requestedSku : state.products[0].sku;
    await Promise.all([selectProduct(initialSku), refreshRanking()]);
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối';
    toast(`Không tải được ứng dụng: ${error.message}`, true);
  }
}

initialize();
