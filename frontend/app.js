const state = {
  products: [], selectedSku: null, activeTier: 'all',
  lastResult: null, loadToken: 0, scenarioRevision: 0
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
  if (!response.ok) throw new Error('Không đọc được danh sách 20 sản phẩm của bước 3.');
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

function renderProductSummary(item) {
  $('product-summary').innerHTML = `<div><strong>${escapeHtml(item.description)}</strong>
    <span>SKU ${escapeHtml(item.sku)} · ${item.weeks} tuần có bán · ${item.n_levels} mức giá</span></div>
    <span class="tag">${escapeHtml(labels[item.tier] || item.tier)}</span>`;
  $('chart-product-name').textContent = `${item.description} · SKU ${item.sku}`;
}

function renderCatalog() {
  const query = $('product-search').value.trim().toLocaleLowerCase('vi');
  const visible = state.products.filter((item) =>
    (state.activeTier === 'all' || item.tier === state.activeTier) &&
    (`${item.sku} ${item.description}`.toLocaleLowerCase('vi').includes(query))
  );
  $('product-table').innerHTML = visible.map((item) => `<tr class="${item.sku === state.selectedSku ? 'selected' : ''}">
    <td><strong title="${escapeHtml(item.description)}">${escapeHtml(item.description)}</strong><small>SKU ${escapeHtml(item.sku)}</small></td>
    <td><span class="tier-tag ${escapeHtml(item.tier)}">${escapeHtml(labels[item.tier] || item.tier)}</span></td>
    <td>${price.format(item.reference_price)}</td><td>${number.format(item.weeks)}</td>
    <td>${new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 }).format(item.elasticity_mean)}</td>
    <td><button type="button" class="table-action" data-sku="${escapeHtml(item.sku)}" aria-label="Thử giá sản phẩm ${escapeHtml(item.description)}, SKU ${escapeHtml(item.sku)}">Thử giá →</button></td></tr>`).join('');
  $('catalog-empty').hidden = visible.length > 0;
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
  $('result-badge').textContent = 'Chưa tính';
  $('result-badge').className = 'result-badge';
}

async function selectProduct(sku) {
  const item = productBySku(sku);
  if (!item) return;
  ++state.loadToken;
  state.selectedSku = sku;
  $('product-select').value = sku;
  $('current-price').value = Number(item.reference_price).toFixed(2);
  $('price-source-label').textContent = 'Giá tham chiếu do API mô hình trả về';
  renderProductSummary(item);
  renderCatalog();
  clearResult();
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
  $('result-warnings').innerHTML = result.warnings.map((warning) => {
    const message = warning.startsWith('Không có unit_cost:')
      ? 'Không có giá vốn: chỉ tối ưu doanh thu, chưa đánh giá lợi nhuận.' : warning;
    return `<div class="warning">${escapeHtml(message)}</div>`;
  }).join('');
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
    const result = await api('/recommend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (stillCurrent()) {
      renderResult(result);
      await updateCurve();
      toast(`Đã tính đề xuất cho SKU ${result.sku}. Kết quả chưa lưu vào SQL Server.`);
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
  });
  $('current-price').addEventListener('input', () => {
    $('price-source-label').textContent = 'Giá do bạn nhập';
    clearResult();
  });
  $('current-price').addEventListener('change', () => updateCurve().catch((error) => toast(error.message, true)));
  $('reset-price').addEventListener('click', () => {
    const item = productBySku(state.selectedSku);
    if (!item) return;
    $('current-price').value = Number(item.reference_price).toFixed(2);
    $('price-source-label').textContent = 'Giá tham chiếu do API mô hình trả về';
    clearResult();
    updateCurve().catch((error) => toast(error.message, true));
  });
  $('max-change').addEventListener('change', () => {
    clearResult();
    updateCurve().catch((error) => toast(error.message, true));
  });
  $('scenario-form').addEventListener('submit', submitScenario);
  $('product-search').addEventListener('input', renderCatalog);
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
  $('product-table').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-sku]');
    if (!button) return;
    await selectProduct(button.dataset.sku);
    document.querySelector('#simulator').scrollIntoView({ behavior: 'smooth' });
  });
  document.querySelectorAll('.nav-link').forEach((link) => link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((node) => node.classList.toggle('active', node === link));
  }));
}

async function initialize() {
  bindEvents();
  try {
    const [health, model, modelProducts, selected] = await Promise.all([
      api('/health'), api('/model_info'), api('/products'), selectedProducts()
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
    $('side-model-version').textContent = model.model_version;
    $('api-status').textContent = 'API GỐC ĐÃ KẾT NỐI';
    renderOverview(health);
    renderProductOptions();
    await selectProduct(state.products[0].sku);
  } catch (error) {
    $('api-status').textContent = 'CHƯA KẾT NỐI API';
    toast(`Không tải được ứng dụng: ${error.message}`, true);
  }
}

initialize();
