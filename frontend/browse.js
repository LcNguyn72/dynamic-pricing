const $ = (id) => document.getElementById(id);
const integer = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 0 });
const amount = new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const weeklyPrice = new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
const decimal = new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 2 });
const tiers = { re: 'Rẻ', vua_thap: 'Vừa thấp', vua_cao: 'Vừa cao', dat: 'Đắt' };

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
}

async function api(path) {
  const response = await fetch(`/api${path}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Yêu cầu thất bại (${response.status})`);
  return payload;
}

function shortDate(value) {
  return value ? String(value).slice(0, 10).split('-').reverse().join('/') : '—';
}

function savedTime(value) {
  const timestamp = String(value || '');
  const parsed = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(timestamp) ? timestamp : `${timestamp}Z`);
  return Number.isNaN(parsed.getTime()) ? timestamp : new Intl.DateTimeFormat('vi-VN', {
    dateStyle: 'short', timeStyle: 'short'
  }).format(parsed);
}

function signed(value) {
  return `${value > 0 ? '+' : ''}${new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 }).format(value)}%`;
}

function priceSource(value) {
  return { model_reference: 'Giá tham chiếu', user_input: 'Giá tự nhập',
    historical_weekly_median: 'Giá lịch sử' }[value] || 'Giá đã nhập';
}

const inventory = { items: [], tier: 'all' };

function renderInventory() {
  const query = $('inventory-search').value.trim().toLocaleLowerCase('vi');
  const sort = $('inventory-sort').value;
  const visible = inventory.items.filter((item) =>
    (inventory.tier === 'all' || inventory.tier === item.tier) &&
    `${item.sku} ${item.description}`.toLocaleLowerCase('vi').includes(query)
  ).sort((a, b) => {
    if (sort === 'name') return a.description.localeCompare(b.description, 'vi');
    const key = { revenue: 'revenue_estimated', weeks: 'weeks', latest: 'latest_weekly_price' }[sort];
    return (Number(b[key]) || 0) - (Number(a[key]) || 0) || a.sku.localeCompare(b.sku);
  });
  $('inventory-count').textContent = `${visible.length} / ${inventory.items.length} sản phẩm`;
  $('inventory-table').innerHTML = visible.map((item) => `<tr>
    <td class="product-cell"><strong>${escapeHtml(item.description)}</strong><small>SKU ${escapeHtml(item.sku)} · ${shortDate(item.first_week)} – ${shortDate(item.last_week)}</small></td>
    <td data-label="Nhóm giá"><span class="tier-tag ${escapeHtml(item.tier)}">${escapeHtml(tiers[item.tier] || item.tier)}</span></td>
    <td class="numeric" data-label="Tuần có bán">${integer.format(item.weeks)}</td>
    <td class="numeric" data-label="Giá tuần cuối">${weeklyPrice.format(item.latest_weekly_price)}</td>
    <td class="numeric" data-label="Lượng bán">${integer.format(item.total_units)}</td>
    <td class="numeric inventory-revenue" data-label="Doanh thu quy đổi">${amount.format(item.revenue_estimated)}</td>
    <td class="action-cell"><a class="table-action" href="/product.html?sku=${encodeURIComponent(item.sku)}">Xem lịch sử</a></td>
  </tr>`).join('');
  $('inventory-empty').hidden = visible.length > 0;
}

async function loadInventory() {
  $('inventory-state').hidden = false;
  $('inventory-wrap').hidden = true;
  $('inventory-state').textContent = 'Đang tải 20 sản phẩm…';
  try {
    const items = await api('/demo/products');
    if (!Array.isArray(items) || items.length !== 20) throw new Error('Cơ sở dữ liệu cần có đúng 20 sản phẩm đã chọn.');
    inventory.items = items;
    $('api-status').textContent = 'Đã kết nối';
    $('summary-products').textContent = integer.format(items.length);
    $('summary-weeks').textContent = integer.format(items.reduce((sum, item) => sum + Number(item.weeks), 0));
    $('summary-revenue').textContent = amount.format(items.reduce((sum, item) => sum + Number(item.revenue_estimated), 0));
    $('inventory-state').hidden = true;
    $('inventory-wrap').hidden = false;
    renderInventory();
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối';
    $('inventory-state').innerHTML = `${escapeHtml(error.message)} <button type="button" class="text-button" data-retry-inventory>Thử lại</button>`;
  }
}

function initInventory() {
  $('inventory-search').addEventListener('input', renderInventory);
  $('inventory-sort').addEventListener('change', renderInventory);
  $('inventory-tiers').addEventListener('click', (event) => {
    const button = event.target.closest('[data-tier]');
    if (!button) return;
    inventory.tier = button.dataset.tier;
    $('inventory-tiers').querySelectorAll('[data-tier]').forEach((node) => {
      node.classList.toggle('active', node === button);
      node.setAttribute('aria-pressed', String(node === button));
    });
    renderInventory();
  });
  $('inventory-state').addEventListener('click', (event) => {
    if (event.target.closest('[data-retry-inventory]')) loadInventory();
  });
  loadInventory();
}

const detail = { sku: null, weeks: [], weeklyLimit: 20 };

function renderHistoryChart(rows, key, chartId, rangeId, firstId, lastId, format, color, fill) {
  const chart = $(chartId);
  if (!rows.length) {
    chart.textContent = 'Không có dữ liệu trong năm đã chọn.';
    $(rangeId).textContent = '—';
    $(firstId).textContent = '—';
    $(lastId).textContent = '—';
    return;
  }
  const values = rows.map((row) => Number(row[key]));
  const low = Math.min(...values), high = Math.max(...values);
  const min = Math.max(0, low - (high - low || high * .1 || 1) * .12);
  const max = high + (high - low || high * .1 || 1) * .12;
  const left = 12, right = 12, top = 16, bottom = 164, width = 560;
  const points = values.map((value, index) => {
    const x = left + index * (width - left - right) / Math.max(1, rows.length - 1);
    const y = bottom - (value - min) / (max - min || 1) * (bottom - top);
    return [x, y];
  });
  const path = points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const grid = Array.from({ length: 4 }, (_, index) => {
    const y = top + index * (bottom - top) / 3;
    return `<line x1="${left}" y1="${y}" x2="${width - right}" y2="${y}" stroke="#e3ece9"/>`;
  }).join('');
  const [lastX, lastY] = points.at(-1);
  chart.innerHTML = `<svg viewBox="0 0 560 180" preserveAspectRatio="none" aria-hidden="true">${grid}
    ${fill ? `<path d="${path} L${lastX.toFixed(1)},${bottom} L${left},${bottom} Z" fill="${fill}"/>` : ''}
    <path d="${path}" fill="none" stroke="${color}" stroke-width="2.7" stroke-linejoin="round"/>
    <circle cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="4" fill="${color}" stroke="white" stroke-width="2"/>
  </svg>`;
  chart.setAttribute('aria-label', `${rows.length} tuần, từ ${shortDate(rows[0].week)} đến ${shortDate(rows.at(-1).week)}. Giá trị từ ${format.format(low)} đến ${format.format(high)}.`);
  $(rangeId).textContent = `${format.format(low)}–${format.format(high)}`;
  $(firstId).textContent = shortDate(rows[0].week);
  $(lastId).textContent = shortDate(rows.at(-1).week);
}

function renderWeekly() {
  const year = $('history-year').value;
  const visible = detail.weeks.filter((row) => year === 'all' || row.week.startsWith(year));
  renderHistoryChart(visible, 'price', 'price-history-chart', 'price-chart-range',
    'price-chart-first', 'price-chart-last', weeklyPrice, '#168d76', null);
  renderHistoryChart(visible, 'revenue_estimated', 'revenue-history-chart', 'revenue-chart-range',
    'revenue-chart-first', 'revenue-chart-last', amount, '#d68b58', '#f7e7d9');
  const tableRows = visible.slice().reverse().slice(0, detail.weeklyLimit);
  $('weekly-count').textContent = `${tableRows.length} / ${visible.length} tuần`;
  $('weekly-more').hidden = tableRows.length >= visible.length;
  $('weekly-table').innerHTML = tableRows.map((row) => `<tr>
    <td data-label="Tuần bắt đầu">${shortDate(row.week)}</td>
    <td class="numeric" data-label="Giá trung vị">${weeklyPrice.format(row.price)}</td>
    <td class="numeric ${row.price_change_pct > 0 ? 'price-up' : row.price_change_pct < 0 ? 'price-down' : ''}" data-label="Đổi từ lần bán trước">${row.price_change_pct == null ? '—' : signed(row.price_change_pct)}</td>
    <td class="numeric" data-label="Lượng bán sau lọc">${integer.format(row.qty)}</td>
    <td class="numeric inventory-revenue" data-label="Doanh thu quy đổi">${amount.format(row.revenue_estimated)}</td>
  </tr>`).join('');
}

function renderProduct(item, weeks) {
  document.title = `${item.description} | Pricing Lab`;
  $('breadcrumb-sku').textContent = item.sku;
  $('detail-title').textContent = item.description;
  $('detail-sku').textContent = item.sku;
  $('detail-tier').textContent = tiers[item.tier] || item.tier;
  $('detail-tier').className = `tier-tag ${item.tier}`;
  $('detail-period').textContent = `${shortDate(item.first_week)} – ${shortDate(item.last_week)}`;
  $('detail-latest-price').textContent = weeklyPrice.format(item.latest_weekly_price);
  $('detail-weeks').textContent = integer.format(item.weeks);
  $('detail-units').textContent = integer.format(item.total_units);
  $('detail-revenue').textContent = amount.format(item.revenue_estimated);
  $('fact-median').textContent = weeklyPrice.format(item.price_median);
  $('fact-range').textContent = `${weeklyPrice.format(item.price_min)} – ${weeklyPrice.format(item.price_max)}`;
  $('fact-reference').textContent = amount.format(item.reference_price);
  $('fact-levels').textContent = integer.format(item.n_levels);
  $('fact-elasticity').textContent = decimal.format(item.elasticity_mean);
  $('try-product').href = `/?sku=${encodeURIComponent(item.sku)}#simulator`;
  $('mobile-try').href = $('try-product').href;
  detail.weeks = weeks.map((row, index) => ({ ...row, price_change_pct: index === 0 ? null :
    (Number(row.price) / Number(weeks[index - 1].price) - 1) * 100 }));
  renderWeekly();
}

async function loadActivity() {
  $('activity-state').hidden = false;
  $('activity-wrap').hidden = true;
  try {
    const rows = await api(`/demo/products/${encodeURIComponent(detail.sku)}/recommendations?limit=100`);
    $('activity-count').textContent = `${rows.length} lần gần nhất`;
    if (!rows.length) {
      $('activity-state').textContent = 'Chưa có lần thử giá nào cho sản phẩm này.';
      return;
    }
    const actions = { lower: 'Giảm giá', raise: 'Tăng giá', keep: 'Giữ giá' };
    $('activity-table').innerHTML = rows.map((row) => `<tr>
      <td data-label="Thời điểm lưu">${escapeHtml(savedTime(row.created_at))}<small>#${integer.format(row.recommendation_id)}</small></td>
      <td data-label="Ngày mô phỏng">${shortDate(row.scenario_date)}</td>
      <td class="numeric" data-label="Giá nhập">${amount.format(row.current_price)}<small>${escapeHtml(priceSource(row.price_source))}</small></td>
      <td class="numeric" data-label="Giá đề xuất">${amount.format(row.recommended_price)}<small>${escapeHtml(actions[row.action] || row.action)}</small></td>
      <td class="numeric" data-label="Dự báo ở giá nhập">${amount.format(row.revenue_current)}</td>
      <td class="numeric" data-label="Dự báo ở giá đề xuất">${amount.format(row.revenue_recommended)}</td>
      <td class="numeric inventory-revenue" data-label="Tăng dự kiến">${signed(row.expected_gain_pct)}</td>
    </tr>`).join('');
    $('activity-state').hidden = true;
    $('activity-wrap').hidden = false;
  } catch (error) {
    $('activity-state').textContent = `Không tải được nhật ký: ${error.message}`;
  }
}

async function loadProduct() {
  const sku = new URLSearchParams(window.location.search).get('sku');
  if (!sku || !/^[A-Za-z0-9]+$/.test(sku)) {
    $('detail-state').innerHTML = 'Chọn một sản phẩm từ <a href="/products.html">danh sách 20 sản phẩm</a> để xem chi tiết.';
    return;
  }
  detail.sku = sku;
  $('detail-state').hidden = false;
  $('detail-state').textContent = 'Đang tải thông tin sản phẩm…';
  $('detail-content').hidden = true;
  try {
    const [item, weeks] = await Promise.all([
      api(`/demo/products/${encodeURIComponent(sku)}`),
      api(`/demo/products/${encodeURIComponent(sku)}/weeks`)
    ]);
    if (!Array.isArray(weeks) || !weeks.length) throw new Error('Sản phẩm chưa có dữ liệu bán theo tuần.');
    $('api-status').textContent = 'Đã kết nối';
    renderProduct(item, weeks);
    $('detail-state').hidden = true;
    $('detail-content').hidden = false;
    await loadActivity();
    if (window.location.hash === '#activity') $('activity').scrollIntoView({ block: 'start' });
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối';
    $('detail-state').innerHTML = `${escapeHtml(error.message)} <a href="/products.html">Về danh sách sản phẩm</a>`;
  }
}

if (document.body.dataset.page === 'products') initInventory();
if (document.body.dataset.page === 'product') {
  $('history-year').addEventListener('change', () => { detail.weeklyLimit = 20; renderWeekly(); });
  $('weekly-more').addEventListener('click', () => { detail.weeklyLimit += 20; renderWeekly(); });
  loadProduct();
}
