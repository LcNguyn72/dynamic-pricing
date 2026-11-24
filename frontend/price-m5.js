const $ = (id) => document.getElementById(id);
const number = (value, digits = 2) => new Intl.NumberFormat('vi-VN', {
  minimumFractionDigits: 0, maximumFractionDigits: digits
}).format(Number(value));
const price = (value) => `${number(value, 4)} USD`;
const signed = (value) => Math.abs(Number(value)) < .005
  ? '0%' : `${Number(value) > 0 ? '+' : ''}${number(value, 2)}%`;
const shortDate = (value) => {
  const [year, month, day] = String(value).slice(0, 10).split('-');
  return `${day}/${month}/${year}`;
};
const state = { options: null, requestRevision: 0 };

function selectedItem() {
  return state.options?.items.find((item) => item.sku === $('price-sku').value);
}

function selectedWeek() {
  return selectedItem()?.weeks.find((item) => item.week === $('price-week').value);
}

function fillSelect(node, choices, current) {
  node.replaceChildren(...choices.map(({ value, label }) => new Option(label, String(value))));
  if (choices.some((choice) => String(choice.value) === String(current))) node.value = String(current);
}

function resetResult() {
  state.requestRevision += 1;
  $('price-submit').disabled = false;
  $('price-submit').textContent = 'Tính giá đề xuất';
  $('price-empty').hidden = false;
  $('price-empty').classList.remove('m5-error');
  $('price-empty').replaceChildren();
  const title = document.createElement('strong');
  title.textContent = 'Tính giá cho lựa chọn này';
  const copy = document.createElement('p');
  copy.textContent = 'Kết quả được tính lại khi bạn nhấn nút bên trái.';
  $('price-empty').append(title, copy);
  $('price-result').hidden = true;
  $('price-comparison').hidden = true;
  $('price-details').hidden = true;
}

function updateWeek() {
  const item = selectedItem();
  const previous = $('price-week').value;
  fillSelect($('price-week'), item.weeks.map((week) => ({
    value: week.week, label: `${shortDate(week.week)} · giá trước đó ${price(week.reference_price)}`
  })), previous);
  updateReference();
}

function updateReference() {
  const item = selectedItem();
  const week = selectedWeek();
  $('price-current').min = String(item.train_min);
  $('price-current').max = String(item.train_max);
  $('price-current').placeholder = week ? String(week.reference_price) : '';
  $('price-reference-note').textContent = `Bỏ trống để dùng giá tuần trước ${price(week.reference_price)}. Miền train: ${price(item.train_min)}–${price(item.train_max)}.`;
  updateMethod();
}

function updateMethod() {
  const item = selectedItem();
  const method = $('price-method').value;
  const learned = ['ppo_frozen', 'ddpg', 'sac'].includes(method);
  $('price-seed-field').hidden = !learned;
  if (learned) fillSelect($('price-seed'), state.options.seeds.map((seed) => ({
    value: seed, label: `Seed ${seed}`
  })), $('price-seed').value || 42);
  $('price-method-note').textContent = method === 'validation_rule'
    ? `SKU này chọn: ${item.selected_method_label} trên validation mô phỏng. Seed được cố định nếu dùng chính sách học.`
    : 'Thử thuật toán trực tiếp trên cùng tuần; không thay đổi quy tắc đã chọn theo validation.';
  resetResult();
}

async function api(path, body) {
  const response = await fetch(`/api/research${path}`, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg).join('; ') : payload.detail;
    throw new Error(detail || `API trả về mã ${response.status}.`);
  }
  $('api-status').textContent = 'Đã kết nối M5';
  return payload;
}

function showError(message, loading = false) {
  const node = loading ? $('price-loading') : $('price-empty');
  node.hidden = false;
  node.classList.add('m5-error');
  node.textContent = message;
  if (!loading) {
    $('price-result').hidden = true;
    $('price-comparison').hidden = true;
    $('price-details').hidden = true;
  }
}

function chartSvg(result) {
  const rows = result.curve.filter((row) => [row.price, row.forecast_revenue,
    row.simulator_revenue].every(Number.isFinite));
  if (rows.length < 2) return '<p>Không có đủ điểm để vẽ.</p>';
  const width = 810, height = 280, left = 64, right = 18, top = 20, bottom = 40;
  const xMin = rows[0].price, xMax = rows[rows.length - 1].price;
  const revenues = rows.flatMap((row) => [row.forecast_revenue, row.simulator_revenue]);
  const smallest = Math.min(...revenues, result.forecast.revenue_recommended,
    result.simulator.revenue_recommended);
  const largest = Math.max(...revenues, result.forecast.revenue_recommended,
    result.simulator.revenue_recommended);
  const padding = Math.max(1, (largest - smallest) * .12);
  const yMin = Math.max(0, smallest - padding), yMax = largest + padding;
  const x = (value) => left + (value - xMin) / (xMax - xMin) * (width - left - right);
  const y = (value) => top + (yMax - value) / (yMax - yMin) * (height - top - bottom);
  const path = (key) => rows.map((row, index) =>
    `${index ? 'L' : 'M'}${x(row.price).toFixed(2)} ${y(row[key]).toFixed(2)}`).join(' ');
  const ticks = [0, .5, 1];
  const grid = ticks.map((step) => {
    const value = yMin + (yMax - yMin) * step;
    return `<line x1="${left}" y1="${y(value).toFixed(2)}" x2="${width - right}" y2="${y(value).toFixed(2)}" stroke="#e1eae7"/>
      <text x="${left - 8}" y="${(y(value) + 4).toFixed(2)}" text-anchor="end">${number(value, 0)}</text>`;
  }).join('');
  const xTicks = ticks.map((step) => {
    const value = xMin + (xMax - xMin) * step;
    return `<text x="${x(value).toFixed(2)}" y="${height - 8}" text-anchor="middle">${number(value, 3)}</text>`;
  }).join('');
  const currentX = x(result.reference_price).toFixed(2);
  const chosenX = x(result.recommended_price).toFixed(2);
  const samePrice = Math.abs(result.reference_price - result.recommended_price) < 1e-8;
  const currentLine = samePrice ? '' : `<line x1="${currentX}" y1="${top}" x2="${currentX}" y2="${height - bottom}" stroke="#81989c" stroke-width="2" stroke-dasharray="4 5"/>`;
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
    <g fill="#617c80" font-size="11" font-family="Be Vietnam Pro, sans-serif">${grid}${xTicks}</g>
    ${currentLine}
    <path d="${path('forecast_revenue')}" fill="none" stroke="#168d76" stroke-width="2.5" vector-effect="non-scaling-stroke"/>
    <path d="${path('simulator_revenue')}" fill="none" stroke="#cc8a5e" stroke-width="2.5" vector-effect="non-scaling-stroke"/>
    <line x1="${chosenX}" y1="${top}" x2="${chosenX}" y2="${height - bottom}" stroke="#173e43" stroke-width="2" stroke-dasharray="5 4"/>
    <circle cx="${chosenX}" cy="${y(result.forecast.revenue_recommended).toFixed(2)}" r="5" fill="#168d76" stroke="#fff" stroke-width="2"/>
    <circle cx="${chosenX}" cy="${y(result.simulator.revenue_recommended).toFixed(2)}" r="5" fill="#cc8a5e" stroke="#fff" stroke-width="2"/>
  </svg>`;
}

function tableCell(primary, secondary, className = '') {
  const cell = document.createElement('td');
  if (className) cell.className = className;
  const strong = document.createElement('strong');
  strong.textContent = primary;
  cell.append(strong);
  if (secondary) {
    const small = document.createElement('small');
    small.textContent = secondary;
    cell.append(small);
  }
  return cell;
}

function renderComparison(result) {
  const comparison = result.comparison;
  $('comparison-context').textContent = `${result.sku} · ${shortDate(result.week)} · USD`;
  $('comparison-note').textContent = comparison.validation_scores_use_historical_reference
    ? 'Validation: mức tăng doanh thu mô phỏng trung bình 5 seed so với giữ giá. Giá mới chưa có doanh thu quan sát.'
    : 'Validation: trung bình 5 seed trên giá lịch sử, không áp dụng trực tiếp cho giá vừa nhập. Giá mới chưa có doanh thu quan sát.';
  $('comparison-rows').replaceChildren(...comparison.rows.map((item) => {
    const row = document.createElement('tr');
    if (item.is_active) row.classList.add('m5-row-active');
    const method = document.createElement('td');
    const label = document.createElement('strong');
    label.textContent = item.method_label;
    method.append(label);
    if (item.is_active || item.is_selected_rule) {
      const badges = document.createElement('span');
      badges.className = 'm5-row-badges';
      if (item.is_active) {
        const active = document.createElement('em');
        active.textContent = 'Đang xem';
        badges.append(active);
      }
      if (item.is_selected_rule) {
        const selected = document.createElement('em');
        selected.className = 'm5-badge-selected';
        selected.textContent = 'Chọn trên validation';
        badges.append(selected);
      }
      method.append(badges);
    }
    if (item.seed !== null) {
      const meta = document.createElement('small');
      meta.textContent = item.seed_source === 'user_choice'
        ? `Seed ${item.seed} · bạn chọn` : `Seed ${item.seed} · đại diện validation`;
      method.append(meta);
    }
    row.append(method,
      tableCell(price(item.recommended_price), `${signed(item.price_change_pct)} giá hiện tại`, 'm5-numeric'),
      tableCell(`${number(item.forecast_revenue, 2)} USD`,
        item.forecast_uplift_pct === null ? '' : `${signed(item.forecast_uplift_pct)} so với giữ`, 'm5-numeric'),
      tableCell(`${number(item.simulator_revenue, 2)} USD`,
        item.simulator_uplift_pct === null ? '' : `${signed(item.simulator_uplift_pct)} so với giữ`, 'm5-numeric'),
      tableCell(signed(item.validation_mean_uplift_pct), 'so với giữ giá', 'm5-numeric'));
    return row;
  }));
  $('price-comparison').hidden = false;
}

function render(result) {
  $('price-empty').hidden = true;
  $('price-result').hidden = false;
  $('price-details').hidden = false;
  $('decision-method').textContent = result.method_requested === 'validation_rule'
    ? `Đã chọn trên validation · ${result.method_label}` : `Thử trực tiếp · ${result.method_label}`;
  $('decision-title').textContent = `${result.sku} · tuần ${shortDate(result.week)}`;
  $('decision-current').textContent = price(result.reference_price);
  $('decision-price').textContent = price(result.recommended_price);
  $('decision-change').textContent = `${signed(result.price_change_pct)} so với giá hiện tại`;
  $('decision-week').textContent = result.reference_source === 'user_input'
    ? 'Giá hiện tại do bạn nhập' : 'Giá hiện tại từ tuần trước';
  const selected = result.selected_rule;
  const selectedSeed = selected.seed === null ? '' : ` · seed ${selected.seed}`;
  $('decision-rule').textContent = `Quy tắc validation: ${selected.method_label}${selectedSeed}`;
  renderComparison(result);
  $('curve-range').textContent = `${price(result.price_limits.local_min)}–${price(result.price_limits.local_max)}`;
  $('price-chart').innerHTML = chartSvg(result);
  $('price-chart').setAttribute('aria-label', `Đường doanh thu dự báo và mô phỏng cho ${result.sku}. Giá hiện tại ${price(result.reference_price)}, giá đề xuất ${price(result.recommended_price)}.`);
  $('curve-caption').textContent = `Lưới ${result.curve.length} mức giá · doanh thu dự báo và mô phỏng.`;
  $('history-context').textContent = `Giá tuần trước ${price(result.context.previous_week_price)}`;
  $('history-rows').replaceChildren(...result.context.history.map((row) => {
    const tr = document.createElement('tr');
    for (const value of [`${row.weeks_before} tuần`, price(row.price), number(row.units, 1)]) {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    }
    return tr;
  }));
}

async function recommend() {
  if (!$('price-form').reportValidity()) return;
  const revision = ++state.requestRevision;
  const input = $('price-current').value.trim();
  const body = {
    sku: $('price-sku').value,
    week: $('price-week').value,
    method: $('price-method').value
  };
  if (input !== '') body.current_price = Number(input);
  if (['ppo_frozen', 'ddpg', 'sac'].includes(body.method)) body.seed = Number($('price-seed').value);
  $('price-submit').disabled = true;
  $('price-submit').textContent = 'Đang tính từ mô hình…';
  $('price-empty').hidden = false;
  $('price-empty').textContent = 'Đang suy luận và mô phỏng cho tuần đã chọn…';
  $('price-result').hidden = true;
  $('price-comparison').hidden = true;
  $('price-details').hidden = true;
  try {
    const result = await api('/price/compare', body);
    if (revision === state.requestRevision) render(result);
  } catch (error) {
    if (revision === state.requestRevision) showError(error.message);
  } finally {
    if (revision === state.requestRevision) {
      $('price-submit').disabled = false;
      $('price-submit').textContent = 'Tính giá đề xuất';
    }
  }
}

async function loadOptions() {
  $('price-loading').hidden = false;
  $('price-loading').textContent = 'Đang kết nối API M5…';
  try {
    state.options = await api('/price/options');
    if (!state.options.items.length) throw new Error('Chưa có SKU M5 đủ điều kiện kiểm thử.');
    const suggested = [...state.options.items].sort((a, b) =>
      b.validation_uplift_pct - a.validation_uplift_pct)[0];
    fillSelect($('price-sku'), state.options.items.map((item) => ({
      value: item.sku, label: `${item.sku} · ${item.selected_method_label}`
    })), new URLSearchParams(location.search).get('sku') || suggested.sku);
    fillSelect($('price-method'), [
      { value: 'validation_rule', label: 'Theo kết quả validation của sản phẩm' },
      ...state.options.methods.map((method) => ({ value: method.id, label: method.label }))
    ], 'validation_rule');
    updateWeek();
    const alternativeRules = state.options.items.filter((item) =>
      item.selected_method !== 'hold_current').length;
    $('price-scope').textContent = `${state.options.eligible_test_skus}/${state.options.pricing_evaluation_skus} SKU có ${state.options.eligible_test_weeks} tuần test trong miền giá train; ${state.options.excluded_test_weeks} tuần ngoài miền được loại. ${alternativeRules} SKU chọn quy tắc khác giữ giá. Giá có thể thay đổi tối đa ±${number(state.options.max_change_pct, 0)}%.`;
    $('price-loading').hidden = true;
    $('price-content').hidden = false;
    await recommend();
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối M5';
    showError(error.message, true);
  }
}

$('price-sku').addEventListener('change', () => { $('price-current').value = ''; updateWeek(); });
$('price-week').addEventListener('change', () => { $('price-current').value = ''; updateReference(); });
$('price-method').addEventListener('change', updateMethod);
$('price-seed').addEventListener('change', resetResult);
$('price-current').addEventListener('input', resetResult);
$('price-form').addEventListener('submit', (event) => { event.preventDefault(); recommend(); });
loadOptions();
