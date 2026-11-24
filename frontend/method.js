const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 2) => new Intl.NumberFormat('vi-VN', {
  maximumFractionDigits: digits
}).format(Number(value));
const pct = (value) => `${Number(value) > 0 ? '+' : ''}${fmt(value, 2)}%`;
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
})[character]);
const methodNames = {
  hold_current: 'Giữ giá', fixed_mean: 'Giá trung bình train', heuristic: 'Quy tắc ±5%',
  forecast_grid_ensemble: 'Lưới 5 Transformer', ppo_frozen: 'PPO', ddpg: 'DDPG', sac: 'SAC'
};
const variantNames = {
  song_kernel: 'Kernel gốc theo Song', lag_half: 'Trọng số cầu gần 0,5',
  lag_selected: 'Simulator đã chọn', lag_full: 'Trọng số cầu gần 1,0',
  flat_cv_level: 'Cầu phẳng chọn trong train-CV', flat_price_response: 'Cầu phẳng theo tuần trước'
};
let overview = null;

function row(cells, className = '') {
  const tr = document.createElement('tr');
  if (className) tr.className = className;
  for (const cell of cells) {
    const td = document.createElement('td');
    td.textContent = cell;
    tr.append(td);
  }
  return tr;
}

function metric(data, split, segment, model, field = 'rmse') {
  return data.find((item) => item.split === split && item.segment === segment &&
    item.model === model)?.[field];
}

function renderForecast(data) {
  const transformer = data.forecast_test.find((item) => item.model === 'song_transformer_price');
  const naive = data.forecast_test.find((item) => item.model === 'naive_last');
  $('forecast-rows').replaceChildren(
    row(['Transformer có giá · 5 seed', fmt(transformer.test_rows, 0),
      `${fmt(transformer.rmse_mean)} ± ${fmt(transformer.rmse_sd)}`,
      `${fmt(transformer.mae_mean)} ± ${fmt(transformer.mae_sd)}`,
      transformer.rmse_mean < naive.rmse_mean ? 'Tốt hơn mốc' : 'Kém mốc dùng tuần trước']),
    row(['Dùng lượng tuần trước', fmt(naive.test_rows, 0), fmt(naive.rmse_mean),
      fmt(naive.mae_mean), 'Mốc so sánh'])
  );
}

function renderDemand(data) {
  const selected = data.demand_selected_model;
  const specs = [
    ['train_cv', 'in_support', 'Train-CV · 3 fold'],
    ['validation', 'in_support', 'Validation'],
    ['test', 'in_support', 'Test · toàn bộ giá trong miền train'],
    ['test', 'phase6_pricing_and_target_in_support', 'Test · cohort định giá Phase 6']
  ];
  $('demand-rows').replaceChildren(...specs.map(([split, segment, label]) => {
    const source = split === 'train_cv' ? data.demand_cv : data.demand_holdout;
    const selectedRmse = metric(source, split, segment, selected);
    const naiveRmse = metric(source, split, segment, 'naive_last');
    const count = metric(source, split, segment, selected, 'n');
    return row([label, fmt(count, 0), fmt(selectedRmse), fmt(naiveRmse),
      selectedRmse < naiveRmse ? 'Thấp hơn mốc' : 'Cao hơn mốc']);
  }));
  $('demand-note').textContent = `Mô hình cầu chọn trong train-CV: ${selected}. Ở test, RMSE cao hơn mốc dùng tuần trước; vì vậy kết quả tối ưu giá chưa đủ chắc để triển khai.`;
}

function renderPolicies(data) {
  const validation = data.policy_summary.filter((item) => item.split === 'validation');
  validation.sort((a, b) => b.uplift_mean_pct - a.uplift_mean_pct);
  $('policy-rows').replaceChildren(...validation.map((value) => {
    const test = data.policy_summary.find((item) => item.split === 'test' &&
      item.method === value.method);
    const role = value.method === data.global_policy.method
      ? 'Chọn chung theo validation' : 'Đối chứng';
    return row([methodNames[value.method] || value.method,
      `${pct(value.uplift_mean_pct)}${value.uplift_sd_pct > 0 ? ` ± ${fmt(value.uplift_sd_pct, 2)}` : ''}`,
      `${pct(test.uplift_mean_pct)}${test.uplift_sd_pct > 0 ? ` ± ${fmt(test.uplift_sd_pct, 2)}` : ''}`,
      role], value.method === data.global_policy.method ? 'm5-policy-winner' : '');
  }));
}

function renderRobustness(data) {
  const original = data.robustness.find((item) => item.variant === 'lag_selected');
  const flat = data.robustness.find((item) => item.variant === 'flat_cv_level');
  $('robust-original').textContent = pct(original.frozen_per_sku_rule_uplift_pct);
  $('robust-flat').textContent = pct(flat.frozen_per_sku_rule_uplift_pct);
  $('robust-rows').replaceChildren(...data.robustness.map((item) => {
    const div = document.createElement('div');
    div.className = 'm5-robust-row';
    const label = document.createElement('span');
    label.textContent = variantNames[item.variant] || item.variant;
    const value = document.createElement('strong');
    value.className = item.frozen_per_sku_rule_uplift_pct >= 0 ? 'm5-good' : 'm5-bad';
    value.textContent = pct(item.frozen_per_sku_rule_uplift_pct);
    div.append(label, value);
    return div;
  }));
}

function paperKind(paper) {
  return paper.kind.includes('Định giá') ? 'pricing' : 'forecast';
}

function renderPapers(filter) {
  const papers = overview.papers.filter((paper) => filter === 'all' ||
    paperKind(paper) === filter);
  $('paper-list').innerHTML = papers.map((paper) => {
    const safeUrl = /^https:\/\//.test(paper.source_url) ? paper.source_url : '#';
    return `<article class="m5-paper">
      <div><h3><a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(paper.title)} ↗</a></h3>
        <span class="m5-paper-kind">${escapeHtml(paper.kind)} · ${escapeHtml(paper.dataset_scope)}</span></div>
      <div><strong>${escapeHtml(paper.published_metric)}</strong>
        <p>${escapeHtml(paper.protocol)}</p><p>${escapeHtml(paper.reason)}</p>
        <em>Chỉ đối chiếu phương pháp</em></div>
    </article>`;
  }).join('');
  document.querySelectorAll('.m5-paper-filter button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.filter === filter));
  });
}

function render(data) {
  const test = data.demand_holdout;
  const selectedRmse = metric(test, 'test', 'in_support', data.demand_selected_model);
  const naiveRmse = metric(test, 'test', 'in_support', 'naive_last');
  $('verdict-title').textContent = selectedRmse > naiveRmse
    ? 'Giá đề xuất vẫn là kết quả thí nghiệm'
    : 'Chỉ số dự báo chưa chứng minh được giá tối ưu';
  $('verdict-copy').textContent = `Trên ${fmt(metric(test, 'test', 'in_support', data.demand_selected_model, 'n'), 0)} tuần test ở giá đã bán, RMSE mô hình cầu là ${fmt(selectedRmse)} so với ${fmt(naiveRmse)} của mốc dùng tuần trước. Cầu ở giá chưa bán không được quan sát, nên mức giá mới chỉ có thể chấm qua giả định mô phỏng.`;
  $('verdict-policy').textContent = methodNames[data.global_policy.method] || data.global_policy.method;
  $('verdict-cohort').textContent = `Đánh giá trên ${fmt(data.pricing_test_eligible_rows, 0)} quyết định test`;
  $('fact-series').textContent = fmt(data.series_count, 0);
  $('fact-skus').textContent = fmt(data.pricing_evaluation_skus, 0);
  $('fact-weeks').textContent = fmt(data.pricing_test_eligible_rows, 0);
  renderForecast(data);
  renderDemand(data);
  renderPolicies(data);
  renderRobustness(data);
  renderPapers('all');
}

async function loadOverview() {
  $('method-loading').hidden = false;
  $('method-loading').textContent = 'Đang đọc chỉ số thí nghiệm…';
  try {
    const response = await fetch('/api/research/method/overview');
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `API trả về mã ${response.status}.`);
    overview = payload;
    render(overview);
    $('api-status').textContent = 'Đã kết nối M5';
    $('method-loading').hidden = true;
    $('method-content').hidden = false;
  } catch (error) {
    $('api-status').textContent = 'Chưa kết nối M5';
    $('method-loading').classList.add('m5-error');
    $('method-loading').textContent = error.message;
  }
}

document.querySelectorAll('.m5-paper-filter button').forEach((button) => {
  button.addEventListener('click', () => { if (overview) renderPapers(button.dataset.filter); });
});
loadOverview();
