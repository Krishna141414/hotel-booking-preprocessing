// Shared Chart.js helpers: theme tokens, one chart style, a reference-line
// plugin and accessible data tables. Colours match src/visualize.py.
const HotelCharts = (() => {
  const FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
  const HOTEL_SLOT = { "City Hotel": 0, "Resort Hotel": 1 };
  const integer = new Intl.NumberFormat("en-US");
  const compact = new Intl.NumberFormat("en-US", { notation: "compact" });

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const tokens = () => ({
    text: cssVar("--text"),
    text2: cssVar("--text-2"),
    text3: cssVar("--text-3"),
    grid: cssVar("--grid"),
    axis: cssVar("--axis"),
    surface: cssVar("--surface"),
    raised: cssVar("--surface-raised"),
    border: cssVar("--border-strong"),
    series: [cssVar("--series-1"), cssVar("--series-2")],
  });

  const formatValue = (value, unit) => {
    if (value === null || value === undefined) return "–";
    if (unit === "%") return `${value.toFixed(1)}%`;
    if (unit === "EUR") return `${value.toFixed(0)} EUR`;
    return integer.format(value);
  };

  // Thin muted line at the overall average, labelled at the end of the line.
  const referenceLine = {
    id: "referenceLine",
    afterDatasetsDraw(chart, _args, options) {
      if (options.value === undefined || options.value === null) return;
      const { ctx, chartArea: area, scales } = chart;
      const horizontal = chart.options.indexAxis === "y";
      ctx.save();
      ctx.strokeStyle = options.color;
      ctx.fillStyle = options.color;
      ctx.lineWidth = 1;
      ctx.font = `500 11px ${FONT}`;
      ctx.beginPath();
      if (horizontal) {
        const x = scales.x.getPixelForValue(options.value);
        ctx.moveTo(x, area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText(options.label, x, area.top - 6);
      } else {
        const y = scales.y.getPixelForValue(options.value);
        ctx.moveTo(area.left, y);
        ctx.lineTo(area.right, y);
        ctx.stroke();
        // Label sits outside the plot, to the right, so it never covers a bar.
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(options.label, area.right + 6, y);
      }
      ctx.restore();
    },
  };

  const baseOptions = ({ horizontal = false, unit = "", reference = null, legend = false, extra = null }) => {
    const t = tokens();
    const valueAxis = {
      beginAtZero: true,
      grid: { color: t.grid, drawTicks: false },
      border: { display: false },
      ticks: {
        color: t.text3,
        padding: 8,
        maxRotation: 0,
        callback: (value) => (unit === "%" ? `${value}%` : compact.format(value)),
      },
    };
    const categoryAxis = {
      grid: { display: false },
      border: { color: t.axis },
      ticks: { color: t.text2, padding: 6 },
    };
    return {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: horizontal ? "y" : "x",
      animation: { duration: 300 },
      layout: { padding: { top: reference && horizontal ? 18 : 4, right: reference && !horizontal ? 84 : 8 } },
      interaction: { mode: "index", intersect: false, axis: horizontal ? "y" : "x" },
      scales: horizontal ? { x: valueAxis, y: categoryAxis } : { x: categoryAxis, y: valueAxis },
      plugins: {
        legend: {
          display: legend,
          align: "start",
          labels: {
            color: t.text2,
            usePointStyle: true,
            pointStyle: "rectRounded",
            boxWidth: 10,
            boxHeight: 10,
            padding: 16,
          },
        },
        tooltip: {
          backgroundColor: t.raised,
          borderColor: t.border,
          borderWidth: 1,
          titleColor: t.text,
          bodyColor: t.text2,
          padding: 10,
          cornerRadius: 8,
          boxPadding: 4,
          usePointStyle: true,
          callbacks: {
            label: (item) => ` ${item.dataset.label}: ${formatValue(item.raw, unit)}`,
            afterBody: extra ? (items) => extra(items[0].dataIndex) : undefined,
          },
        },
        referenceLine: reference
          ? { value: reference.value, label: reference.label, color: t.text3 }
          : { value: null },
      },
    };
  };

  // Thin bars, rounded at the data end and square at the baseline.
  const barStyle = (color) => ({
    backgroundColor: color,
    hoverBackgroundColor: color,
    borderRadius: 4,
    borderSkipped: "start",
    maxBarThickness: 24,
    categoryPercentage: 0.72,
  });

  const bar = (canvas, { labels, values, label, unit = "%", horizontal = false, reference = null, extra = null }) =>
    new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: [{ label, data: values, ...barStyle(tokens().series[0]) }] },
      options: baseOptions({ horizontal, unit, reference, extra }),
      plugins: [referenceLine],
    });

  const groupedBar = (canvas, { labels, groups, unit = "%", horizontal = true }) => {
    const t = tokens();
    return new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        // `slot` lets a group keep its colour (colour follows the entity, not the order).
        datasets: groups.map((group, i) => ({
          label: group.label,
          data: group.values,
          ...barStyle(t.series[group.slot ?? i]),
        })),
      },
      options: baseOptions({ horizontal, unit, legend: true }),
    });
  };

  // One line per hotel. Colour follows the hotel (the entity), never its position,
  // so filtering to "Resort Hotel" keeps Resort orange.
  const line = (canvas, { labels, series, unit = "%" }) => {
    const t = tokens();
    const datasets = Object.entries(series).map(([name, values]) => {
      const color = t.series[HOTEL_SLOT[name] ?? 0];
      return {
        label: name,
        data: values,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2,
        tension: 0, // straight segments: no invented peaks between months
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: color,
        pointBorderColor: t.surface,
        pointBorderWidth: 2,
      };
    });
    return new Chart(canvas, { type: "line", data: { labels, datasets }, options: baseOptions({ unit, legend: true }) });
  };

  const escapeHtml = (text) =>
    String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

  // Accessible table view of a chart's numbers (also used for result tables).
  const table = (container, columns, rows) => {
    const cell = (tag, column, content) =>
      `<${tag}${column.numeric ? ' class="num"' : ""}>${escapeHtml(content)}</${tag}>`;
    const head = columns.map((column) => cell("th", column, column.label)).join("");
    const body = rows
      .map((row) => `<tr>${columns.map((column) => cell("td", column, column.format ? column.format(row[column.key]) : row[column.key])).join("")}</tr>`)
      .join("");
    container.innerHTML = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  };

  const applyDefaults = () => {
    Chart.defaults.font.family = FONT;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = tokens().text2;
  };

  // Draw with `render` now, and again (with fresh colours) whenever the theme changes.
  const mount = (render) => {
    let charts = [];
    const draw = () => {
      charts.forEach((chart) => chart.destroy());
      applyDefaults();
      charts = render() || [];
    };
    window.addEventListener("themechange", draw);
    draw();
    return draw;
  };

  return { bar, groupedBar, line, table, mount, formatValue, integer };
})();
