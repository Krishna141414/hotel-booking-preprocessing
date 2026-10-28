// Explore page: fetch chart data for the selected hotel and draw every chart.
(() => {
  const cache = new Map();
  const buttons = document.querySelectorAll("[data-hotel]");
  const summary = document.getElementById("filter-summary");
  let data = null;
  let redraw = null;

  const rateColumns = (label) => [
    { key: "label", label },
    { key: "bookings", label: "Bookings", numeric: true, format: (v) => HotelCharts.integer.format(v) },
    { key: "rate", label: "Canceled", numeric: true, format: (v) => HotelCharts.formatValue(v, "%") },
  ];

  const rows = (series) => series.labels.map((label, i) => ({ label, bookings: series.bookings[i], rate: series.rate[i] }));

  const monthlyRows = (monthly) =>
    monthly.labels.map((month, i) => {
      const row = { month };
      for (const [hotel, values] of Object.entries(monthly.series)) row[hotel] = values[i];
      return row;
    });

  const monthlyColumns = (monthly, unit) => [
    { key: "month", label: "Month" },
    ...Object.keys(monthly.series).map((hotel) => ({
      key: hotel,
      label: hotel,
      numeric: true,
      format: (v) => HotelCharts.formatValue(v, unit),
    })),
  ];

  const renderTables = () => {
    const tables = {
      lead_time: [rateColumns("Lead time (days)"), rows(data.lead_time)],
      monthly_cancel: [monthlyColumns(data.monthly_cancel, "%"), monthlyRows(data.monthly_cancel)],
      monthly_adr: [monthlyColumns(data.monthly_adr, "EUR"), monthlyRows(data.monthly_adr)],
      deposit: [rateColumns("Deposit type"), rows(data.deposit)],
      segment: [rateColumns("Market segment"), rows(data.segment)],
      countries: [rateColumns("Country"), rows(data.countries)],
      requests: [rateColumns("Special requests"), rows(data.requests)],
    };
    for (const [key, [columns, tableRows]] of Object.entries(tables)) {
      HotelCharts.table(document.querySelector(`[data-table="${key}"]`), columns, tableRows);
    }
  };

  const reference = () => ({ value: data.overall_rate, label: `overall ${data.overall_rate.toFixed(1)}%` });
  const bookingsNote = (series) => (index) => `Bookings: ${HotelCharts.integer.format(series.bookings[index])}`;

  const rateBar = (id, series, horizontal) =>
    HotelCharts.bar(document.getElementById(id), {
      labels: series.labels,
      values: series.rate,
      label: "Canceled",
      horizontal,
      reference: reference(),
      extra: bookingsNote(series),
    });

  const renderCharts = () => [
    rateBar("chart-lead", data.lead_time, false),
    HotelCharts.line(document.getElementById("chart-month-cancel"), data.monthly_cancel),
    HotelCharts.line(document.getElementById("chart-month-adr"), { ...data.monthly_adr, unit: "EUR" }),
    rateBar("chart-deposit", data.deposit, true),
    rateBar("chart-segment", data.segment, true),
    rateBar("chart-countries", data.countries, true),
    rateBar("chart-requests", data.requests, false),
  ];

  const load = async (hotel) => {
    if (!cache.has(hotel)) {
      const response = await fetch(`/api/eda?hotel=${encodeURIComponent(hotel)}`);
      if (!response.ok) throw new Error((await response.json()).error);
      cache.set(hotel, await response.json());
    }
    data = cache.get(hotel);
    summary.textContent = `${HotelCharts.integer.format(data.bookings)} bookings · ${data.overall_rate.toFixed(1)}% canceled`;
    renderTables();
    if (redraw) redraw();
    else redraw = HotelCharts.mount(renderCharts);
  };

  for (const button of buttons) {
    button.addEventListener("click", () => {
      for (const other of buttons) other.setAttribute("aria-pressed", String(other === button));
      load(button.dataset.hotel).catch((error) => (summary.textContent = `Could not load data: ${error.message}`));
    });
  }

  load(buttons[0].dataset.hotel).catch((error) => (summary.textContent = `Could not load data: ${error.message}`));
})();
