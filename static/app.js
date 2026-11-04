// Show kickoff times in the visitor's own time zone. The server renders an
// Eastern-time fallback, so the page still works without JavaScript.
document.querySelectorAll("time[data-kickoff]").forEach((el) => {
  const when = new Date(el.getAttribute("datetime"));
  if (isNaN(when)) return;
  const long = el.dataset.kickoff === "long";
  el.textContent = new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    ...(long ? { month: "short", day: "numeric" } : {}),
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(when);
});

// Sortable season-odds table: click a header to sort, click again to reverse.
document.querySelectorAll("table.odds").forEach((table) => {
  const heads = [...table.querySelectorAll("thead th")];
  heads.forEach((th, col) => {
    th.querySelector("button").addEventListener("click", () => {
      const numeric = th.querySelector("button").dataset.sort === "num";
      const dir = th.getAttribute("aria-sort") === "descending" ? "ascending" : "descending";
      heads.forEach((h) => h.setAttribute("aria-sort", "none"));
      th.setAttribute("aria-sort", dir);
      const body = table.tBodies[0];
      const value = (row) => row.cells[col].dataset.value;
      const rows = [...body.rows].sort((a, b) => {
        const cmp = numeric ? value(a) - value(b) : value(a).localeCompare(value(b));
        return dir === "ascending" ? cmp : -cmp;
      });
      body.append(...rows);
    });
  });
});

// "Copy link": the URL carries the seed, so the same season can be shared.
document.querySelectorAll("[data-copy-link]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => (btn.textContent = "Copy link"), 1500);
  });
});
