// Light/dark theme toggle. The saved choice is applied in <head> before paint;
// this file handles clicks and tells charts to redraw with the new colours.
(() => {
  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  const effectiveTheme = () => root.dataset.theme || (media.matches ? "dark" : "light");
  const announce = () => window.dispatchEvent(new CustomEvent("themechange"));

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch (_) {
      /* storage unavailable: the choice lasts for this page only */
    }
    announce();
  });

  media.addEventListener("change", () => {
    if (!root.dataset.theme) announce();
  });
})();
