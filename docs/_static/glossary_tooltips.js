// Glossary links: dashed underline, definition on hover, click jumps to the glossary.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("a.reference.internal > span.std-term").forEach(function (span) {
    var link = span.parentElement;
    var id = decodeURIComponent((link.getAttribute("href") || "").split("#")[1] || "");
    var dt = id && document.getElementById(id);
    var dd = dt && dt.nextElementSibling;
    link.classList.add("glossary-term");
    if (dd && dd.tagName === "DD") {
      link.setAttribute("data-tip", dd.textContent.trim().replace(/\s+/g, " "));
    }
  });
});
