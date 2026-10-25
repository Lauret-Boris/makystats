(function () {
  var KEY_NAV = "makystats-nav-collapsed";
  var KEY_MODELS = "makystats-models-collapsed";
  var mq = window.matchMedia("(max-width: 768px)");

  function isMobile() { return mq.matches; }

  function applyNav(collapsed) {
    document.body.classList.toggle("nav-collapsed", !!collapsed);
    var btn = document.getElementById("menu-btn");
    if (btn) {
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      btn.setAttribute("aria-label", collapsed ? "Open menu" : "Close menu");
    }
  }

  function preferredNav() {
    if (isMobile()) return true;
    try {
      var v = localStorage.getItem(KEY_NAV);
      if (v === "1") return true;
      if (v === "0") return false;
    } catch (e) {}
    return false;
  }

  function setNavCollapsed(collapsed) {
    applyNav(collapsed);
    if (!isMobile()) {
      try { localStorage.setItem(KEY_NAV, collapsed ? "1" : "0"); } catch (e) {}
    }
  }

  function toggleNav() {
    setNavCollapsed(!document.body.classList.contains("nav-collapsed"));
  }

  /** Models fold: only hides .nav-group-items; toggle button always stays */
  function applyModels(collapsed) {
    var g = document.getElementById("nav-models");
    var t = document.getElementById("nav-models-toggle");
    if (!g || !t) return;
    g.classList.toggle("is-collapsed", !!collapsed);
    t.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }

  function onModelPage() {
    var g = document.getElementById("nav-models");
    return !!(g && g.querySelector("a.nav-item.active"));
  }

  function preferredModelsCollapsed() {
    // On a model page, always start expanded so the active link is visible
    if (onModelPage()) return false;
    try {
      var v = localStorage.getItem(KEY_MODELS);
      if (v === "1") return true;
      if (v === "0") return false;
    } catch (e) {}
    return false; // default expanded
  }

  function setModelsCollapsed(collapsed) {
    // Allow fold even on model pages (user request); header remains
    applyModels(collapsed);
    try { localStorage.setItem(KEY_MODELS, collapsed ? "1" : "0"); } catch (e) {}
  }

  function toggleModels() {
    var g = document.getElementById("nav-models");
    if (!g) return;
    setModelsCollapsed(!g.classList.contains("is-collapsed"));
  }

  document.addEventListener("DOMContentLoaded", function () {
    applyNav(preferredNav());
    applyModels(preferredModelsCollapsed());

    var btn = document.getElementById("menu-btn");
    if (btn) btn.addEventListener("click", toggleNav);

    var back = document.getElementById("sidebar-backdrop");
    if (back) back.addEventListener("click", function () { setNavCollapsed(true); });

    var mt = document.getElementById("nav-models-toggle");
    if (mt) {
      mt.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        toggleModels();
      });
    }

    document.querySelectorAll(".sidebar a.nav-item").forEach(function (a) {
      a.addEventListener("click", function () {
        if (isMobile()) setNavCollapsed(true);
      });
    });

    if (mq.addEventListener) {
      mq.addEventListener("change", function () { applyNav(preferredNav()); });
    } else if (mq.addListener) {
      mq.addListener(function () { applyNav(preferredNav()); });
    }
  });
})();