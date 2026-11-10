(function () {
  var KEY_NAV = "makystats-nav-collapsed";
  var KEY_MODELS = "makystats-models-collapsed";
  var KEY_TESTS = "makystats-tests-collapsed";
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

  function applyGroup(groupId, toggleId, collapsed) {
    var g = document.getElementById(groupId);
    var t = document.getElementById(toggleId);
    if (!g || !t) return;
    g.classList.toggle("is-collapsed", !!collapsed);
    t.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }

  function onGroupPage(groupId) {
    var g = document.getElementById(groupId);
    return !!(g && g.querySelector("a.nav-item.active"));
  }

  function preferredGroupCollapsed(storageKey, groupId) {
    if (onGroupPage(groupId)) return false;
    try {
      var v = localStorage.getItem(storageKey);
      if (v === "1") return true;
      if (v === "0") return false;
    } catch (e) {}
    return false;
  }

  function setGroupCollapsed(groupId, toggleId, storageKey, collapsed) {
    applyGroup(groupId, toggleId, collapsed);
    try { localStorage.setItem(storageKey, collapsed ? "1" : "0"); } catch (e) {}
  }

  function wireToggle(toggleId, groupId, storageKey) {
    var mt = document.getElementById(toggleId);
    if (!mt) return;
    mt.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      var g = document.getElementById(groupId);
      if (!g) return;
      setGroupCollapsed(
        groupId,
        toggleId,
        storageKey,
        !g.classList.contains("is-collapsed")
      );
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    applyNav(preferredNav());
    applyGroup(
      "nav-models",
      "nav-models-toggle",
      preferredGroupCollapsed(KEY_MODELS, "nav-models")
    );
    applyGroup(
      "nav-tests",
      "nav-tests-toggle",
      preferredGroupCollapsed(KEY_TESTS, "nav-tests")
    );

    var btn = document.getElementById("menu-btn");
    if (btn) btn.addEventListener("click", toggleNav);

    var back = document.getElementById("sidebar-backdrop");
    if (back) back.addEventListener("click", function () { setNavCollapsed(true); });

    wireToggle("nav-models-toggle", "nav-models", KEY_MODELS);
    wireToggle("nav-tests-toggle", "nav-tests", KEY_TESTS);

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
